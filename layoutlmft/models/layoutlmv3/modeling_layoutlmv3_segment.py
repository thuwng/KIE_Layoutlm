import torch
import torch.nn as nn
import math
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)

class SegmentSpatialAttention(nn.Module):
    """ Cải tiến 2: 2D Spatial Bias Segment Transformer """
    def __init__(self, hidden_size, num_heads=8, max_dist=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # Bảng look-up khoảng cách x và y
        self.x_bias = nn.Embedding(max_dist * 2 + 1, num_heads)
        self.y_bias = nn.Embedding(max_dist * 2 + 1, num_heads)
        self.max_dist = max_dist

    def forward(self, x, seg_bbox, mask):
        B, S, H = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Dot product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # [B, H, S, S]

        # Tính khoảng cách tâm (Center Distance) cho spatial bias
        cx = (seg_bbox[:, :, 0] + seg_bbox[:, :, 2]) / 2 # [B, S]
        cy = (seg_bbox[:, :, 1] + seg_bbox[:, :, 3]) / 2
        
        # Scale về bucket (giả sử tọa độ 0-1000, chia lưới 20 pixel/bucket)
        cx_b = (cx / 20).long()
        cy_b = (cy / 20).long()
        
        dx = (cx_b.unsqueeze(2) - cx_b.unsqueeze(1)).clamp(-self.max_dist, self.max_dist) + self.max_dist
        dy = (cy_b.unsqueeze(2) - cy_b.unsqueeze(1)).clamp(-self.max_dist, self.max_dist) + self.max_dist
        
        bias_x = self.x_bias(dx).permute(0, 3, 1, 2) # [B, H, S, S]
        bias_y = self.y_bias(dy).permute(0, 3, 1, 2)
        
        attn = attn + bias_x + bias_y
        
        # Masking
        attn_mask = mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, H)
        return self.proj(out)

class MultimodalSegmentEncoderLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = SegmentSpatialAttention(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size), nn.Dropout(0.1)
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, seg_bbox, mask):
        x = x + self.attn(self.ln1(x), seg_bbox, mask)
        x = x + self.ffn(self.ln2(x))
        return x

class LayoutLMv3ForSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # Cải tiến 1: Fusion layer cho Text + Vision
        self.multimodal_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        # Cải tiến 2: Segment Context Encoder
        self.segment_encoder = MultimodalSegmentEncoderLayer(config.hidden_size)
        
        # Cải tiến 3: BIO-aware Gating
        self.bio_embed = nn.Embedding(2, config.hidden_size)
        self.gate_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        torch.nn.init.constant_(self.gate_linear.bias, 3.0)

        self.init_weights()

    def _pool_text(self, h, seg_id, max_seg):
        B, L, H = h.shape
        valid = (seg_id >= 0)
        idx = seg_id.clamp(min=0)
        cnt = torch.zeros(B, max_seg, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(1, idx, valid.to(h.dtype))
        summ = torch.zeros(B, max_seg, H, device=h.device, dtype=h.dtype)
        summ.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, H), h * valid.unsqueeze(-1).to(h.dtype))
        return summ / cnt.clamp(min=1).unsqueeze(-1)

    def _pool_vision(self, image_hidden, seg_bbox, seg_mask):
        B, S, _ = seg_bbox.shape
        g = 14  # Giả định lưới 14x14 của LayoutLMv3
        cell = 1000.0 / g
        
        # Tạo các điểm ranh giới của lưới
        edges = torch.arange(g + 1, device=seg_bbox.device).float() * cell
        lo, hi = edges[:-1], edges[1:]
        
        b = seg_bbox.float()
        
        # So sánh tọa độ bbox với lưới
        ix = (b[..., 0:1] < hi) & (b[..., 2:3] >= lo)  # B, S, g
        iy = (b[..., 1:2] < hi) & (b[..., 3:4] >= lo)  # B, S, g
        
        # Tạo mask giao điểm (intersection mask)
        m = (iy.unsqueeze(-1) & ix.unsqueeze(-2)).flatten(2).float() * seg_mask.unsqueeze(-1)
        
        # Pool đặc trưng và trung bình hóa, clamp để tránh chia cho 0
        pooled = (m @ image_hidden) / m.sum(-1, keepdim=True).clamp(min=1)
        
        return pooled

    def forward(
        self,
        input_ids=None, bbox=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
        images=None, valid_span=None, seg_id=None, seg_bbox=None, seg_mask=None, is_first=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, images=images, valid_span=valid_span,
        )

        sequence_output = outputs[0]
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len]
        
        # Nếu có segment info
        if seg_id is not None and seg_bbox is not None and seg_mask is not None:
            max_seg = seg_bbox.size(1)
            
            # Cải tiến 1: Multimodal Pooling
            t_pool = self._pool_text(text_hidden, seg_id, max_seg)
            if images is not None and sequence_output.shape[1] > text_len:
                image_hidden = sequence_output[:, text_len+1:] # +1 để bỏ qua visual [CLS]
                v_pool = self._pool_vision(image_hidden, seg_bbox, seg_mask)
            else:
                v_pool = torch.zeros_like(t_pool)
            
            seg_vec = self.multimodal_fusion(torch.cat([t_pool, v_pool], dim=-1))
            
            active_seg_mask = seg_mask

            # Cải tiến 2: 2D Bias Transformer
            seg_ctx = self.segment_encoder(seg_vec, seg_bbox, active_seg_mask)

            # Khôi phục vector từ Segment về Token
            idx = seg_id.clamp(min=0)
            valid = (seg_id >= 0)
            broadcast_ctx = seg_ctx.gather(1, idx.unsqueeze(-1).expand(-1, -1, text_hidden.size(-1)))
            broadcast_ctx = broadcast_ctx * valid.unsqueeze(-1).to(text_hidden.dtype)
            
            # Cải tiến 3: BIO-Aware Gating Fusion
            if is_first is not None:
                bio_bias = self.bio_embed(is_first)
                broadcast_ctx = broadcast_ctx + bio_bias
            
            gate_input = torch.cat([text_hidden, broadcast_ctx], dim=-1)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            
            # Fuse mềm mại, giữ lại đặc trưng từ vựng
            text_hidden = gate * text_hidden + (1 - gate) * broadcast_ctx
            
        if images is not None and sequence_output.shape[1] > text_len:
            pooled_sequence = torch.cat([text_hidden, sequence_output[:, text_len:]], dim=1)
        else:
            pooled_sequence = text_hidden

        pooled_sequence = self.dropout(pooled_sequence)
        logits = self.classifier(pooled_sequence)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)