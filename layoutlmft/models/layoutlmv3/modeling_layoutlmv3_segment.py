#layoutlmft/models/layoutlmv3/modeling_layoutlmv3_segment.py
# coding=utf-8
"""
LayoutLMv3ForSegmentTokenClassification

Core idea (grounded in error analysis on FUNSD + CORD):
  - Segment self-consistency is already ~98-99% solved by the base model
    (confirmed empirically) -> a consistency REGULARIZER has little to gain.
  - The real errors are (a) whole segments classified wrong as a unit
    (esp. long free-text spans dropped entirely via BIO "drift"), and
    (b) confusions that depend on the NEIGHBORING segment's role
    (HEADER vs QUESTION on FUNSD; parent vs sub-item on CORD).
  - Fix: pool each segment's token hidden states into one vector, run a
    tiny Transformer encoder over the SEQUENCE of segment vectors (reading
    order) so adjacent segments exchange information, then broadcast the
    context-enriched vector back to every token in the segment before the
    (unchanged) token classifier.
  - To keep the existing BIO scheme / seqeval / compute_metrics pipeline
    100% unchanged, we do NOT collapse labels to entity-type-only. Instead
    we add a tiny learned "is-first-token-of-segment" embedding so the
    (otherwise identical) broadcast vector can still support the B-/I-
    distinction at the classifier.

This class does NOT touch attention, does NOT build any graph/hypergraph,
and does NOT modify the pretrained backbone. It only replaces what the
token classifier head "sees" for tokens inside multi-token segments -- an
orthogonal mechanism to HGA / GraphLayoutLM.
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

from .modeling_layoutlmv3 import (
    LayoutLMv3ClassificationHead,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
)

class LayoutLMv3ForSegmentTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # ---- NEW: lightweight inter-segment context module ----
        # Config knobs (optional; safe defaults if not set on the config object).
        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)

        if seg_ctx_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=seg_ctx_heads,
                dim_feedforward=config.hidden_size * 2,
                dropout=seg_ctx_dropout,
                batch_first=True,
            )
            self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=seg_ctx_layers)
            self.segment_context_gate = nn.Parameter(torch.zeros(config.hidden_size))
            # ĐÃ XÓA 1D EMBEDDING Ở ĐÂY
        else:
            self.segment_context = None
            self.segment_context_gate = None

        # Small embedding so the classifier can still tell "first token of the
        # segment" (-> should predict B-xxx) apart from the rest (-> I-xxx),
        # even though every token in the segment otherwise shares one pooled
        # vector. Initialized near zero so early training resembles the
        # unmodified baseline.
        self.is_first_token_embedding = nn.Embedding(2, config.hidden_size)
        nn.init.normal_(self.is_first_token_embedding.weight, mean=0.0, std=0.02)

        # ==== THÊM MỚI: khôi phục 1D order embedding cho chuỗi segment
        # (đã bị xóa - xem comment "ĐÃ XÓA 1D EMBEDDING Ở ĐÂY" ở bản gốc,
        # nhưng báo cáo mục 3.3 vẫn mô tả có "Positional Embedding 1D") ====
        if self.segment_context is not None:
            self.segment_position_embedding = nn.Embedding(
                getattr(config, "max_segment_position", 512), config.hidden_size
            )
            nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.segment_position_embedding = None

        self.token_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.init_weights()

        if getattr(self.config, "use_hpe", False) and self.layoutlmv3.embeddings.hpe_proj is not None:
            nn.init.normal_(self.layoutlmv3.embeddings.hpe_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.layoutlmv3.embeddings.hpe_proj.bias)

        self.segment_attn_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.segment_attn_proj = nn.Linear(config.hidden_size, 1)

        nn.init.normal_(self.segment_attn_query.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.segment_attn_query.bias)
        nn.init.zeros_(self.segment_attn_proj.weight)
        nn.init.zeros_(self.segment_attn_proj.bias)

        # ---- NEW (Bước 2): auxiliary segment-level loss ----
        # Vấn đề: aux/token gate bị "kẹt" ở ~0 vì nhánh classifier chính (trên
        # text_hidden thô) đã đủ sức fit train loss một mình -> không có áp
        # lực gradient nào buộc segment_context phải học điều gì hữu ích.
        # Fix: gắn thêm 1 classifier NHỎ ngay trên seg_vecs_ctx (vector đã
        # qua Transformer liên-segment, TRƯỚC khi bị token_gate làm loãng),
        # dự đoán LOẠI THỰC THỂ (entity type, bỏ tiền tố B-/I-) của cả segment.
        # Loss này cộng thêm vào loss chính -> gradient đi thẳng vào
        # segment_context_gate + segment_context + segment_attn_* mà KHÔNG
        # cần đợi token_gate mở trước. Đây là con đường tắt để nhánh mới
        # buộc phải học, độc lập với nhánh baseline.
        id2label = getattr(config, "id2label", None)
        if id2label is not None and self.segment_context is not None:
            def _entity_type(label_name):
                # "B-QUESTION"/"I-QUESTION" -> "QUESTION"; "O" -> "O"
                return label_name.split("-", 1)[1] if "-" in label_name else label_name

            type_names = sorted({_entity_type(name) for name in id2label.values()})
            self.entity_type2id = {t: i for i, t in enumerate(type_names)}
            self.num_entity_types = len(type_names)

            max_label_id = max(int(k) for k in id2label.keys())
            label_id2type_id = torch.full((max_label_id + 1,), -100, dtype=torch.long)
            for lid, name in id2label.items():
                label_id2type_id[int(lid)] = self.entity_type2id[_entity_type(name)]
            # buffer (không phải Parameter): không train, chỉ để tra cứu, tự
            # theo device của model khi .to(device)/.cuda() được gọi.
            self.register_buffer("label_id2type_id", label_id2type_id, persistent=False)

            self.segment_aux_classifier = nn.Linear(config.hidden_size, self.num_entity_types)
            nn.init.normal_(self.segment_aux_classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.segment_aux_classifier.bias)
            self.segment_aux_loss_weight = getattr(config, "segment_aux_loss_weight", 0.3)
        else:
            # Không có id2label hợp lệ, hoặc segment_context đã tắt (seg_ctx_layers=0)
            # -> bỏ qua an toàn, không aux loss, hành vi y hệt trước đây.
            self.segment_aux_classifier = None

    def _segment_pool_and_contextualize(self, text_hidden, seg_id, text_bbox, is_first=None, labels=None):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        if is_first is not None:
            text_hidden = text_hidden + self.is_first_token_embedding(is_first)

        broadcast_hidden = text_hidden.clone()

        # NEW (Bước 2): gom logits/nhãn phụ cấp-segment qua toàn batch để
        # tính 1 CrossEntropyLoss duy nhất ở cuối hàm (rẻ hơn nhiều so với
        # tính loss riêng từng segment).
        all_aux_logits = []
        all_aux_targets = []

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)  # reading order
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_bboxes = torch.zeros(n_seg, 4, device=device, dtype=torch.long) # Hộp bao ngoài cho không gian
            seg_masks = []

            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                token_feats = text_hidden[b, mask]
                
                # Bbox của segment: Lấy tọa độ min (trái, trên) và max (phải, dưới)
                seg_box = text_bbox[b, mask]
                seg_bboxes[i, 0] = seg_box[:, 0].min()
                seg_bboxes[i, 1] = seg_box[:, 1].min()
                seg_bboxes[i, 2] = seg_box[:, 2].max()
                seg_bboxes[i, 3] = seg_box[:, 3].max()

                # Attention Pooling
                score = self.segment_attn_proj(torch.tanh(self.segment_attn_query(token_feats)))
                attn_weights = torch.softmax(score, dim=0) 
                seg_vecs[i] = torch.sum(token_feats * attn_weights, dim=0)

            if self.segment_context is not None:
                order_ids = torch.arange(n_seg, device=device).clamp(max=self.segment_position_embedding.num_embeddings - 1)
                order_emb = self.segment_position_embedding(order_ids)

                # Dùng chính bộ mã hoá không gian tuyệt đối 2D của LayoutLMv3 cho segment!
                # Điều này giúp mạng Segment Context hiểu chính xác tương quan gần/xa, trên/dưới.
                spatial_emb = self.layoutlmv3.embeddings._calc_spatial_position_embeddings(seg_bboxes.unsqueeze(0)).squeeze(0)

                # Cộng cả vector thứ tự đọc và tọa độ không gian vào đại diện của segment
                seg_vecs_with_pos = seg_vecs + order_emb + spatial_emb
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)
                seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)
            else:
                seg_vecs_ctx = seg_vecs

            # NEW (Bước 2): supervise seg_vecs_ctx trực tiếp bằng loại thực
            # thể (entity type) của segment, lấy majority-vote trên các
            # token hợp lệ (label != -100) trong segment đó.
            if self.segment_aux_classifier is not None and labels is not None:
                seg_targets = torch.full((n_seg,), -100, dtype=torch.long, device=device)
                for i, mask in enumerate(seg_masks):
                    seg_label_ids = labels[b, mask]
                    valid_lbl = seg_label_ids[seg_label_ids != -100]
                    if valid_lbl.numel() > 0:
                        type_ids = self.label_id2type_id.to(device)[valid_lbl]
                        seg_targets[i] = torch.mode(type_ids).values
                valid_seg = seg_targets != -100
                if valid_seg.any():
                    aux_logits = self.segment_aux_classifier(seg_vecs_ctx[valid_seg])
                    all_aux_logits.append(aux_logits)
                    all_aux_targets.append(seg_targets[valid_seg])

            for i, mask in enumerate(seg_masks):
                # THAY ĐỔI LỚN (Priority 2): Sử dụng Residual Injection thay vì Overwrite
                broadcast_hidden[b, mask] = text_hidden[b, mask] + self.token_gate * (seg_vecs_ctx[i] - text_hidden[b, mask])

        aux_loss = None
        if self.segment_aux_classifier is not None and len(all_aux_logits) > 0:
            aux_logits_cat = torch.cat(all_aux_logits, dim=0)
            aux_targets_cat = torch.cat(all_aux_targets, dim=0)
            aux_loss = CrossEntropyLoss()(aux_logits_cat, aux_targets_cat)

        return broadcast_hidden, aux_loss
    
    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        valid_span=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        seg_id=None,  # NEW input: (batch, text_seq_len), see docstring above
        is_first=None,
        line_id=None,      # ==== THÊM MỚI ====
        block_id=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            valid_span=valid_span,
            line_id=line_id,      # ==== THÊM MỚI ====
            block_id=block_id,
        )

        sequence_output = outputs[0]  # (B, text_len + image_len, H)
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len, :]
        image_hidden = sequence_output[:, text_len:, :]

        aux_seg_loss = None
        if seg_id is not None:
            text_bbox = bbox[:, :text_len, :]
            # [SỬA]: Truyền thẳng is_first + labels vào bên trong hàm
            text_hidden, aux_seg_loss = self._segment_pool_and_contextualize(
                text_hidden, seg_id, text_bbox, is_first=is_first, labels=labels
            )
            # Bỏ phần cộng is_first ở bên ngoài đi

        if image_hidden.shape[1] > 0:
            pooled_sequence = torch.cat([text_hidden, image_hidden], dim=1)
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
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # NEW (Bước 2): cộng thêm aux loss cấp-segment.
            # Đây chính là "áp lực gradient" ép segment_context_gate,
            # segment_context và segment_attn_* phải học, thay vì bị bỏ
            # quên ở gần 0 như trước.
            if aux_seg_loss is not None:
                loss = loss + self.segment_aux_loss_weight * aux_seg_loss
                self._last_aux_seg_loss = aux_seg_loss.detach()  # để CustomTrainer.log() đọc, xem bước log bên dưới

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    