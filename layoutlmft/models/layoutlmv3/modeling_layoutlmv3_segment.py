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
from .graph_module import SpatialGraphEncoder

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

        # Cấu hình cho sequence Transformer (nhánh A)
        seg_ctx_layers = getattr(config, "segment_context_layers", 1)
        seg_ctx_heads = getattr(config, "segment_context_heads", 4)
        seg_ctx_dropout = getattr(config, "segment_context_dropout", config.hidden_dropout_prob)

        # Thêm 2 lớp Linear để chiếu feature
        self.ctx_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.gph_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Khởi tạo trọng số bằng 0 để ban đầu hoạt động như identity
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.zeros_(self.gph_proj.weight)
        nn.init.zeros_(self.gph_proj.bias)

        if seg_ctx_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=seg_ctx_heads,
                dim_feedforward=config.hidden_size * 2,
                dropout=seg_ctx_dropout,
                batch_first=True,
            )
            self.segment_context = nn.TransformerEncoder(encoder_layer, num_layers=seg_ctx_layers)
        else:
            self.segment_context = None

    
        max_pos = getattr(config, "segment_context_max_positions", 512)
        self.segment_position_embedding = nn.Embedding(max_pos, config.hidden_size)
        nn.init.normal_(self.segment_position_embedding.weight, mean=0.0, std=0.02)

        # Cấu hình GNN (nhánh B)
        self.graph_encoder = SpatialGraphEncoder(
            config.hidden_size,
            n_layers=getattr(config, "graph_layers", 2),
            n_heads=getattr(config, "graph_heads", 4),
            n_rel=8,
            dropout=getattr(config, "graph_dropout", 0.1),
        )

        self.init_weights()

    def _init_weights(self, module):
        super()._init_weights(module)
        if module is getattr(self, "ctx_proj", None) or module is getattr(self, "gph_proj", None):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def _pool_segments(self, h, seg_id, max_seg):
        B, L, H = h.shape
        valid = (seg_id >= 0)
        idx = seg_id.clamp(min=0)

        cnt = torch.zeros(B, max_seg, device=h.device, dtype=h.dtype)
        cnt.scatter_add_(1, idx, valid.to(h.dtype))

        summ = torch.zeros(B, max_seg, H, device=h.device, dtype=h.dtype)
        summ.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, H),
                        h * valid.unsqueeze(-1).to(h.dtype))

        seg_vec = summ / cnt.clamp(min=1).unsqueeze(-1)
        seg_mask = cnt > 0
        return seg_vec, seg_mask, idx, valid

    def _broadcast_back(self, h, seg_ctx, idx, valid):
        b = seg_ctx.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        return b * valid.unsqueeze(-1).to(h.dtype)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        images=None,          
        valid_span=None,      
        seg_id=None,
        seg_rel=None,     
        seg_mask=None,    
    ):
        if seg_id is not None and seg_rel is None:
            raise RuntimeError("seg_rel is None: Graph branch is skipping! "
                           "Check remove_unused_columns in TrainingArguments.")
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
        )

        sequence_output = outputs[0]
        text_len = input_ids.shape[1]
        text_hidden = sequence_output[:, :text_len]
        image_hidden = sequence_output[:, text_len:]

        if seg_id is not None and seg_rel is not None and seg_mask is not None:
            max_seg_in_batch = seg_rel.size(1)
            seg_vec, current_seg_mask, idx, valid = self._pool_segments(text_hidden, seg_id, max_seg_in_batch)

            pos = torch.arange(seg_vec.size(1), device=seg_vec.device).clamp(max=self.segment_position_embedding.num_embeddings - 1)
            ctx = self.segment_context(
                seg_vec + self.segment_position_embedding(pos),
                src_key_padding_mask=~current_seg_mask
            )

            gph = self.graph_encoder(seg_vec, seg_rel, current_seg_mask)

            seg_ctx = self.ctx_proj(ctx - seg_vec) + self.gph_proj(gph - seg_vec)

            broadcast_ctx = self._broadcast_back(text_hidden, seg_ctx, idx, valid)
            text_hidden = text_hidden + broadcast_ctx 
            
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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )