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

        self.init_weights()

        if getattr(self.config, "use_hpe", False) and self.layoutlmv3.embeddings.hpe_proj is not None:
            nn.init.zeros_(self.layoutlmv3.embeddings.hpe_proj.weight)
            nn.init.zeros_(self.layoutlmv3.embeddings.hpe_proj.bias)

        self.segment_attn_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.segment_attn_proj = nn.Linear(config.hidden_size, 1)

    def _segment_pool_and_contextualize(self, text_hidden, seg_id):
        B, L, H = text_hidden.shape
        device = text_hidden.device
        broadcast_hidden = text_hidden.clone()

        for b in range(B):
            ids = seg_id[b]
            valid = ids >= 0
            if valid.sum() == 0:
                continue

            uniq_segs = torch.unique(ids[valid], sorted=True)  # reading order
            n_seg = uniq_segs.shape[0]

            seg_vecs = torch.zeros(n_seg, H, device=device, dtype=text_hidden.dtype)
            seg_masks = []

            for i, s in enumerate(uniq_segs):
                mask = ids == s
                seg_masks.append(mask)
                token_feats = text_hidden[b, mask]  # Shape: (num_tokens_in_seg, H)
                
                # ---- ADAPTIVE ATTENTION POOLING LOGIC ----
                # Tính điểm attention cho các token trong segment
                # score shape: (num_tokens_in_seg, 1)
                score = self.segment_attn_proj(torch.tanh(self.segment_attn_query(token_feats)))
                attn_weights = torch.softmax(score, dim=0) # Chuẩn hóa trọng số tổng bằng 1
                
                # Vector đại diện segment là tổng có trọng số (weighted sum) thay vì mean pooling
                seg_vecs[i] = torch.sum(token_feats * attn_weights, dim=0)

            if self.segment_context is not None:
                # 1D order embedding theo thứ tự đọc (reading order)
                order_ids = torch.arange(n_seg, device=device).clamp(
                    max=self.segment_position_embedding.num_embeddings - 1
                )
                order_emb = self.segment_position_embedding(order_ids)

                seg_vecs_with_pos = seg_vecs + order_emb
                ctx_out = self.segment_context(seg_vecs_with_pos.unsqueeze(0)).squeeze(0)
                seg_vecs_ctx = seg_vecs + self.segment_context_gate * (ctx_out - seg_vecs)
            else:
                seg_vecs_ctx = seg_vecs

            for i, mask in enumerate(seg_masks):
                broadcast_hidden[b, mask] = seg_vecs_ctx[i]

        return broadcast_hidden
    
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

        if seg_id is not None:
            # Truyền thêm tham số bbox (chỉ lấy phần của text)
            text_hidden = self._segment_pool_and_contextualize(text_hidden, seg_id)

            # Add the is-first-token-of-segment signal so the classifier can
            # still distinguish B- from I- despite the shared pooled vector.
            is_first = torch.zeros_like(seg_id, dtype=torch.long)
            is_first[:, 0] = 0  # position 0 is always a special token ([CLS]) -> irrelevant, seg_id=-1 there anyway
            if seg_id.shape[1] > 1:
                prev = seg_id[:, :-1]
                cur = seg_id[:, 1:]
                changed = (cur != prev) & (cur >= 0)
                is_first[:, 1:] = changed.long()
            # A token whose seg_id == -1 (special/pad) is never "first of a segment".
            is_first = is_first * (seg_id >= 0).long()

            text_hidden = text_hidden + self.is_first_token_embedding(is_first)
        # if seg_id is None (e.g. an old checkpoint / different dataloader),
        # fall back to plain per-token behavior -- text_hidden is untouched.

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
