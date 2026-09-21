import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy

from typing import NewType
InputDataClass = NewType("InputDataClass", Any)

def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span

@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    structural_mask_prob: float = 0.1  # THÊM MỚI
    training: bool = True              # THÊM MỚI

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0].keys() else None

        # --- BÓC TÁCH CÁC TRƯỜNG ĐỒ THỊ ---
        edge_src = [feature.pop("edge_src") for feature in features] if "edge_src" in features[0] else None
        edge_dst = [feature.pop("edge_dst") for feature in features] if "edge_dst" in features[0] else None
        edge_rel = [feature.pop("edge_rel") for feature in features] if "edge_rel" in features[0] else None
        n_seg = [feature.pop("n_seg") for feature in features] if "n_seg" in features[0] else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # --- XỬ LÝ STRUCTURAL DROPOUT ---
        if self.structural_mask_prob > 0 and self.training and "seg_id" in batch:
            B = batch["input_ids"].size(0)
            for b in range(B):
                sid = batch["seg_id"][b]
                uniq = sid[sid >= 0].unique()
                if len(uniq) < 3:
                    continue
                n = max(1, int(self.structural_mask_prob * len(uniq)))
                chosen = uniq[torch.randperm(len(uniq))[:n]]
                hit = torch.isin(sid, chosen)
                batch["input_ids"][b][hit] = self.tokenizer.mask_token_id

        # --- DỰNG MA TRẬN ĐỒ THỊ ---
        if edge_src is not None:
            B = len(features)
            S = max(n_seg) if n_seg else 1
            rel_mat = torch.zeros((B, S, S), dtype=torch.long)
            seg_mask_tensor = torch.zeros((B, S), dtype=torch.bool)
            
            for b in range(B):
                if len(edge_src[b]) > 0:
                    rel_mat[b, edge_src[b], edge_dst[b]] = torch.tensor(edge_rel[b], dtype=torch.long)
                seg_mask_tensor[b, :n_seg[b]] = True
                
            batch["seg_rel"] = rel_mat
            batch["seg_mask"] = seg_mask_tensor

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        has_seg_id_input = "seg_id" in features[0]  # NEW: for LayoutLMv3ForSegmentTokenClassification
        padding_idx=self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]
            if has_seg_id_input:
                # -1 = "not part of any segment" (padding / special tokens),
                # must NOT collide with a real segment id (which start at 0).
                batch["seg_id"] = [seg + [-1] * (sequence_length - len(seg)) for seg in batch["seg_id"]]

        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]
            if has_seg_id_input:
                batch["seg_id"] = [[-1] * (sequence_length - len(seg)) + seg for seg in batch["seg_id"]]

        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (sequence_length - len(batch['segment_ids'][i])) + [
                    batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)

        return batch
