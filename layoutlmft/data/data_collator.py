import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import random
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
    training: bool = True              # THÊM MỚI

    def _generate_dynamic_segments(self, bboxes):
        """ Sinh segment_ids và seg_bbox động theo khoảng cách tọa độ """
        S = len(bboxes)
        seg_ids = [-1] * S
        seg_bboxes = []
        
        current_seg = 0
        for i in range(S):
            if bboxes[i] == [0, 0, 0, 0]: # Bỏ qua token đặc biệt/padding
                continue
            if i == 0 or bboxes[i-1] == [0, 0, 0, 0]:
                seg_ids[i] = current_seg
            else:
                box_prev = bboxes[i-1]
                box_curr = bboxes[i]
                
                # Tính độ chồng lấn trục Y và khoảng cách trục X
                y_overlap = min(box_prev[3], box_curr[3]) - max(box_prev[1], box_curr[1])
                x_dist = box_curr[0] - box_prev[2]
                
                # Heuristic: Cùng dòng ngang (chồng lấn Y > 0) và khoảng cách X không quá xa
                if y_overlap > 0 and -20 <= x_dist < 60:
                    seg_ids[i] = current_seg
                else:
                    current_seg += 1
                    seg_ids[i] = current_seg
                    
        # Tính toán lại khung Bbox bao trọn cho từng Segment mới
        for s in range(current_seg + 1):
            s_boxes = [bboxes[i] for i in range(S) if seg_ids[i] == s]
            if s_boxes:
                min_x = min(b[0] for b in s_boxes)
                min_y = min(b[1] for b in s_boxes)
                max_x = max(b[2] for b in s_boxes)
                max_y = max(b[3] for b in s_boxes)
                seg_bboxes.append([min_x, min_y, max_x, max_y])
            else:
                seg_bboxes.append([0, 0, 0, 0])
                
        return seg_ids, seg_bboxes

    def __call__(self, features):
        for f in features:
            f.pop("overflow_to_sample_mapping", None)

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0].keys() else None

        # 2. THÊM LOGIC PHÁ BỎ SEGMENT Ở ĐÂY
        if self.training and random.random() < 0.5:
            for f in features:
                if "bbox" in f:
                    new_seg_ids, new_seg_bboxes = self._generate_dynamic_segments(f["bbox"])
                    f["seg_id"] = new_seg_ids
                    f["seg_bbox"] = new_seg_bboxes

        # 3. SỬA ĐOẠN POP (LOẠI BỎ IS_FIRST HOÀN TOÀN)
        seg_id = [feature.pop("seg_id") for feature in features] if "seg_id" in features[0] else None
        
        # Xóa vĩnh viễn is_first khỏi features để không gây lỗi
        if "is_first" in features[0]:
            for feature in features:
                feature.pop("is_first", None)
                
        seg_bbox = [feature.pop("seg_bbox") for feature in features] if "seg_bbox" in features[0] else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox = "bbox" in batch
        sequence_length = len(batch["input_ids"][0]) if isinstance(batch["input_ids"], list) else batch["input_ids"].shape[1]
        
        # Padding thủ công cho Sequence
        batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        if has_bbox:
            batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
        if seg_id is not None:
            batch["seg_id"] = [s + [-1] * (sequence_length - len(s)) for s in seg_id]
            batch["is_first"] = [f + [0] * (sequence_length - len(f)) for f in is_first]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        # Padding thủ công cho Segment Bbox (2D padding)
        if seg_bbox is not None:
            max_seg = max([len(sb) for sb in seg_bbox]) if seg_bbox else 0
            padded_seg_bbox = torch.zeros((len(seg_bbox), max_seg, 4), dtype=torch.long)
            seg_mask = torch.zeros((len(seg_bbox), max_seg), dtype=torch.bool)
            for b in range(len(seg_bbox)):
                n_s = len(seg_bbox[b])
                if n_s > 0:
                    padded_seg_bbox[b, :n_s] = torch.tensor(seg_bbox[b], dtype=torch.long)
                    seg_mask[b, :n_s] = True
            batch["seg_bbox"] = padded_seg_bbox
            batch["seg_mask"] = seg_mask

        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)

        return batch