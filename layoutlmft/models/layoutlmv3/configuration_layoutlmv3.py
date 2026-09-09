# coding=utf-8
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/config.json",
    "layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/resolve/main/config.json",
    # See all LayoutLMv3 models at https://huggingface.co/models?filter=layoutlmv3
}


class LayoutLMv3Config(BertConfig):
    model_type = "layoutlmv3"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=None,
        shape_size=None,
        has_relative_attention_bias=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        has_spatial_attention_bias=False,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        visual_embed=True,
        mim=False,
        wpa_task=False,
        discrete_vae_weight_path='',
        discrete_vae_type='dall-e',
        input_size=224,
        second_input_size=112,
        device='cuda',
        # ==== THÊM MỚI: Hierarchical Position Encoding (HPE) ====
        use_hpe=False,
        max_line_id=512,
        max_block_id=512,
        hpe_embedding_size=32,
        theta_line=5,
        theta_block_x=50,
        theta_block_y=20,
        # ==== THÊM MỚI: 1D order embedding cho chuỗi segment ====
        max_segment_position=512,
        **kwargs
    ):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        ...
        self.second_input_size = second_input_size
        self.device = device
        # ==== THÊM MỚI ====
        self.use_hpe = use_hpe
        self.max_line_id = max_line_id
        self.max_block_id = max_block_id
        self.hpe_embedding_size = hpe_embedding_size
        self.theta_line = theta_line
        self.theta_block_x = theta_block_x
        self.theta_block_y = theta_block_y
        self.max_segment_position = max_segment_position
