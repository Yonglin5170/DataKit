import torch
import numpy as np

from smore_xrack.utils.timer import profile
from smore_xrack.module.module_builder import build_module
from smore_xrack.pipeline.pipeline_builder import build_pipeline
from smore_xrack.pipeline.pipeline_base import PipelineBase
from smore_xrack.pipeline.pipeline_builder import PIPELINE
from smore_xrack.utils.constant import SegOutputConstants

USE_GPU = True

@PIPELINE.register_module()
class LocatePipeline(PipelineBase):
    def __init__(self, loc_module_cfg: dict, **kwargs):
        super().__init__()
        # 定位模型的配置
        self.loc_module_cfg = loc_module_cfg
        self._loc_module = None

    @property
    def loc_module(self):
        if self._loc_module is None:
            self._loc_module = build_module(self.loc_module_cfg)
        return self._loc_module

    @profile("LocatePipeline", use_gpu=USE_GPU)
    def forward(self, img_data: np.ndarray, **kwargs):
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, -1)
        img_tensor = torch.from_numpy(img_data)
        if USE_GPU:
            img_tensor = img_tensor.cuda(non_blocking=False)
        loc_module_output = self.loc_module.forward([img_tensor])[0]
        output_dict = {
            'contours': loc_module_output[SegOutputConstants.CONTOURS][0],
            'rects': loc_module_output[SegOutputConstants.CONTOURS_RECT][0],
        }
        return output_dict


def build_locate_pipeline(cfg):
    assert 'type' in cfg, f"Not found 'type' in cfg: {cfg}"
    assert cfg['type'] == 'LocatePipeline', \
        f"Wrong 'type' in cfg, expected 'LocatePipeline', got '{cfg['type']}'"
    return build_pipeline(cfg)