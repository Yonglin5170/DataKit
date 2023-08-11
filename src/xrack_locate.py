import cv2
import json
import torch
import numpy as np
from dataclasses import dataclass

import sys
sys.path.append('/dataset/yonglinwu/SMore/DataKit')

from utils import pb_utils

from smore_xrack.utils.timer import profile
from smore_xrack.module.module_builder import build_module
from smore_xrack.pipeline.pipeline_base import PipelineBase
from smore_xrack.pipeline.pipeline_builder import PIPELINE, build_pipeline
from smore_xrack.utils.constant import SegOutputConstants


@PIPELINE.register_module()
class LocatePipeline(PipelineBase):
    def __init__(self, 
                 loc_module_cfg: dict,
                 cropped_cfg: dict,
                 contour_filter_mode: str,
                 **kwargs):
        super().__init__()

        # 定位模型的配置
        self.preloc_module_cfg = loc_module_cfg
        self._preloc_module = None
        # 裁剪相关的配置
        self.cropped_cfg = cropped_cfg
        self.contour_filter_mode = contour_filter_mode

    @property
    def preloc_module(self):
        if self._preloc_module is None:
            self._preloc_module = build_module(self.preloc_module_cfg)
        return self._preloc_module

    def edges_loc_forward(self, img_tensor):
        preloc_module_output = self.preloc_module.forward([img_tensor])[0]
        try:
            d_rects = preloc_module_output[SegOutputConstants.CONTOURS_RECT][0]
            contours = preloc_module_output[SegOutputConstants.CONTOURS][0]
            areas = preloc_module_output[SegOutputConstants.CONTOURS_AREA][0]
            assert self.contour_filter_mode in ['first_index', 'rightmost'], \
                'contour_filter_mode only support ["first_index", "rightmost"], but got {}'.format(self.contour_filter_mode)
            if self.contour_filter_mode == 'first_index':
                index = 0
            elif self.contour_filter_mode == 'rightmost':
                index, max_x = 0, 0
                for i, d_rect in enumerate(d_rects):
                    (r_x, r_y), (r_w, r_h), angle = d_rect
                    if r_x + r_w > max_x:
                        max_x = r_x + r_w
                        index = i
            d_rect = d_rects[index]
            contour = contours[index]
            area = areas[index]

            # contour.shape: (n, 1, 2)
            w_start, h_start = np.min(contour, axis=(0, 1))
            w_end, h_end = np.max(contour, axis=(0, 1))
            return [w_start, h_start, w_end, h_end], contour, d_rect, area

        except Exception as e:
            print(e)
            return [0, 0, img_tensor.shape[1], img_tensor.shape[0]], None, None, None
    
    def crop_to_cropped_size(self, img_data, edges, edge_contour, product_type):
        w_start, h_start, w_end, h_end = edges
        print('w, h:', w_end - w_start, h_end - h_start, flush=True)

        cropped_cfg = self.cropped_cfg[product_type]
        assert cropped_cfg['anchor_point'] in ['center', 'right'], \
                'contour_filter_mode only support ["center", "right"], but got {}'.format(self.contour_filter_mode)
        offset = cropped_cfg['offset']

        if cropped_cfg['anchor_point'] == 'center':
            w_center, h_center = (w_start + w_end) // 2 + offset[0], (h_start + h_end) // 2 + offset[1]
        elif cropped_cfg['anchor_point'] == 'right':
            index = int(np.argmax(edge_contour[:, :, 0], axis=0))
            right_point = edge_contour[index][0]
            w_center, h_center = right_point[0] + offset[0], right_point[1] + offset[1]

        crop_w, crop_h = cropped_cfg['output_size']
        x1 = min(img_data.shape[1] - crop_w, max(0, w_center - crop_w // 2))
        y1 = min(img_data.shape[0] - crop_h, max(0, h_center - crop_h // 2))
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        print('new w, h:', x2 - x1, y2 - y1, flush=True)
        return (x1, y1, x2, y2)

    @profile("LocatePipeline", use_gpu=True)
    def forward(self, img_data: np.ndarray, **kwargs):
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, -1)
        img_tensor = torch.from_numpy(img_data).cuda(non_blocking=False)

        edges, edge_contour, edge_rect, edge_area = self.edges_loc_forward(img_tensor)
        edge_box = cv2.boxPoints(edge_rect)
        edge_box = np.int0(edge_box).tolist()

        crop_roi = self.crop_to_cropped_size(img_data, edges, edge_contour, kwargs.get('product_type'))
        output_dict = {
            'crop_roi': crop_roi,
            'edge_contour': edge_contour,
            'edge_box': edge_box
        }
        return output_dict


def draw(img: np.ndarray, outputs: dict):
    x1, y1, x2, y2 = outputs['crop_roi']
    edge_contour = outputs['edge_contour']
    edge_box = outputs['edge_box']
    crop_contour = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.int32)
    cv2.drawContours(img, [crop_contour], -1, color=[0, 0, 255], thickness=5)
    cv2.putText(img, 'crop_contour', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    color=[0, 0, 255], thickness=5, lineType=cv2.LINE_AA)
    cv2.drawContours(img, [np.array(edge_contour)], -1, color=[0, 255, 0], thickness=2)
    cv2.drawContours(img, [np.array(edge_box)], -1, color=[0, 255, 0], thickness=2)
    cv2.putText(img, 'edge_box', edge_box[0], cv2.FONT_HERSHEY_SIMPLEX, 2,
                    color=[0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
    return img


@dataclass
class DnnRequest(object):
    inputs: list
    input_config: str


@dataclass
class DnnResponse(object):
    outputs: list
    output_config: str


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class SMoreDnnModule:
    def __init__(self) -> None:
        self.init = False


    def ToDnnRequest(self, request) -> DnnRequest:
        if request == None:
            print('Get request failed...')
        else:
            print('Get request successfully...')
        
        dnn_inputs = []
        input_tensors = pb_utils.get_input_tensors(request)
        for input_tensor in input_tensors:
            dnn_inputs.append(input_tensor.as_numpy())
        dnn_input_config = pb_utils.get_input_config(request)
        return DnnRequest(dnn_inputs, dnn_input_config)


    def FromDnnResponse(self, dnn_response: DnnResponse):
        outputs = dnn_response.outputs
        output_config = dnn_response.output_config

        output_tensors = []
        i = 0
        for output in outputs:
            name = 'OUTPUT_' + str(i)
            i += 1
            output_tensor = pb_utils.Tensor(name, output)
            output_tensors.append(output_tensor)

        return pb_utils.InferenceResponse(
                                    output_tensors=output_tensors,
                                    output_config=output_config)
    

    def Run(self, requests):
        """`Run` MUST be implemented in every Python model. `Run`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            dnn_req = self.ToDnnRequest(request)
            dnn_resp = self.RunImpl(dnn_req)
            response = self.FromDnnResponse(dnn_resp)
            responses.append(response)
        return responses


    def Init(self, pipeline_cfg: dict):
        print("Inited config: ", pipeline_cfg)
        self.locate_pipeline = build_pipeline(pipeline_cfg)
        self.init = True


    def Version(self):
        if self.init != True:
            raise Exception("Not Initialized")

        return "1.0.1"


    def Finalize(self):
        print('Cleaning up...')
    

    # Add your code here...
    def RunImpl(self, req: DnnRequest) -> DnnResponse:
        if self.init != True:
            raise Exception("Not Initialized")

        print('input_config:', req.input_config, flush=True)
        config = json.loads(req.input_config)

        img_data = req.inputs[0]
        outputs = self.locate_pipeline.forward(img_data=img_data, product_type=config['product_type'])

        response_images = []
        if config['is_draw']:
            if len(img_data.shape) == 2:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
            result_image = draw(img_data, outputs)
            response_images.append(result_image)

        result_js_str = json.dumps(outputs, cls=NpEncoder)
        return DnnResponse(response_images, result_js_str)
