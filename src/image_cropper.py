import os
import cv2
import copy
import argparse
import numpy as np

import sys
sys.path.append('/dataset/yonglinwu/SMore/DataKit')

from src import locate_pipeline, utils

from smore_xrack.utils.logger import get_root_logger
from smore_xrack.utils.timer import print_profile
from smore_xrack.pipeline.pipeline_builder import build_pipeline


class ImageCropper(object):
    """
    用于批量裁剪图片的类
    """
    def __init__(self, image_cropper_cfg, **kwargs):
        self.base_dir = image_cropper_cfg['base_dir']
        self.img_json_dirs = image_cropper_cfg['img_json_dirs']
        self.save_dir = image_cropper_cfg['save_dir']
        self.grayscale = image_cropper_cfg['grayscale']
        self.save_as_jpg = image_cropper_cfg['save_as_jpg']
        self.is_draw = image_cropper_cfg['is_draw']
        self.only_json = image_cropper_cfg['only_json']
        self.enable_time_printing = image_cropper_cfg['enable_time_printing']
        self.roi_mappings = image_cropper_cfg['roi_mappings']
        self.locate_pipeline_cfg = image_cropper_cfg['locate_pipeline_cfg']
        self.locate_cropped_cfg = image_cropper_cfg['locate_cropped_cfg']
        self._locate_pipeline = None
        self._cropped_params_by_product = None
        self.logger = None

    def draw(self, img_data, roi_info):
        """
        在原图上绘制roi
        """
        if len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        roi_list = roi_info['roi_list']
        contour_list = roi_info['contour_list']
        rect_list = roi_info['rect_list']
        for i, roi in enumerate(roi_list):
            x1, y1, x2, y2 = roi
            roi_contour = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2],
            ], dtype=np.int32)
            cv2.drawContours(img_data, [roi_contour], -1, color=[0, 0, 255], thickness=5)
            cv2.putText(img_data, 'roi', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color=[0, 0, 255], thickness=5, lineType=cv2.LINE_AA)
            if len(contour_list) > 0:
                cv2.drawContours(img_data, [np.array(contour_list[i])], -1, color=[0, 255, 0], thickness=2)
            if len(rect_list) > 0:
                cv2.drawContours(img_data, [np.array(rect_list[i])], -1, color=[0, 255, 0], thickness=2)
                cv2.putText(img_data, 'rect', rect_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 2,
                            color=[0, 255, 0], thickness=2, lineType=cv2.LINE_AA)
        return img_data

    def locate_pipeline_forward(self, img_data):
        if self._locate_pipeline is None:
            self._locate_pipeline = build_pipeline(self.locate_pipeline_cfg)
        outputs = self._locate_pipeline.forward(img_data=img_data)
        return outputs

    @property
    def cropped_params_by_product(self):
        if self._cropped_params_by_product is None:
            self._cropped_params_by_product = {}
            for each_cropped_cfg in self.locate_cropped_cfg:
                product_type = each_cropped_cfg['product_type']
                self._cropped_params_by_product[product_type] = each_cropped_cfg['cropped_params']
        return self._cropped_params_by_product

    @staticmethod
    def sort_contours(contours, rects, contour_sort_mode):
        """
        按指定的排序规则对contour排序
        """
        if contour_sort_mode is None:
            sorted_rects, sorted_contours = rects, contours
        elif contour_sort_mode == 'rightmost':
            def get_right_x(rect):
                # rect: (r_x, r_y), (r_w, r_h), angle
                return rect[0][0] + rect[1][0]

            # 打包并按照rect的最右边位置降序排序
            packed_data = zip(rects, contours)
            sorted_data = sorted(packed_data, key=lambda x: get_right_x(x[0]), reverse=True)
            # 解压排序后的数据
            sorted_rects, sorted_contours = zip(*sorted_data)
        return sorted_contours, sorted_rects

    def get_roi_info_by_locate_pipeline(self, img_data, product_type):
        outputs = self.locate_pipeline_forward(img_data)
        if self.enable_time_printing:
            profile_info = print_profile()
            self.logger.info(profile_info)
        contours = outputs['contours']
        rects = outputs['rects']
        cropped_params = self.cropped_params_by_product[product_type]
        roi_info = {
            'roi_list': [],
            'contour_list': [],
            'rect_list': []
        }
        for cropped_param in cropped_params:
            # 按指定的排序规则对contour排序，并根据index选取contour
            contour_sort_mode = cropped_param['contour_sort_mode']
            assert contour_sort_mode is None or contour_sort_mode in ['rightmost'], \
                'contour_sort_mode only support [None, "rightmost"], but got {}'.format(contour_sort_mode)
            contour_index = cropped_param['contour_index']
            sorted_contours, sorted_rects = self.sort_contours(contours, rects, contour_sort_mode)
            _contour = sorted_contours[contour_index]
            _rect = sorted_rects[contour_index]
            # _contour.shape: (n, 1, 2)
            # .tolist()很重要！否则保存json会出错
            x_start, y_start = np.min(_contour, axis=(0, 1)).tolist()
            x_end, y_end = np.max(_contour, axis=(0, 1)).tolist()
            x_center, y_center = (x_start + x_end) // 2, (y_start + y_end) // 2
            print('w, h:', x_end - x_start, y_end - y_start, flush=True)

            # 根据锚点的位置和偏移量计算roi中心点的位置
            anchor_position = cropped_param['anchor_position']
            anchor_offset = cropped_param['anchor_offset']
            if anchor_position == 'center':
                x_center, y_center = x_center + anchor_offset[0], y_center + anchor_offset[1]
            elif anchor_position == 'right':
                index = int(np.argmax(_contour[:, :, 0], axis=0))
                # .tolist()很重要！否则保存json会出错
                right_point = _contour[index][0].tolist()
                x_center, y_center = right_point[0] + anchor_offset[0], right_point[1] + anchor_offset[1]
            # 计算roi
            roi_w, roi_h = cropped_param['output_size']
            x1 = min(img_data.shape[1] - roi_w, max(0, x_center - roi_w // 2))
            y1 = min(img_data.shape[0] - roi_h, max(0, y_center - roi_h // 2))
            x2 = x1 + roi_w
            y2 = y1 + roi_h
            print('new w, h:', x2 - x1, y2 - y1, flush=True)
            roi_info['roi_list'].append((x1, y1, x2, y2))
            roi_info['contour_list'].append(_contour)
            roi_info['rect_list'].append(np.int0(cv2.boxPoints(_rect)).tolist())
        
        return roi_info

    def find_roi_mapping(self, img_path):
        """
        找到与img_path最匹配的roi_mapping
        """
        i = 0
        tag_num = len(self.roi_mappings[0]['tags'])
        mapping_queue = self.roi_mappings.copy()
        new_mapping_queue = []
        # 对每个tag做循环
        while i < tag_num:
            new_mapping_queue = [mapping for mapping in mapping_queue if mapping['tags'][i] in img_path]
            # 所有mapping都不匹配，则加入tag == '_else_'的mapping
            if len(new_mapping_queue) == 0:
                new_mapping_queue = [mapping for mapping in mapping_queue if mapping['tags'][i] == '_else_']
            mapping_queue = new_mapping_queue.copy()
            new_mapping_queue = []
            i += 1
        # mapping_queue中唯一的元素就是最匹配的mapping
        mapping = mapping_queue[0]
        return mapping

    def get_roi_info(self, img_path, img_data):
        """
        获取roi用于裁剪图片
        """
        roi_mapping = self.find_roi_mapping(img_path)
        product_type = roi_mapping['product_type']
        roi_list = roi_mapping['roi_list']
        roi_info = {
            'roi_list': roi_list,
            'contour_list': [],
            'rect_list': []
        }
        if roi_list is None:  # 未指定roi_list
            def get_roi_load_path(roi_index):
                """
                根据roi索引生成roi加载路径
                """
                img_save_path = self.add_index_to_filename(
                    img_path.replace(self.base_dir, self.save_dir), roi_index)
                roi_load_path = img_save_path.replace(img_path[-4:], '.txt')
                return roi_load_path
            if self.only_json:  # 只保存json
                # 加载已保存的roi信息
                roi_info['roi_list'] = []
                roi_load_path = get_roi_load_path(len(roi_info['roi_list']))
                while os.path.exists(roi_load_path):
                    lines = [line.strip() for line in utils.read_file(roi_load_path)]
                    # lines[0]: 'x1 y1 x2 y2'
                    roi = tuple(map(int, lines[0].split(' ')))
                    roi_info['roi_list'].append(roi)
                    roi_load_path = get_roi_load_path(len(roi_info['roi_list']))

            if not self.only_json or len(roi_info['roi_list']) == 0:
                # 未加载到roi信息，使用locate pipeline获取
                roi_info = self.get_roi_info_by_locate_pipeline(img_data, product_type)
        return roi_info

    @staticmethod
    def add_index_to_filename(filename, index):
        """
        在filename（不包含扩展名）后面加入'_index'
        """
        prefix, ext = os.path.splitext(filename)
        new_filename = prefix + '_%d%s' % (index, ext)
        return new_filename

    @staticmethod
    def save_roi_to_file(roi, roi_save_path):
        """
        将roi信息写入到文件
        """
        x1, y1, x2, y2 = roi
        lines = ['%d %d %d %d\n' % (x1, y1, x2, y2)]
        utils.write_to_file(lines, roi_save_path)

    def crop_image_by_roi(self, img_data, img_path, roi, roi_index):
        x1, y1, x2, y2 = roi
        # crop image by roi
        cropped_img = img_data[y1 : y2, x1 : x2]
        img_save_path = img_path.replace(self.base_dir, self.save_dir)
        img_save_path = self.add_index_to_filename(img_save_path, roi_index)
        if self.save_as_jpg:
            img_save_path = img_save_path.replace(img_save_path[-4:], '.jpg')
        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
        if not self.only_json:
            # 同时保存image和json，否则只保存json
            cv2.imwrite(img_save_path, cropped_img)
        # 保存roi信息
        roi_save_path = img_save_path.replace(img_save_path[-4:], '.txt')
        self.save_roi_to_file(roi, roi_save_path)
        return img_save_path

    def crop_json_data_by_roi(self, json_data, json_path, roi, roi_index, img_save_path):
        x1, y1, x2, y2 = roi
        new_json_data = copy.deepcopy(json_data)
        new_json_data['shapes'] = []
        # crop points in json
        for i, shape in enumerate(json_data['shapes']):
            points = np.array(shape['points'])
            x_in_roi = np.count_nonzero((points[:, 0] >= x1) & (points[:, 0] < x2)) > 0
            y_in_roi = np.count_nonzero((points[:, 1] >= y1) & (points[:, 1] < y2)) > 0
            if x_in_roi and y_in_roi:
                # all points is in roi
                points[:, 0] -= x1
                points[:, 1] -= y1
                # clip points to the right and bottom boundaries
                points[:, 0] = np.clip(points[:, 0], 0, x2 - x1)
                points[:, 1] = np.clip(points[:, 1], 0, y2 - y1)
                shape['points'] = points.tolist()
                new_json_data['shapes'].append(shape)
        new_json_data['imagePath'] = os.path.basename(img_save_path)
        new_json_data['imageHeight'] = y2 - y1
        new_json_data['imageWidth'] = x2 - x1
        # 保存json
        json_save_path = json_path.replace(self.base_dir, self.save_dir)
        json_save_path = self.add_index_to_filename(json_save_path, roi_index)
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        utils.json_dump(new_json_data, json_save_path)

    def crop_images_of_single_dir(self, img_dir, json_dir):
        log_path = os.path.join(self.save_dir, 'log.txt')
        self.logger = get_root_logger(log_path)
        # cnt = 0
        for root, dirnames, filenames in os.walk(os.path.join(self.base_dir, img_dir)):
            for filename in filenames:
                if '.jpg' not in filename and '.bmp' not in filename:
                    continue
                # if cnt >= 1:
                #     break
                # cnt += 1
                img_path = os.path.join(root, filename)
                print('img_path:', img_path, flush=True)
                if self.grayscale:
                    img_data = cv2.imread(img_path, 0)
                else:
                    img_data = cv2.imread(img_path)
                roi_info = self.get_roi_info(img_path, img_data)

                if self.is_draw:
                    # 将roi绘制在图像上，仅保存绘制后的图像，json不做crop
                    img_with_drawing = self.draw(img_data, roi_info)
                    img_save_path = img_path.replace(self.base_dir, self.save_dir)
                    if self.save_as_jpg:
                        img_save_path = img_save_path.replace(img_save_path[-4:], '.jpg')
                    os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                    cv2.imwrite(img_save_path, img_with_drawing)
                    continue

                # 裁剪图片和json
                for roi_index, roi in enumerate(roi_info['roi_list']):
                    img_save_path = self.crop_image_by_roi(img_data, img_path, roi, roi_index)

                    json_path = os.path.join(root, filename.replace(filename[-4:], '.json'))
                    if json_dir is not None:  #  指定了json的目录
                        json_path = json_path.replace('%s/' % img_dir, '%s/' % json_dir)
                    # 没有对应的json，直接跳过
                    if not os.path.exists(json_path):
                        continue
                    json_data = utils.json_load(json_path)

                    self.crop_json_data_by_roi(json_data, json_path, roi, roi_index, img_save_path)

    def crop_images(self):
        for img_dir, json_dir in self.img_json_dirs:
            print('data dir:', os.path.join(self.base_dir, img_dir))
            self.crop_images_of_single_dir(img_dir, json_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    config = utils.yaml_load(args.config_path)
    image_cropper = ImageCropper(config['image_cropper_cfg'])
    image_cropper.crop_images()