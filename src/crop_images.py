import os
import cv2
import copy
import json
import yaml
import argparse
import numpy as np

from xrack_locate import SMoreDnnModule, DnnRequest
from smore_xrack.utils.logger import get_root_logger
from smore_xrack.utils.timer import print_profile


# def get_product_locations(dnn_module, base_dir, img_dir, save_dir, product_type_func, grayscale=True, enable_time_printing=True):
#     print('img_dir:', img_dir)
#     log_path = os.path.join(save_dir, 'log.txt')
#     logger = get_root_logger(log_path)
#     for root, dirnames, filenames in os.walk(os.path.join(base_dir, img_dir)):
#         for filename in filenames:
#             if '.jpg' not in filename and '.bmp' not in filename:
#                 continue
#             img_path = os.path.join(root, filename)
#             if grayscale:
#                 img_data = cv2.imread(img_path, 0)
#             else:
#                 img_data = cv2.imread(img_path)

#             product_type = product_type_func(img_path)

#             req = DnnRequest([img_data], json.dumps({"product_type": product_type, "is_draw": True}))
#             try:
#                 rsp = dnn_module.RunImpl(req)
#             except:
#                 print('error path:', img_path)
#                 raise
#             if enable_time_printing:
#                 profile_info = print_profile()
#                 logger.info(profile_info)

#             save_path = img_path.replace(base_dir, save_dir)
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             cv2.imwrite(save_path, rsp.outputs[0])

#     dnn_module.Finalize()


def add_str_to_filename(filename, s):
    prefix, ext = os.path.splitext(filename)
    new_filename = prefix + '%s%s' % (s, ext)
    return new_filename


def crop_images(base_dir, img_dir, json_dir, save_dir, get_roi_func, grayscale=False, save_as_jpg=False, is_draw=False):
    log_path = os.path.join(save_dir, 'log.txt')
    logger = get_root_logger(log_path)
    # cnt = 0
    for root, dirnames, filenames in os.walk(os.path.join(base_dir, img_dir)):
        for filename in filenames:
            if '.jpg' not in filename and '.bmp' not in filename:
                continue
            # if cnt >= 1:
            #     break
            # cnt += 1
            img_path = os.path.join(root, filename)
            print('img_path:', img_path)
            if grayscale:
                img = cv2.imread(img_path, 0)
            else:
                img = cv2.imread(img_path)
            # roi_collection: roi列表，rsp_image：原图像+roi绘制结果
            roi_collection, rsp_image = get_roi_func(img_path, img, logger, is_draw)

            for roi_index, roi in enumerate(roi_collection):
                x1, y1, x2, y2 = roi
                if not is_draw:
                    # crop image by roi
                    cropped_img = img[y1 : y2, x1 : x2]
                else:
                    cropped_img = rsp_image
                img_save_path = add_str_to_filename(img_path.replace(base_dir, save_dir), '_%d' % roi_index)
                if save_as_jpg:
                    img_save_path = img_save_path.replace(img_save_path[-4:], '.jpg')
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(img_save_path, cropped_img)

                json_path = os.path.join(root, filename.replace(filename[-4:], '.json'))
                if json_dir is not None:  #  指定了json的目录
                    json_path = json_path.replace('%s/' % img_dir, '%s/' % json_dir)
                if not os.path.exists(json_path):
                    continue

                with open(json_path, encoding='utf-8') as f:
                    json_data = json.load(f)
                # print('before:', json_data['shapes'])
                new_json_data = copy.deepcopy(json_data)
                new_json_data['shapes'] = []
                # crop points in json
                for i, shape in enumerate(json_data['shapes']):
                    points = np.array(shape['points'])
                    x_in_roi = np.count_nonzero((points[:, 0] >= x1) & (points[:, 0] < x2)) > 0
                    y_in_roi = np.count_nonzero((points[:, 1] >= y1) & (points[:, 1] < y2)) > 0
                    if x_in_roi and y_in_roi:
                        points[:, 0] -= x1
                        points[:, 1] -= y1
                        shape['points'] = points.tolist()
                        new_json_data['shapes'].append(shape)
                new_json_data['imagePath'] = os.path.basename(img_save_path)
                new_json_data['imageHeight'] = y2 - y1
                new_json_data['imageWidth'] = x2 - x1
                # print('after:', new_json_data['shapes'])
                json_save_path = add_str_to_filename(json_path.replace(base_dir, save_dir), '_%d' % roi_index)
                os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(new_json_data, f, indent=4)


def crop_images_by_config(config_path, grayscale=False, is_draw=False, enable_time_printing=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    crop_cfg = config['crop_cfg']
    base_dir = crop_cfg['base_dir']
    img_json_dirs = crop_cfg['img_json_dirs']
    save_dir = crop_cfg['save_dir']
    save_as_jpg = crop_cfg['save_as_jpg']
    product_type_mapping = crop_cfg['product_type_mapping']
    locate_pipeline_cfg = crop_cfg['locate_pipeline_cfg']

    dnn_module = SMoreDnnModule()
    if locate_pipeline_cfg is not None:
        dnn_module.Init(locate_pipeline_cfg)

    def get_roi_func(img_path, img_data, logger, is_draw=False):
        # 找到与img_path最匹配的mapping
        i = 0
        tag_num = len(product_type_mapping[0]['tags'])
        mapping_queue = product_type_mapping.copy()
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
        product_type = mapping['product_type']
        roi_collection = mapping['roi_collection']
        rsp_image = None
        if roi_collection is None:
            req = DnnRequest([img_data], json.dumps({"product_type": product_type, "is_draw": is_draw}))
            try:
                rsp = dnn_module.RunImpl(req)
            except:
                print('error path:', img_path)
                raise
            if enable_time_printing:
                profile_info = print_profile()
                logger.info(profile_info)
            if is_draw:
                rsp_image = rsp.outputs[0]
            outputs = json.loads(rsp.output_config)
            roi_collection = [outputs['crop_roi']]
        return roi_collection, rsp_image


    for img_dir, json_dir in img_json_dirs:
        print('data dir:', os.path.join(base_dir, img_dir))
        crop_images(base_dir, img_dir, json_dir, save_dir, get_roi_func, grayscale=grayscale,
                    save_as_jpg=save_as_jpg, is_draw=is_draw)

    dnn_module.Finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--is_draw', action='store_true')
    parser.add_argument('--enable_time_printing', action='store_true')
    args = parser.parse_args()

    crop_images_by_config(args.config_path, grayscale=args.grayscale, is_draw=args.is_draw,
                          enable_time_printing=args.enable_time_printing)
    # dnn_module = SMoreDnnModule()
    # dnn_module.Init(
    #     'DataKit/configs/anjie.yaml'
    # )
    # base_dir = './Kersen_DATASETS/origin'
    # img_dirs = [
    #     '20230724',
    # ]
    # save_dir = 'DataKit/locate_results/temp'
    # for img_dir in img_dirs:
    #     get_product_locations(dnn_module, base_dir, img_dir, save_dir)