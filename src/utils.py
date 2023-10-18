import json
import yaml
import os
import cv2
import numpy as np


def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def write_to_file(lines, path, sort=False):
    if sort:
        lines.sort()
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def json_load(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


def json_dump(cfg_dict, dump_path, indent=4):
    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(cfg_dict, f, indent=indent, ensure_ascii=False)


def yaml_load(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath, flags=cv2.IMREAD_UNCHANGED):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), flags)
    return cv_img


def cv_imwrite(save_path, image):
    _, ext = os.path.splitext(save_path)
    try:
        cv2.imencode(ext=ext, img=image)[1].tofile(save_path)
    except Exception as e:
        print(f"Error saving image: {e}")


def crop_center(img_path, save_path, pixel_thres=80, dst_size=(384, 384), area_thres=2500, avg_num=1,
                draw_or_crop='crop', save_img=True):
    color_img = cv_imread(img_path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    mask = np.array(gray_img >= pixel_thres, dtype=np.uint8)
    # cv_imwrite('mask.png', 255 * mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # filter by center coordinates and area
    h, w = mask.shape
    new_centroids = []
    edge_left, edge_right, edge_top, edge_bottom = int(w * 0.1), int(w * 0.95), int(h * 0.1), int(h * 0.95)
    for i, stat in enumerate(stats):
        # stat: x, y, w, h, s
        coord1 = (stat[0], stat[1])
        coord2 = (stat[0] + stat[2], stat[1] + stat[3])
        if coord1[0] < edge_left or coord1[1] < edge_top or coord2[0] > edge_right \
                or coord2[1] > edge_bottom:
            continue
        if stat[-1] >= area_thres:
            new_centroids.append(centroids[i])
    assert len(new_centroids) >= 1, "img_path: %s, no center image!" % img_path

    # find connected components closest to the image center
    dist_L = []
    center_x, center_y = w // 2, h // 2
    for i, (x, y) in enumerate(new_centroids):
        dist = (x - center_x) ** 2 + (y - center_y) ** 2
        dist_L.append([i, dist])
    dist_L.sort(key=lambda x: x[1])

    min_centroids = []
    for i in range(len(new_centroids)):
        if i >= avg_num:
            break
        idx = dist_L[i][0]
        min_centroids.append(new_centroids[idx])
    min_centroids = np.array(min_centroids)
    x, y = np.average(min_centroids, axis=0).astype(np.int32).tolist()

    # crop image to the dst size
    top_left = (x - dst_size[0] // 2, y - dst_size[1] // 2)
    bottom_right = (top_left[0] + dst_size[0], top_left[1] + dst_size[1])

    assert draw_or_crop in ['draw', 'crop']
    if draw_or_crop == 'draw':
        # draw rectangle on the original image
        res_img = color_img
        cv2.rectangle(res_img, top_left, bottom_right, (25, 25, 255), 3)
        # # add text for rectangle
        # cv2.putText(res_img, str(1), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 25, 25), 2)
    else:
        # crop rectangle from the original image, adjust two points to avoid out of range
        if top_left[0] < 0 or top_left[1] < 0:
            top_left = (max(0, top_left[0]), max(0, top_left[1]))
            bottom_right = (top_left[0] + dst_size[0], top_left[1] + dst_size[1])
        elif bottom_right[0] >= w or bottom_right[1] >= h:
            bottom_right = (min(w, bottom_right[0]), min(h, bottom_right[1]))
            top_left = (bottom_right[0] - dst_size[0], bottom_right[1] - dst_size[1])
        res_img = color_img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    if save_img:
        cv_imwrite(save_path, res_img)
    return top_left, bottom_right


if __name__ == '__main__':

    data_dir = r'D:\SmartMore-Projects\AIDIXIU_chip_defect\工程对接20230524\Feedback20230529'
    save_dir = r'results'
    config = {
        'pixel_thres': 80,
        'dst_size': (384, 384),
        'area_thres': 50 * 50,
        'avg_num': 1,
        'draw_or_crop': 'crop',
    }
    # data_dir = r"D:\SmartMore-Projects\Maiweishi_filter_defect\datasets\202209\202209"
    # save_dir = r'results2'
    # config = {
    #     'pixel_thres': 90,
    #     'dst_size': (1400, 1600),
    #     'area_thres': 200 * 200,
    #     'avg_num': 3,
    #     'draw_or_crop': 'crop',
    # }

    L = os.listdir(data_dir)
    L.sort()
    for root, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            img_path = os.path.join(root, filename)
            if '.bmp' not in img_path:
                continue
            save_path = img_path.replace(data_dir, save_dir)
            os.makedirs(root.replace(data_dir, save_dir), exist_ok=True)
            crop_center(img_path, save_path, **config)
