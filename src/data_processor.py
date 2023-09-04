import os
import json
import matplotlib.pyplot as plt
from collections import Counter

import sys
sys.path.append('/dataset/yonglinwu/SMore/DataKit')

from src import utils


class DataProcessor(object):
    """
    用于处理和分析数据的类
    """
    def __init__(self, delimiter='||', plt_fig_size=(10, 4)):
        self.delimiter = delimiter
        self.plt_fig_size = plt_fig_size

    def clear_imageData_in_json(self, base_dir, sub_dirs):
        """
        清除json中的imageData
        """
        for sub_dir in sub_dirs:
            data_dir = os.path.join(base_dir, sub_dir)
            for root, _, filenames in os.walk(data_dir):
                for filename in filenames:
                    if '.json' not in filename:
                        continue
                    json_path = os.path.join(root, filename)
                    data = utils.json_load(json_path)
                    if data['imageData'] is not None:
                        print(json_path)
                        data['imageData'] = None
                        utils.json_dump(data, json_path)

    @staticmethod
    def load_data_paths_of_exp(exp_yaml_path):
        config = utils.yaml_load(exp_yaml_path)
        data_paths = {}
        for mode in ['train', 'eval']:
            dataset_config = config['data'][f'{mode}_data']['dataset']
            if mode == 'eval':
                dataset_config = dataset_config[0]
            data_paths[mode] = dataset_config['data_path']
        return data_paths

    @staticmethod
    def create_single_symlink(source, target):
        if not os.path.exists(source):
            print(f"source do not exists: {source}")
            return
        try:
            os.symlink(source, target)
            print(f"Symlink created: {target} -> {source}")
        except FileExistsError:
            print(f"Symlink already exists: {target}")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.symlink(source, target)
            print(f"Symlink created: {target} -> {source}")

    def create_symlinks(self, exp_yaml_path, src_dir, dst_dir, tags):
        """
        创建符号链接，指向exp.yaml中的数据路径，并保持目录结构
        """
        data_paths = self.load_data_paths_of_exp(exp_yaml_path)

        for mode in ['train', 'eval']:
            for data_path in data_paths[mode]:
                datalist_path = data_path['path']
                lines = utils.read_file(datalist_path)
                for line in lines:
                    image_path, json_path = line.strip().split(self.delimiter)
                    full_image_path = os.path.join(data_path['root'], image_path)
                    full_json_path = os.path.join(data_path['root'], json_path)

                    if any(tag in image_path for tag in tags):
                        dst_image_path = full_image_path.replace(src_dir, dst_dir)
                        dst_json_path = dst_image_path.replace(dst_image_path[-4:], '.json')
                        self.create_single_symlink(full_image_path, dst_image_path)
                        self.create_single_symlink(full_json_path, dst_json_path)

    def plot_distribution(self, distribution, index, mode):
        if index == 1:
            plt.figure(figsize=self.plt_fig_size)

        labels, values = zip(*distribution.items())
        plt.subplot(1, 2, index)
        plt.bar(labels, values)
        plt.title(f'{mode} Label Distribution')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=90)

        if index == 2:
            plt.show()

    def get_stats_of_data_path(self, data_path, certain_labels=[]):
        """
        获取指定数据集的标签分布
            data_path: {'root': xxx, 'path': xxx}
        """
        datalist_path = data_path['path']
        lines = utils.read_file(datalist_path)
        label_distribution = Counter()
        OK_cnt, NG_cnt = 0, 0
        for line in lines:
            image_path, json_path = line.strip().split(self.delimiter)
            full_json_path = os.path.join(data_path['root'], json_path)

            if not os.path.exists(full_json_path):
                OK_cnt += 1
                continue
            json_data = utils.json_load(full_json_path)
            for shape in json_data['shapes']:
                label = shape['label']
                label_distribution[label] += 1
            OK_cnt += int(len(json_data['shapes']) == 0)
            NG_cnt += int(len(json_data['shapes']) > 0)
            
            if len(certain_labels) > 0:
                # 输出包含certain_labels的json路径
                certain_label_dict = {key: 0 for key in certain_labels}
                for shape in json_data['shapes']:
                    label = shape['label']
                    if label in certain_label_dict:
                        certain_label_dict[label] += 1
                if sum(certain_label_dict.values()) > 0:
                    print(json_path, certain_label_dict)
            
        sorted_label_distribution = Counter(dict(sorted(label_distribution.items())))
        return sorted_label_distribution, OK_cnt, NG_cnt

    def get_stats_of_exp(self, exp_yaml_path, certain_labels=[]):
        """
        获取exp.yaml中train和eval数据集的标签分布
        """
        data_paths = self.load_data_paths_of_exp(exp_yaml_path)

        for i, mode in enumerate(['train', 'eval']):
            results = [self.get_stats_of_data_path(data_path, certain_labels) for data_path in data_paths[mode]]
            all_label_distribution = sum((res[0] for res in results), Counter())
            all_label_distribution = dict(sorted(all_label_distribution.items()))
            total_OK_cnt = sum(res[1] for res in results)
            total_NG_cnt = sum(res[2] for res in results)
            print(f'all {mode} label distribution:', json.dumps(all_label_distribution, indent=4))
            print(f'total_OK_cnt: {total_OK_cnt}, total_NG_cnt: {total_NG_cnt}')

            self.plot_distribution(all_label_distribution, i + 1, mode)

    def get_stats_of_dirs(self, base_dir, img_json_dirs, tags, certain_labels=[]):
        """
        获取指定文件夹中所有数据的标签分布
        """
        print('tags:', tags)
        all_label_distribution = Counter()
        total_OK_cnt, total_NG_cnt = 0, 0
        for img_dir, json_dir in img_json_dirs:
            # process each data_dir
            data_dir = os.path.join(base_dir, img_dir)
            print('data_dir:', data_dir)
            OK_cnt, NG_cnt = 0, 0
            label_distribution = Counter()
            for root, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    contain_any_tag = any(tag in filename for tag in tags)
                    contain_any_img = any(ext in filename for ext in ['.jpg', '.bmp', '.png'])
                    if (not contain_any_tag) or (not contain_any_img):
                        continue

                    json_path = os.path.join(root, filename.replace(filename[-4:], '.json'))
                    if json_dir is not None:
                        json_path = json_path.replace(img_dir, json_dir)
                    if not os.path.exists(json_path):
                        OK_cnt += 1
                        continue
                    json_data = utils.json_load(json_path)
                    for shape in json_data['shapes']:
                        label = shape['label']
                        label_distribution[label] += 1
                    OK_cnt += int(len(json_data['shapes']) == 0)
                    NG_cnt += int(len(json_data['shapes']) > 0)

                    if len(certain_labels) > 0:
                        # 输出包含certain_labels的json路径
                        certain_label_dict = {key: 0 for key in certain_labels}
                        for shape in json_data['shapes']:
                            label = shape['label']
                            if label in certain_label_dict:
                                certain_label_dict[label] += 1
                        if sum(certain_label_dict.values()) > 0:
                            print(json_path, certain_label_dict)
                    
            label_distribution = dict(sorted(label_distribution.items(), key=lambda x: x[0]))
            all_label_distribution += label_distribution
            total_OK_cnt += OK_cnt
            total_NG_cnt += NG_cnt
            # print(img_dir.split('/')[-1])
            # print('OK:', OK_cnt, 'NG:', NG_cnt)
            # print('label_distribution:', json.dumps(label_distribution, indent=4))

        all_label_distribution = dict(sorted(all_label_distribution.items(), key=lambda x: x[0]))
        print('all_label_distribution:', json.dumps(all_label_distribution, indent=4))
        print(f'total_OK_cnt: {total_OK_cnt}, total_NG_cnt: {total_NG_cnt}')

        self.plot_distribution(all_label_distribution, 1, 'all')


if __name__ == '__main__':
    data_processor = DataProcessor()

    # base_dir = '/dataset/yonglinwu/SMore/B698/temp/Dataset'
    # sub_dirs = [
    #     '20230711'
    # ]
    # data_processor.clear_imageData_in_json(base_dir, sub_dirs)

    # create symbolic links
    # exp_yaml_path = '/dataset/yonglinwu/SMore/B698/EXPERIMENTS/pos3_v6/exp.yaml'
    # src_dir = '/dataset/yonglinwu/SMore/B698/DATASETS/cropped2'
    # dst_dir = '/dataset/yonglinwu/SMore/B698/temp/Dataset/3工位_all'
    # tags = ['3工位']
    # data_processor.create_symlinks(exp_yaml_path, src_dir, dst_dir, tags)

    # get label statistics of a exp
    exp_yaml_path = '/dataset/yonglinwu/SMore/B698/EXPERIMENTS/pos3_v6/exp.yaml'
    certain_labels = ['yise', 'aokeng']
    data_processor.get_stats_of_exp(exp_yaml_path, certain_labels)

    # get label statistics of a series of dataset directories
    # base_dir = '/dataset/yonglinwu/SMore/B698/DATASETS/origin'
    # img_json_dirs = [
    #     ['20230828', None],
    # ]
    # tags = ['2工位']
    # certain_labels = ['yise', 'aokeng']
    # data_processor.get_stats_of_dirs(base_dir, img_json_dirs, tags, certain_labels)
