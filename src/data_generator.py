import os
import json
import yaml
import random
import argparse


class DataGenerator(object):
    """
    用于生成训练和测试数据（列表）的类
    """
    def __init__(self, data_generator_cfg, **kwargs):
        self.processing_queue = data_generator_cfg['processing_queue']
        self.set_all_labels_ok = data_generator_cfg['set_all_labels_ok']
        self.seed = data_generator_cfg['seed']
        self.delimiter = data_generator_cfg['delimiter']
        self.task_dict = {}
        for task in data_generator_cfg['tasks']:
            task_name = task['task_name']
            self.task_dict[task_name] = task

    @staticmethod
    def write_to_file(path, lines):
        if path is not None:
            lines.sort()
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

    def generate_datalist(self, base_dir, img_json_dirs, tags, datalist_paths, train_ratio):
        """
        生成train、test和unknown数据列表
        """
        random.seed(self.seed)
        lines_dict = {key: [] for key in datalist_paths.keys()}  # 初始化lines_dict
        for img_dir, json_dir in img_json_dirs:
            data_dir = os.path.join(base_dir, img_dir)
            print('data_dir:', data_dir)
            for root, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    # 筛选出包含任意一个tag的文件
                    contain_any_tag = any(tag in filename for tag in tags)
                    if not contain_any_tag:
                        continue
                    if '.jpg' in filename or '.bmp' in filename:  # image
                        img_path = os.path.join(root, filename)
                        json_path = os.path.join(root, filename.replace(filename[-4:], '.json'))
                        if json_dir is not None:  # 指定了json的目录
                            json_path = json_path.replace('%s/' % img_dir, '%s/' % json_dir)
                        if self.set_all_labels_ok or not os.path.exists(json_path):
                            lines_dict['train_test'].append('%s%s#\n' % (img_path, self.delimiter))
                        else:
                            # 判断json是否包含unknown label
                            with open(json_path, encoding='utf-8') as f:
                                json_data = json.load(f)
                            contain_unknown_label = any('unknown' in shape['label'] for shape in json_data['shapes'])
                            if contain_unknown_label:
                                lines_dict['unknown'].append('%s%s%s\n' % (img_path, self.delimiter, json_path))
                            else:
                                lines_dict['train_test'].append('%s%s%s\n' % (img_path, self.delimiter, json_path))
        lines_dict['train_test'].sort()
        random.shuffle(lines_dict['train_test'])
        train_cnt = int(train_ratio * len(lines_dict['train_test']))
        lines_dict['train'] = lines_dict['train_test'][:train_cnt]
        lines_dict['test'] = lines_dict['train_test'][train_cnt:]
        for key, path in datalist_paths.items():
            self.write_to_file(path, lines_dict[key])
        return lines_dict

    def process_tasks(self):
        for task_name in self.processing_queue:
            print('processing "%s" task ...' % task_name)
            task = self.task_dict[task_name]
            base_dir = task['base_dir']
            img_json_dirs = task['img_json_dirs']
            train_ratio = task['train_ratio']
            for generate_cfg in task['generate_cfgs']:
                tags = generate_cfg['tags']
                save_dir = generate_cfg['save_dir']
                os.makedirs(save_dir, exist_ok=True)
                datalist_names = generate_cfg['datalist_names']
                def get_path_or_none(key):
                    if datalist_names[key] is None:
                        return None
                    else:
                        return os.path.join(save_dir, datalist_names[key])

                datalist_paths = {
                    'train': get_path_or_none('train'),
                    'test': get_path_or_none('test'),
                    'unknown': get_path_or_none('unknown'),
                    'train_test': None  # 用于临时存储训练和测试数据
                }
                _ = self.generate_datalist(base_dir, img_json_dirs, tags, datalist_paths, train_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data_generator = DataGenerator(config['data_generator_cfg'])
    data_generator.process_tasks()