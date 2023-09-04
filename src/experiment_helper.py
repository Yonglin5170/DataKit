import os
import shutil
import hashlib
import pandas as pd

import sys
sys.path.append('/dataset/yonglinwu/SMore/DataKit')

from src import utils


class ExperimentHelper(object):
    """
    实验辅助器，自动化执行实验前后的一些操作
    """
    def __init__(self, data_root, experiment_root):
        self.data_root = os.path.abspath(data_root)
        self.experiment_root = os.path.abspath(experiment_root)

    def create_experiment_dir(self, baseline_exp_dir_name, new_exp_dir_name):
        new_exp_dir = os.path.join(self.experiment_root, new_exp_dir_name)
        os.makedirs(new_exp_dir, exist_ok=True)
        baseline_exp_dir = os.path.join(self.experiment_root, baseline_exp_dir_name)
        shutil.copy(
            os.path.join(baseline_exp_dir, 'exp.yaml'),
            os.path.join(new_exp_dir, 'exp.yaml')
        )
        return new_exp_dir

    def modify_dataset(self, exp_dir, new_trainlist, new_validlist=None, new_testlist=None):
        exp_yaml_path = os.path.join(exp_dir, 'exp.yaml')
        lines = utils.read_file(exp_yaml_path)
        insert_indexes = []
        for i, line in enumerate(lines):
            if 'category_map: *category_map' in line:
                insert_indexes.append(i)

        if new_testlist is not None:
            testlist_str = f'          - root: {self.data_root}\n' + \
                f'            path: {new_testlist}\n'
            lines.insert(insert_indexes[2], testlist_str)
        if new_validlist is not None:
            validlist_str = f'          - root: {self.data_root}\n' + \
                f'            path: {new_validlist}\n'
            lines.insert(insert_indexes[1], validlist_str)
        trainlist_str = f'        - root: {self.data_root}\n' + \
            f'          path: {new_trainlist}\n'
        lines.insert(insert_indexes[0], trainlist_str)
        utils.write_to_file(lines, exp_yaml_path)

    def extract_md5(self, file_path):
        md5_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def modify_sdk_config(self, config_path, keys_value_pairs=[]):
        config_data = utils.json_load(config_path)
        for keys, value in keys_value_pairs:
            temp_config_data = config_data
            for key in keys[:-1]:
                temp_config_data = temp_config_data[key]
            temp_config_data[keys[-1]] = value
        utils.json_dump(config_data, config_path, indent=2)

    def record_result(self, baseline_exp_dir_name, new_exp_dir):
        metric_dirs = os.listdir(os.path.join(new_exp_dir, 'result/val'))
        metric_dirs.sort(key=lambda x: int(x))
        metric_path = os.path.join(new_exp_dir, f'result/val/{metric_dirs[-1]}/metrics.json')
        metrics = utils.json_load(metric_path)
        selected_keys = [
            'PixelBasedEvaluator/iou/mean_valid',
            'PixelBasedEvaluator/recall/mean_valid',
            'PixelBasedEvaluator/precision/mean_valid'
        ]
        metric_to_record = ''
        for key in selected_keys:
            value = metrics[key]
            key = '/'.join(key.split('/')[1:])
            metric_to_record += f'{key}: {value:.2f}\n'

        df = pd.DataFrame(columns=['exp_name', 'baseline', 'diff', 'result', 'notes'])
        record_list = [os.path.basename(new_exp_dir), baseline_exp_dir_name, '', metric_to_record, '']
        df.loc[len(df)] = record_list
        df.to_csv(os.path.join(new_exp_dir, 'metric_record.csv'))


if __name__ == '__main__':
    data_root = '/dataset/yonglinwu/SMore/B698/'
    experiment_root = '/dataset/yonglinwu/SMore/B698/temp/EXPs'

    experiment_helper = ExperimentHelper(data_root, experiment_root)

    # before experiment: create experiment directory and add new dataset
    baseline_exp_dir_name = 'pos3_v6'
    new_exp_dir_name = 'pos3_v7'
    new_trainlist = '/dataset/yonglinwu/SMore/B698/DATASETS/datalist/3工位/train_20230828.txt'
    new_validlist = '/dataset/yonglinwu/SMore/B698/DATASETS/datalist/3工位/train_20230828.txt'
    new_testlist = '/dataset/yonglinwu/SMore/B698/DATASETS/datalist/3工位/train_20230828.txt'

    new_exp_dir = experiment_helper.create_experiment_dir(baseline_exp_dir_name, new_exp_dir_name)
    experiment_helper.modify_dataset(new_exp_dir, new_trainlist, new_validlist, new_testlist)

    # after experiment: update model path and md5 value, record result
    model_path = f'{new_exp_dir}/deploy/16000.onnx'
    md5_value = experiment_helper.extract_md5(model_path)
    print('MD5 Value:', md5_value)

    sdk_config_path = '/dataset/yonglinwu/SMore/B698/temp/sdk_settings/PipelineHousingPos3.json'
    keys_value_pairs = [
        [['xrack_cfg', 'xrack_module', 'engine_cfg', 'model_path'], model_path],
        [['xrack_cfg', 'xrack_module', 'engine_cfg', 'md5'], md5_value]
    ]
    experiment_helper.modify_sdk_config(sdk_config_path, keys_value_pairs)
    experiment_helper.record_result(baseline_exp_dir_name, new_exp_dir)
