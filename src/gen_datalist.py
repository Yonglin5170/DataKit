import os
import json
import yaml
import random
import argparse

seed = 42
random.seed(seed)


def gen_datalist(base_dir, img_json_dirs, tags, trainlist, testlist, unknownlist,
                 train_ratio, delimiter='||', only_ok=False):
    txt_lines = []
    unknown_lines = []  # 包含unknown label的数据
    for img_dir, json_dir in img_json_dirs:
        data_dir = os.path.join(base_dir, img_dir)
        print('data_dir:', data_dir)
        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                continue_flag = True
                # 筛选出包含任意一个tag的文件
                for tag in tags:
                    if tag in filename:
                        continue_flag = False
                        break
                if continue_flag:
                    continue
                if '.json' not in filename:  # image
                    img_path = os.path.join(root, filename)
                    json_path = os.path.join(root, filename.replace(filename[-4:], '.json'))
                    if json_dir is not None:  #  指定了json的目录
                        json_path = json_path.replace('/%s/' % img_dir, '/%s/' % json_dir)
                    if only_ok or not os.path.exists(json_path):
                        txt_lines.append('%s%s#\n' % (img_path, delimiter))
                    else:
                        # print('json_path:', json_path)
                        with open(json_path) as f:
                            json_data = json.load(f)
                        contain_unknown_label = False
                        # 判断是否包含unknown label
                        for shape in json_data['shapes']:
                            if 'unknown' in shape['label']:
                                contain_unknown_label = True
                                break
                        if contain_unknown_label:
                            unknown_lines.append('%s%s%s\n' % (img_path, delimiter, json_path))
                        else:
                            txt_lines.append('%s%s%s\n' % (img_path, delimiter, json_path))
    txt_lines.sort()
    random.shuffle(txt_lines)
    train_cnt = int(train_ratio * len(txt_lines))
    train_lines, test_lines = txt_lines[:train_cnt], txt_lines[train_cnt:]
    if trainlist is not None:
        train_lines.sort()
        os.makedirs(os.path.dirname(trainlist), exist_ok=True)
        with open(trainlist, 'w') as f:
            f.writelines(train_lines)
    if testlist is not None:
        test_lines.sort()
        os.makedirs(os.path.dirname(testlist), exist_ok=True)
        with open(testlist, 'w') as f:
            f.writelines(test_lines)
    if unknownlist is not None:
        unknown_lines.sort()
        os.makedirs(os.path.dirname(unknownlist), exist_ok=True)
        with open(unknownlist, 'w') as f:
            f.writelines(unknown_lines)
    return txt_lines, train_lines, test_lines, unknown_lines


def gen_datalist_by_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    datalist_cfg = config['datalist_cfg']
    for each_task_cfg in datalist_cfg:
        base_dir = each_task_cfg['base_dir']
        img_json_dirs = each_task_cfg['img_json_dirs']
        train_ratio = each_task_cfg['train_ratio']
        for each_generate_cfg in each_task_cfg['generate_cfg']:
            tags = each_generate_cfg['tags']
            save_dir = each_generate_cfg['save_dir']
            if 'trainlist_name' in each_generate_cfg:
                trainlist_name = each_generate_cfg['trainlist_name']
                trainlist = os.path.join(save_dir, trainlist_name)
            else:
                trainlist = None
            if 'testlist_name' in each_generate_cfg:
                testlist_name = each_generate_cfg['testlist_name']
                testlist = os.path.join(save_dir, testlist_name)
            else:
                testlist = None
            if 'unknownlist_name' in each_generate_cfg:
                unknownlist_name = each_generate_cfg['unknownlist_name']
                unknownlist = os.path.join(save_dir, unknownlist_name)
            else:
                unknownlist = None
            _ = gen_datalist(base_dir, img_json_dirs, tags, trainlist, testlist, unknownlist,
                             train_ratio, '||', False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    gen_datalist_by_config(args.config_path)