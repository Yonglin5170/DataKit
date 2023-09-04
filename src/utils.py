import json
import yaml


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