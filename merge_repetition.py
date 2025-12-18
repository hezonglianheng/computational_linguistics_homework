# encoding: utf8

import json
from pathlib import Path
from typing import Any

def repetition_sum(repetition_counts: list[dict[str, Any]]) -> int:
    return sum(item['count'] for item in repetition_counts)

def main(file_dir: str):
    # 获得file_dir下的全部json文件
    file_dir_path = Path(file_dir)
    json_files = list(file_dir_path.glob("*.json"))

    data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf8') as f:
            file_data: list[dict] = json.load(f)
            dataset, model, _ = json_file.stem.split('_', 2)
            # 添加来源信息
            for item in file_data:
                item['dataset'] = dataset
                item['model'] = model
            data.extend(file_data)
    print(f"总记录数: {len(data)}")

    # 排序
    data.sort(key=lambda x: (x['index'], repetition_sum(x.get('repetitions', []))))
    flags = [1] * len(data)
    for i in range(1, len(data)):
        if data[i]['index'] == data[i-1]['index']:
            flags[i] = 0

    selected_data = [data[i] for i in range(len(data)) if flags[i] == 1]

    # 保存合并结果
    print(f"合并后的记录数: {len(selected_data)}")
    result_path = "merged_repetition_results.json"
    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="包含JSON文件的目录路径")
    args = parser.parse_args()
    main(args.dir)