# encoding: utf8

from datasets import load_dataset
import jsonlines
import json
import random

random.seed(0) # for reproducibility

DATASET_INFO: dict[str, dict[str, str]] = {
    'DeepMath-103K': {
        'source': 'web',
        'path': 'zwhe99/DeepMath-103K',
        'split': 'train'
    },
    'gsm8k': {
        'source': 'local',
        'file_path': 'datasets/math/gsm8k/test.jsonl',
        'file_format': 'jsonl'
    }, 
    'piqa': {
        'source': 'local',
        'file_path': 'datasets/commonsense/piqa/test.jsonl',
        'file_format': 'jsonl'
    }, 
    'Com2': {
        'source': 'local',
        'file_path': 'datasets/commonsense/com_2/main.json',
        'file_format': 'json'
    }, 
    'KnowLogic': {
        'source': 'local', 
        'path': 'datasets/commonsense/KnowLogic/test.jsonl', 
        'file_format': "jsonl"
    }
}

def load_data_from_web(dataset_path: str, split: str|None = None):
    ds = load_dataset(dataset_path, split=split)
    return ds

def load_data_from_local(file_path: str, file_format: str):
    if file_format == 'jsonl':
        data: list[dict] = []
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                data.append(obj)
        return data
    if file_format == 'json':
        with open(file_path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def load_data(dataset_name: str) -> list[dict[str, str]]:
    """加载指定数据集

    Args:
        dataset_name (str): 数据集名称

    Raises:
        ValueError: 不支持的数据来源

    Returns:
        list[dict[str, str]]: 整理后的数据集，包括question、answer、raw_data字段
    """
    assert dataset_name in DATASET_INFO, f"Dataset info for '{dataset_name}' not found."
    info = DATASET_INFO[dataset_name]
    source = info.get('source', '')
    
    # Load data based on the source type
    if source == 'web':
        path = info.get('path', '')
        split = info.get('split', None)
        ds = load_data_from_web(path, split)
    elif source == 'local':
        file_path = info.get('file_path', '')
        file_format = info.get('file_format', '')
        ds = load_data_from_local(file_path, file_format)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    # arrange data
    if dataset_name == 'DeepMath-103K':
        # ensure ds is indexable (convert to list if necessary) and get total count
        try:
            total = len(ds)
        except Exception:
            ds = list(ds)
            total = len(ds)

        target = max(1, total // 10)

        # build weights from 'difficulty' field (float), clamp negatives to 0
        weights = []
        for item in ds:
            diff = item.get('difficulty', 0.0) if isinstance(item, dict) else getattr(item, 'difficulty', 0.0)
            try:
                w = float(diff)
            except Exception:
                w = 0.0
            weights.append(max(0.0, w))

        # fallback to uniform weights if all weights are zero
        if sum(weights) == 0:
            weights = [1.0] * total

        # weighted sampling without replacement
        selected = set()
        while len(selected) < target:
            idx = random.choices(range(total), weights=weights, k=1)[0]
            selected.add(idx)

        # keep selected items in original order
        selected_indices = sorted(selected)
        ds = [ds[i] for i in selected_indices]
        final_dataset = [
            {
                'question': item['question'],
                'answer': item['final_answer'], 
                'raw_data': item, 
            }
            for item in ds
        ]
    elif dataset_name == 'Com2':
        final_dataset = [
            {
                'question': item.get('scenario', '') + "\n" + item['question'] + "\n" + item.get('options', ''),
                'answer': item['answer'], 
                'raw_data': item, 
            }
            for item in ds
        ]
    elif dataset_name == 'Score':
        final_dataset = [
            {
                'question': item['text'] + '\n' + item['question'] + '\n' + '\t'.join([f"{k}: {v}" for k, v in item['options'].items()]),
                'answer': ', '.join(item['answer']), 
                'raw_data': item, 
            }
            for item in ds
        ]
    else:
        final_dataset = [
            {
                'question': item['question'],
                'answer': item['answer'],
                'raw_data': item, 
            }
            for item in ds
        ]

    return final_dataset

    
if __name__ == "__main__":
    # dataset = load_data('gsm8k')
    # print(dataset[:2])
    dataset = load_data('DeepMath-103K')
    print(dataset[:2])
    # dataset = load_data('Com2')
    # print(dataset[:2])
    # dataset = load_data('Score')
    # print(dataset[:2])