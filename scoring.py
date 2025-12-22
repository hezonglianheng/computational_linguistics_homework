# encoding: utf8

"""
用于对模型输出的结果进行评分的工具模块
"""

import json
from typing import Literal

def single_score(standard_answer: list[str], model_answer: list[str], mode: Literal['strict', 'lenient']) -> float:
    """计算单个模型回答的得分

    Args:
        standard_answer (list[str]): 标准答案列表
        model_answer (list[str]): 模型生成的答案列表

    Returns:
        float: 计算得到的得分
    """
    if not standard_answer or not model_answer:
        return 0.0

    standard_set = set(standard_answer)
    model_set = set(model_answer)
    if mode == 'strict':
        if standard_set >= model_set and model_set >= standard_set:
            return 1.0
        else:
            return 0.0
    elif mode == 'lenient':
        intersection = standard_set & model_set
        return len(intersection) / len(standard_set)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def batch_score(standard_answers: list[list[str]], model_answers: list[list[str]], mode: Literal['strict', 'lenient']) -> float:
    """计算整个文件的模型回答得分

    Args:
        standard_answers (list[list[str]]): 标准答案列表
        model_answers (list[list[str]]): 模型生成的答案列表

    Returns:
        float: 计算得到的平均得分
    """
    assert len(standard_answers) == len(model_answers), "标准答案和模型答案的数量必须相同"

    total_score = 0.0
    for std_ans, mod_ans in zip(standard_answers, model_answers):
        total_score += single_score(std_ans, mod_ans, mode)
    
    return total_score / len(standard_answers)

def file_score(json_path: str, mode: Literal['strict', 'lenient']):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    standard_answers: list[list[str]] = [item['answer'] for item in data]
    model_answers: list[list[str]] = [item['extracted_answers'] for item in data]

    avg_score = batch_score(standard_answers, model_answers, mode)
    print(f'试题数量: {len(data)}')
    print(f'模型回答的平均得分（{mode}）: {avg_score:.4f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', type=str, help="包含标准答案和模型答案的JSON文件路径")
    parser.add_argument('--mode', type=str, choices=['strict', 'lenient'], default='strict', help="评分模式，支持'strict'和'lenient'")
    args = parser.parse_args()
    file_score(args.json_path, args.mode)