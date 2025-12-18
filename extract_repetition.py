# encoding: utf8

"""提取模型响应中的重复部分
"""

from langdetect import detect
import json
from collections import defaultdict
from typing import Any
from pathlib import Path

FORMAT_SIGNALS = [
    "**", 
    "###"
]

SPLIT_SIGNALS = [
    "\n", 
]

CN_SPLIT_SIGNALS = [
    "\n", 
    "。", 
    "，", 
]

EN_SPLIT_SIGNALS = [
    ".\n", 
    "\n", 
    '."', 
]

REPETITION_LOWER_THRESHOLD = 50  # 重复计数的下限
REPETITION_LEN_THRESHOLD = 5 # 重复片段的长度下限
REPETITION_THRESHOLD_WITH_LEN = 5 # 重复片段的长度下限
REPETITION_WINDOW_THRESHOLD = 10  # 重复检测的窗口大小

def remove_format_signals(text: str) -> str:
    """移除文本中的格式标记

    Args:
        text (str): 输入文本

    Returns:
        str: 移除格式标记后的文本
    """
    for signal in FORMAT_SIGNALS:
        text = text.replace(signal, "")
    return text

def split_response(response: str) -> list[str]:
    """根据预定义的格式和分割信号，拆分模型响应为多个部分

    Args:
        response (str): 模型完整响应内容

    Returns:
        list[str]: 拆分后的响应部分列表
    """
    def _split_by_signals(text: str, signal: str) -> tuple[list[str], str]: 
        """根据具体的标记拆分文本

        Args:
            text (str): 文本
            signal (str): 拆分标记

        Returns:
            tuple[list[str], str]: 拆分后的文本部分列表和语言
        """
        parts = text.split(signal)
        return parts

    parts = [response]
    signals = []
    # signals.extend(SPLIT_SIGNALS)
    if (lang := detect(response)) == 'zh-cn' or lang == 'zh-tw':
        signals.extend(CN_SPLIT_SIGNALS)
    elif lang == 'en':
        signals.extend(EN_SPLIT_SIGNALS)
    for signal in signals:
        new_parts = []
        for part in parts:
            split_parts = _split_by_signals(part, signal)
            new_parts.extend(split_parts)
        parts = new_parts
    
    return parts, lang

def extract_repetition(response_parts: list[str], lang: str) -> dict[str, int]:
    """提取响应部分中的重复内容

    Args:
        response_parts (list[str]): 模型响应的各个部分
        lang (str): 语言

    Returns:
        dict[str, int]: 重复内容及其计数
    """
    repetition_counts = defaultdict(int)
    for i in range(len(response_parts)):
        part = response_parts[i]
        if not part.strip():
            continue
        # 在前面查找重复
        window_start = max(0, i - REPETITION_WINDOW_THRESHOLD)
        if part in response_parts[window_start:i]:
            repetition_counts[part] += 1
    
    # 过滤掉计数低于阈值的重复内容
    # repetition_counts = {part: count + 1 for part, count in repetition_counts.items() if count + 1 >= REPETITION_LOWER_THRESHOLD}
    # 过滤掉较短的无效重复内容
    # repetition_counts = {part: count for part, count in repetition_counts.items() if len(part) >= REPETITION_LEN_THRESHOLD}
    if lang == 'zh-cn' or lang == 'zh-tw':
        # 获得中文字符不少于5个的part
        repetition_counts = {part: count for part, count in repetition_counts.items() if (count + 1 >= REPETITION_THRESHOLD_WITH_LEN and sum(1 for c in part if '\u4e00' <= c <= '\u9fff') >= REPETITION_LEN_THRESHOLD) or count >= REPETITION_LOWER_THRESHOLD}
    elif lang == "en":
        repetition_counts = {part: count for part, count in repetition_counts.items() if (count + 1 >= REPETITION_THRESHOLD_WITH_LEN and len(part.split()) >= REPETITION_LEN_THRESHOLD) or count >= REPETITION_LOWER_THRESHOLD}
    else:
        repetition_counts = {}

    return repetition_counts

def main(json_path: str, result_dir: str):
    with open(json_path, 'r', encoding='utf8') as f:
        data: list[dict[str, Any]] = json.load(f)

    items_with_repetitions = []
    for i, item in enumerate(data):
        response = item.get('response', '')
        if not response:
            continue
        # cleaned_response = remove_format_signals(response)
        cleaned_response = response
        response_parts, lang = split_response(cleaned_response)
        repetitions = extract_repetition(response_parts, lang)
        if repetitions:
            repetition_info = [{"repetition_text": text, "count": count} for text, count in repetitions.items()]
            repetition_info.sort(key=lambda x: x["count"], reverse=True)
            new_item = {"index": i} | item | {"repetitions": repetition_info}
            items_with_repetitions.append(new_item)

    print(f"在 {json_path} 的 {len(data)} 条记录中，发现 {len(items_with_repetitions)} 条包含重复内容。")
    # 保存包含重复内容的结果
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(result_dir) / Path(json_path).name
    with open(result_path, 'w', encoding='utf8') as f:
        json.dump(items_with_repetitions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract repetitions from model responses.")
    parser.add_argument('--input', type=str, required=True, help="输入的JSON文件路径")
    parser.add_argument('--output_dir', type=str, required=True, help="输出结果的目录")
    args = parser.parse_args()
    main(args.input, args.output_dir)