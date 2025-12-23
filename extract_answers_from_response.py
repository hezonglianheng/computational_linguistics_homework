# encoding: utf8

"""Extract answers from API response JSON structure. Use LLM.
"""

import call_api
from typing import Any

PROMPT_TEMPLATE = """
从以下模型回复的内容中提取出答案部分，忽略其他多余信息。
这些回复内容可能包含解释、背景信息或格式化文本。
答案可能是A、B、C、D字母中的一个或多个。只返回答案本身，不要添加任何解释说明。
如果有多个答案，请以顿号分割每个答案（如A、B）。
如果无法提取出明确答案，返回“无法获得答案”。"""

def response_text2answer_list(response: str) -> list[str]:
    """将模型回复文本中的答案部分提取为答案列表

    Args:
        response (str): 模型回复文本

    Returns:
        list[str]: 提取出的答案列表
    """
    response = response.strip()
    if response == "无法获得答案":
        return []
    # 按顿号分割答案
    answers = [ans.strip() for ans in response.split("、") if ans.strip()]
    return answers

def extract_answer_from_responses(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从模型的response中通过模型自动提取答案

    Args:
        responses (list[dict[str, Any]]): 模型的response

    Returns:
        list[dict[str, Any]]: 包含提取答案的结果
    """
    response_texts = [resp['response'] for resp in responses]
    # 在所有回复前添加提示
    combined_contexts = [PROMPT_TEMPLATE + "\n\n" + text for text in response_texts]
    extract_responses = call_api.batch_call_api_async_wrapper(model_name="deepseek-V3.2", contexts=combined_contexts)
    extract_responses_texts = [i['response'] for i in extract_responses]
    answer_lists = [response_text2answer_list(text) for text in extract_responses_texts]
    results = [r | {"model_extract_response": i} | {"extracted_answers": ans} for r, i, ans in zip(responses, extract_responses, answer_lists)]
    return results

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="输入的JSON文件路径，包含模型response")
    parser.add_argument("--output_path", type=str, required=True, help="输出的JSON文件路径，包含提取的答案")
    args = parser.parse_args()

    call_api.load_model_info()
    with open(args.input_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    extracted_results = extract_answer_from_responses(data)
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(extracted_results, f, ensure_ascii=False, indent=4)
    print(f"Extracted answers saved to: {args.output_path}")