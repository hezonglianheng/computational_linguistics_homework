# encoding: utf8

import call_api
import json
import extract_answers_from_response as extract_answers
from pathlib import Path

# 提示语
# 用于抑制模型生成多余的解释和推理过程
# PROMPT = """Please answer the following reasoning questions by directly providing one or more letters from ABCD as the answer, without adding any other explanations or reasoning processes."""
# 用于引导模型进行枚举推理时，先列出所有可能的情况，再逐一推理，避免重复
PROMPT = """Please complete the following reasoning problem. If enumeration is needed to deduce the answer, first list all possible scenarios, and then reason through them one by one, avoiding repetition."""

def main(json_path: str, result_path: str, model_name: str):
    global PROMPT
    # 加载模型信息
    call_api.load_model_info()

    with open(json_path, 'r', encoding='utf8') as f:
        data: list[dict] = json.load(f)

    print(f"Using model: {model_name}")
    contexts = [item['question'] for item in data]
    print(f"Calling API for {len(contexts)} contexts...")
    contexts_with_prompt = [PROMPT + context for context in contexts]
    responses = call_api.batch_call_api_async_wrapper(model_name, contexts_with_prompt, max_concurrency=5)
    print(f"API calls completed for model '{model_name}'.")
    responses_with_answers = extract_answers.extract_answer_from_responses(responses)
    # 保存结果
    result = [{"original": d} | resp | ans for d, resp, ans in zip(data, responses, responses_with_answers)]
    result_file_path = f"{result_path}/{Path(json_path).stem}_{model_name}_results.json"
    
    with open(result_file_path, 'w', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {result_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Call API with prompt for a list of reasoning questions.")
    parser.add_argument("json_path", type=str, help="Path to the input JSON file containing questions.")
    parser.add_argument("result_path", type=str, help="Directory to save the API call results.")
    parser.add_argument("model_name", type=str, help="Name of the model to use for API calls.")
    args = parser.parse_args()

    main(args.json_path, args.result_path, args.model_name)