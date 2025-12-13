# encoding: utf8

import load_data
import call_api
import json

USE_DATASETS = [
    "Com2",
]

USE_MODELS = [
    "deepseek-r1-distill-qwen-1.5b",
    "deepseek-r1-distill-qwen-7b", 
    "deepseek-r1-distill-llama-8b", 
    "qwen3-0.6b", 
    "qwen3-1.7b",
    "qwen3-4b", 
    "qwen3-8b", 
    "glm-4-flash-250414", 
    "llama3-8b-instruct", 
]

RESULT_DIR = "./results/"

def run():
    # 加载模型信息
    call_api.load_model_info()

    for dataset_name in USE_DATASETS:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_data.load_data(dataset_name)
        print(f"Dataset '{dataset_name}' loaded with {len(dataset)} samples.")

        for model_name in USE_MODELS:
            print(f"Using model: {model_name}")
            contexts = [item['question'] for item in dataset]
            print(f"Calling API for {len(contexts)} contexts...")
            responses = call_api.batch_call_api_async_wrapper(model_name, contexts, max_concurrency=4)
            print(f"API calls completed for model '{model_name}' on dataset '{dataset_name}'.")
            # 保存结果
            result = [{'question': ctx} | resp for ctx, resp in zip(contexts, responses)]
            result_path = f"{RESULT_DIR}/{dataset_name}_{model_name}_results.json"
            with open(result_path, 'w', encoding='utf8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"Results saved to: {result_path}")

def test():
    # 测试单个API调用
    call_api.load_model_info()
    
    for model_name in USE_MODELS:
        print(f"Testing model: {model_name}")
        context = "What is the capital of France?"
        response = call_api.single_call_api(model_name, context)
        print(f"Response from model '{model_name}': {response}")
        print("-" * 50)

if __name__ == "__main__":
    # test()
    run()