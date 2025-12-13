# encoding: utf8

import asyncio
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai

MODEL_INFO: dict[str, dict[str, str]] = {}

def load_model_info() -> None:
    """自./api_keys/api_keys.json加载模型信息
    """
    global MODEL_INFO
    assert os.path.exists('./api_keys/api_keys.json'), "api_keys.json isn't found in ./api_keys/"
    with open('./api_keys/api_keys.json', 'r', encoding='utf8') as f:
        MODEL_INFO = json.load(f)

def _callapi(api_key: str, base_url: str, model: str, context: str) -> dict[str, Any]:
    """调用OpenAI API接口

    Args:
        api_key (str): API密钥
        base_url (str): API基础URL
        model (str): 模型名称
        context (str): 用户输入内容
    """
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    try:
        if model.startswith("qwen3"):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": context}
                ],
                temperature=0.0, 
                extra_body={"enable_thinking": False}  # 关闭“思考”模式
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": context}
                ], 
                temperature=0.0,
            )
        return {
            'response': response.choices[0].message.content,
            'status': "success", 
            'completion_token_cost': response.usage.completion_tokens, 
        }
    except Exception as e:
        return {
            'response': str(e),
            'status': "error", 
            'completion_token_cost': 0,
        }

async def _callapi_async(api_key: str, base_url: str, model: str, context: str) -> dict[str, Any]:
    """异步调用OpenAI API接口，返回与同步版一致的结构"""

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        if model.startswith("qwen3"):
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": context}
                ],
                temperature=0.0,
                extra_body={"enable_thinking": False}
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": context}
                ], 
                temperature=0.0,
            )
        return {
            'response': response.choices[0].message.content,
            'status': "success",
            'token_cost': response.usage.total_tokens,
        }
    except Exception as e:
        return {
            'response': str(e),
            'status': "error",
            'token_cost': 0,
        }

def get_model_info(model_name: str) -> dict[str, str]:
    """获取指定模型的信息
    """
    return MODEL_INFO.get(model_name, {})

def single_call_api(model_name: str, context: str) -> dict[str, Any]:
    """单次调用指定模型的API接口

    Args:
        model_name (str): 模型名称
        context (str): 用户输入内容

    Returns:
        str: 模型返回的内容
    """
    model_info = get_model_info(model_name)
    assert model_info, f"Model info for '{model_name}' not found."
    api_key = model_info.get('api_key', '')
    base_url = model_info.get('base_url', '')
    model = model_info.get('model_name', '')
    assert api_key and base_url and model, f"Incomplete model info for '{model_name}': {model_info}"
    return _callapi(api_key, base_url, model, context)

def batch_call_api(model_name: str, contexts: list[str], max_workers: int = 5) -> list[dict[str, Any]]:
    """批量调用指定模型的API接口，每次请求前增加一个随机的sleep时间

    Args:
        model_name (str): 模型名称
        contexts (list[str]): 用户输入内容列表
        max_workers (int, optional): 最大并发线程数. Defaults to 5.

    Returns:
        list[dict[str, Any]]: 模型返回的内容列表
    """
    model_info = get_model_info(model_name)
    assert model_info, f"Model info for '{model_name}' not found."
    api_key = model_info.get('api_key', '')
    base_url = model_info.get('base_url', '')
    model = model_info.get('model_name', '')
    assert api_key and base_url and model, f"Incomplete model info for '{model_name}'."

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_context = {executor.submit(_callapi, api_key, base_url, model, context): context for context in contexts}
        completed = 0
        total = len(contexts)
        for future in as_completed(future_to_context):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"已完成 {completed} / {total} 个请求")
    return results

async def batch_call_api_async(model_name: str, contexts: list[str], max_concurrency: int = 5) -> list[dict[str, Any]]:
    """异步批量调用指定模型的API接口，保持与同步版一致的返回"""

    model_info = get_model_info(model_name)
    assert model_info, f"Model info for '{model_name}' not found."
    api_key = model_info.get('api_key', '')
    base_url = model_info.get('base_url', '')
    model = model_info.get('model_name', '')
    # max_concurrency = model_info.get('max_concurrency', max_concurrency) # 使用模型配置的最大并发数
    assert api_key and base_url and model, f"Incomplete model info for '{model_name}'."

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call_with_random_sleep(idx: int, context: str) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            # await asyncio.sleep(random.uniform(0, 1))
            res = await _callapi_async(api_key, base_url, model, context)
            # res["context"] = context  # 让调用方可以直接定位对应输入
            return idx, res

    tasks = [asyncio.create_task(_call_with_random_sleep(i, context)) for i, context in enumerate(contexts)]

    indexed_results: list[tuple[int, dict[str, Any]]] = []
    completed = 0
    total = len(contexts)
    start_time = time.time()
    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        indexed_results.append((idx, result))
        completed += 1
        if completed % 50 == 0 or completed == total:
            elapsed = time.time() - start_time
            success_rate = sum(1 for _, r in indexed_results if r['status'] == 'success') / completed * 100
            print(f"已完成 {completed} / {total} 个请求，耗时 {elapsed:.2f} 秒，成功率 {success_rate:.2f}%")

    indexed_results.sort(key=lambda item: item[0])
    results = [result for _, result in indexed_results]
    return results

def batch_call_api_async_wrapper(model_name: str, contexts: list[str], max_concurrency: int = 5) -> list[dict[str, Any]]:
    """提供同步接口包装，内部运行异步批处理，便于平滑迁移"""

    try:
        print("使用异步批处理API调用...")
        return asyncio.run(batch_call_api_async(model_name, contexts, max_concurrency))
    except RuntimeError as exc:
        # 如果已有事件循环在运行（如在部分笔记本环境），回退到旧同步实现
        print(f"异步批处理在当前上下文不可用（{exc}），回退到线程并发实现。")
        return batch_call_api(model_name, contexts, max_workers=max_concurrency)