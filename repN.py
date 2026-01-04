# encoding: utf8

import json
import statistics

def repN(full_text: str, n: int) -> float:
    # 预处理
    for ch in ["*", "-", "\n", "#", "-"]:
        full_text = full_text.replace(ch, "")
    text_len = len(full_text)

    # 计算rep-n的值
    n_gram_list = [full_text[i:i+n] for i in range(text_len - n + 1)]
    n_gram_set = set(n_gram_list)
    rep_n = 1 - len(n_gram_set) / (text_len - n + 1) if text_len - n + 1 > 0 else 0.0
    return rep_n


if __name__ == "__main__":
    file = r'result_with_frequency_penalty\KnowLogic_qwen3-8b_results_623.json'
    # file = r'E:\CL_homework\computational_linguistics_homework\answer_with_enumeration\merged_repetition_results_qwen3-8b_results.json'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    repn_list = []
    for tempd in data:
        full_text = tempd['response']
        repn_value = repN(full_text, 3)
        print(f"rep-3: {repn_value}")
        repn_list.append(repn_value)
    print(f"Average rep-3: {statistics.mean(repn_list)}, std: {statistics.stdev(repn_list)}")