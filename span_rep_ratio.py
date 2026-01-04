# -*- coding: utf-8 -*-

import ast
import re
import json
import random
import copy
from collections import defaultdict
import time



def get_span_ratio(
    text,
    n_min=2,
    n_max=50,
    count_thred = 5,
    length_thred = 15
):
    # ---------- 预处理 ----------
    for ch in ["*", "-", "\n", "#", "-"]:
        text = text.replace(ch, "")
    text_len = len(text)

    # ---------- Phase 1：枚举 n-gram + 记录全局信息 ----------
    ngram_positions = defaultdict(list)

    for n in range(n_min, n_max + 1):
        for i in range(text_len - n + 1):
            gram = text[i:i+n]
            ngram_positions[gram].append(i)

    # 只保留重复 n-gram
    repeated = {
        g: pos for g, pos in ngram_positions.items()
        if len(pos) >= 2
    }

    # 预先计算全局事实
    repeated_info = {}
    for g, pos in repeated.items():
        n = len(g)
        repeated_info[g] = {
            "positions": pos,
            "count": len(pos),
            "n": n,
            "span_start": min(pos),
            "span_end": max(pos) + n
        }

    # ---------- Phase 2：按长的算（结构消解） ----------
    sorted_ngrams = sorted(repeated_info.keys(), key=len, reverse=True)
    final_ngrams = {}

    def is_contained(short_g, short_info, long_g, long_info):
        for sp in short_info["positions"]:
            if not any(
                lp <= sp and lp + long_info["n"] >= sp + short_info["n"]
                for lp in long_info["positions"]
            ):
                return False
        return True

    for g in sorted_ngrams:
        info = repeated_info[g]
        drop = False
        for lg, lg_info in final_ngrams.items():
            if g in lg and is_contained(g, info, lg, lg_info):
                drop = True
                break
        if not drop:
            final_ngrams[g] = info

    # ---------- Phase 3：全局统计指标 ----------
    result = {}

    for g, info in final_ngrams.items():
        n = info["n"]
        positions = info["positions"]

        span_start = info["span_start"]
        span_end = info["span_end"]
        span_len = span_end - span_start

        # 字符级 coverage（合并区间）
        intervals = sorted((p, p + n) for p in positions)
        covered = 0
        cs, ce = intervals[0]

        for s, e in intervals[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                covered += ce - cs
                cs, ce = s, e
        covered += ce - cs

        result[g] = {
            "length": n,
            "count": info["count"],
            "span": (span_start, span_end),
            "coverage": covered,
            "span_ratio": round(covered / span_len, 4),
            "text_ratio": round(covered / text_len, 4)
        }

    # ---------- Phase 4：全局排序 ----------
    result = dict(
        sorted(
            result.items(),
            key=lambda x: (
                x[1]["span_ratio"],
                x[1]["count"],
                x[1]["length"],
                x[1]["text_ratio"],
            ),
            reverse=True
        )
    )

    # print(result)


    temp_lst = []
    for tdict_k, tdict_v in result.items():
        if (tdict_v['length'] > length_thred) and (tdict_v['count'] > count_thred):
            return tdict_v['span_ratio']

    return 0

curr_file = r'result_with_frequency_penalty\KnowLogic_qwen3-0.6b_results_623.json'
with open(curr_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

span_ratio_list = []
for tempd in data:
    full_text = tempd['response']
    span_ratio = get_span_ratio(text = full_text)
    print(f"Span ratio: {span_ratio}")
    span_ratio_list.append(span_ratio)
print(f"Average span ratio: {sum(span_ratio_list)/len(span_ratio_list)}")


# model = "qwen3-0.6b"
# count_thred = 10
# length_thred = 10
# file_name = f"D:\GZ_workspace\enumeration_repetition\computational_linguistics_homework-main\original_res\{model}.json"
# with open(file_name, 'r', encoding='utf-8') as file:
#     data = json.load(file)
# # file_name = f"D:\\GZ_workspace\\enumeration_repetition\\computational_linguistics_homework-main\\answer_with_enumeration\\merged_repetition_results_{model}_results.json"
# # with open(file_name, 'r', encoding='utf-8') as file:
# #     data = json.load(file)

# error_lst = []
# res_dict = {}
# for temp_i in range(len(data)):
# # for temp_i in range(3):

#     # print(temp_i)

#     # try:
#         # if temp_i%5 == 0:
#         #     time.sleep(10)

#         # temp_index = data[temp_i]['original']['index']
#         temp_index = temp_i
#         print(temp_index)
        
#         ratio = get_span_ratio(text = data[temp_i]['response'], count_thred = count_thred, length_thred = length_thred)
#         res_dict[temp_index] = ratio

#     # except:
#     #     error_lst.append(temp_index)
#     #     continue

# with open(f"eval_res_count{count_thred}_length{length_thred}/{model}_ori_ratio.json", 'w', encoding='utf8') as f:
#     json.dump(res_dict, f, ensure_ascii=False, indent=4)
# with open(f'eval_res_count{count_thred}_length{length_thred}/{model}_ori_ratio_error_lst.txt', 'w', encoding='utf-8') as file:
#     for item in error_lst:
#         file.write(str(item) + '\n')
# # with open(f"eval_res_count{count_thred}_length{length_thred}/{model}_ratio.json", 'w', encoding='utf8') as f:
# #     json.dump(res_dict, f, ensure_ascii=False, indent=4)
# # with open(f'eval_res_count{count_thred}_length{length_thred}/{model}_ratio_error_lst.txt', 'w', encoding='utf-8') as file:
# #     for item in error_lst:
# #         file.write(str(item) + '\n')