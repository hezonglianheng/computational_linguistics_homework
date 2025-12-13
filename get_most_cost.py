# encoding: utf8

import json

def main(json_path: str, ratio: float = 0.1):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    
    if not data:
        print("No data found.")
        return

    data_with_idx = [(idx, item) for idx, item in enumerate(data)]
    # 按照 cost 降序排序
    data_sorted = sorted(data_with_idx, key=lambda x: x[1].get('token_cost', 0), reverse=True)
    top_n = max(1, int(len(data_sorted) * ratio))
    most_cost_data = data_sorted[:top_n]
    print(f"Total samples: {len(data)}, Top {ratio*100}% samples count: {top_n}")
    print('-' * 50)
    print("Indices of most costly samples:")
    for d in most_cost_data:
        print(f"Index: {d[0]}, Token Cost: {d[1].get('token_cost', 0)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get most costly samples from JSON results.")
    parser.add_argument('json_path', type=str, help="Path to the JSON results file.")
    parser.add_argument('--ratio', type=float, default=0.1, help="Ratio of top costly samples to retrieve.")
    args = parser.parse_args()
    main(args.json_path, args.ratio)