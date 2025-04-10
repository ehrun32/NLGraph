import numpy as np
import os
import re
import json

def get_cycle_prediction(ans: str) -> str:
    ans = ans.strip().lower().replace("\n", " ").replace("  ", " ")

    no_patterns = [
        r"there is no cycle",
        r"no cycle (was|is) (detected|found)",
        r"graph does not contain a cycle",
        r"not.*form.*cycle",
        r"cycle.*not.*present",
        r"does not contain.*cycle",
        r"this graph has no cycle",
    ]

    yes_patterns = [
        r"there is a cycle",
        r"a cycle exists",
        r"forms a cycle",
        r"contains.*cycle",
        r"cycle is present",
        r"cycle was detected",
        r"this graph has a cycle",
    ]

    for pattern in yes_patterns:
        if re.search(pattern, ans):
            return "yes"

    for pattern in no_patterns:
        if re.search(pattern, ans):
            return "no"

    return "unknown"

def normalize_label(label: str) -> str:
    label = label.lower()
    if "no" in label and "cycle" in label:
        return "no"
    if "yes" in label or "there is a cycle" in label:
        return "yes"
    return "unknown"

def evaluate_run_with_ground_truth(folder_path, main_json_path):
    answers = np.load(os.path.join(folder_path, "answer.npy"), allow_pickle=True)
    with open(main_json_path, "r") as f:
        gt = json.load(f)

    correct = 0
    total = len(answers)
    wrong_examples = []

    for idx, ans in enumerate(answers):
        prediction = get_cycle_prediction(ans)
        ground_truth = normalize_label(gt[str(idx)]["answer"])

        if prediction == ground_truth:
            correct += 1
        else:
            wrong_examples.append((idx, prediction, ground_truth))

    print(f"Checked: {total} answers")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

    if wrong_examples:
        print("\nWrong predictions:")
        for idx, pred, gt_ans in wrong_examples[:10]:  # print only first 10
            print(f"Graph {idx}: predicted='{pred}', expected='{gt_ans}'")

# Example usage
if __name__ == "__main__":
    folder = "log/cycle/gpt-4o-easy-20250409---14-33-k-shot"
    main_json = "NLgraph/cycle/main.json"
    evaluate_run_with_ground_truth(folder_path=folder, main_json_path=main_json)