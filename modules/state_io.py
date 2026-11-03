import os
import json

def save_state(best_exprs, experience, problem, samples, path="state.json"):
    """
    保存当前搜索轮次的状态，包括：
    - 最优表达式列表（带分数）
    - 当前经验总结
    - 问题信息（info）与代表样本（samples）
    """
    if isinstance(best_exprs, str):
        best_exprs = [best_exprs]
    best_exprs = best_exprs[-15:]  # 最多保存最近的 15 个表达式

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "best_exprs": best_exprs,
            "experience": experience,
            "problem": problem,
            "samples": samples
        }, f, ensure_ascii=False, indent=2)


def load_state(path="state.json"):
    """
    读取已保存的轮次状态。
    若不存在 state.json，返回空值（首次运行用）。
    """
    if not os.path.exists(path):
        return [], "", "", ""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["best_exprs"], data["experience"], data["problem"], data["samples"]
