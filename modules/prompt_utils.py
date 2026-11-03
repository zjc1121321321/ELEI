import re
from config import PROJECT_ROOT

def build_prompt(num_candidates, problem, samples, base=None, best_exprs=None, experience=None):
    """
    构建大模型输入的完整 prompt 内容。
    """
    if base is None:
        with open(PROJECT_ROOT / "base_prompt with SRSD_feynman.txt", "r", encoding="utf-8") as f:
            base = f.read()

    prompt = base.replace("{x}", str(num_candidates))
    prompt = prompt.replace("数据信息：", f"数据信息：\n{problem.strip()}")
    prompt = prompt.replace("代表性数据点（筛选后样本）：", f"代表性数据点（筛选后样本）：\n{samples.strip()}")

    # 插入历史最佳表达式（带分数）
    if best_exprs:
        if isinstance(best_exprs, list):
            best_expr_block = "\n".join(f"{i+1}. {expr}" for i, expr in enumerate(best_exprs))
        else:
            best_expr_block = str(best_exprs)
    else:
        best_expr_block = "（无）"

    # 插入经验总结
    experience_block = experience.strip() if experience else "（无）"

    # 使用字符串替换插入对应部分
    prompt = prompt.replace("3. 当前的函数与评分（R2越接近1越好）：",
                            f"3. 当前的函数骨架与评分（R2越接近1越好）：\n{best_expr_block}")
    prompt = prompt.replace("4. 经验信息（来自历史推理或思维链）：",
                            f"4. 经验信息（来自历史推理或思维链）：\n{experience_block}")


    return prompt


def extract_result(text):
    """
    从大模型输出中提取表达式列表和经验总结文本。
    """
    expressions = re.findall(r"f\([^)]+\)\s*=\s*.+", text)
    summary_match = re.search(r"经验总结\s*：\s*(.*)", text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else ""
    return expressions, summary


# -------------------------
# English version functions
# -------------------------
def build_prompt_en(num_candidates, problem, samples, base=None, best_exprs=None, experience=None, template_path=None):
    """
    Build the complete prompt content for LLM input (English version).
    """
    if base is None:
        template_path = template_path or PROJECT_ROOT / "base_prompt with SRSD_feynman_en.txt"
        with open(template_path, "r", encoding="utf-8") as f:
            base = f.read()

    prompt = base.replace("{x}", str(num_candidates))
    prompt = prompt.replace("1. Data Information:", f"1. Data Information:\n{problem.strip()}")
    prompt = prompt.replace("2. Representative Data Points (filtered samples):", f"2. Representative Data Points (filtered samples):\n{samples.strip()}")

    # Insert historical best expressions (with scores)
    if best_exprs:
        if isinstance(best_exprs, list):
            best_expr_block = "\n".join(f"{i+1}. {expr}" for i, expr in enumerate(best_exprs))
        else:
            best_expr_block = str(best_exprs)
    else:
        best_expr_block = "(None)"

    # Insert experience summary
    experience_block = experience.strip() if experience else "(None)"

    # Use string replacement to insert corresponding parts
    prompt = prompt.replace("3. Current Functions and Scores (R2 closer to 1 is better):",
                            f"3. Current Function Skeletons and Scores (R2 closer to 1 is better):\n{best_expr_block}")
    prompt = prompt.replace("4. Empirical Information (from historical reasoning or chain-of-thought):",
                            f"4. Empirical Information (from historical reasoning or chain-of-thought):\n{experience_block}")

    return prompt


def extract_result_en(text):
    """
    Extract expression list and experience summary text from LLM output (English version).
    """
    expressions = re.findall(r"f\([^)]+\)\s*=\s*.+", text)
    summary_match = re.search(r"Experience Summary\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    summary = summary_match.group(1).strip() if summary_match else ""
    return expressions, summary
