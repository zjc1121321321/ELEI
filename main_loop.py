import os
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from config import DATA_CONFIG, RUN_CONFIG, PROJECT_ROOT
from modules.data_process1 import summarize_dataset,summarize_dataset_en
from modules.prompt_utils import build_prompt, extract_result,build_prompt_en, extract_result_en
from modules.models import query_model, get_backend_config
from modules.fitter_simple import ExpressionFitter, evaluate_on_train_and_test
from modules.state_io import save_state, load_state  # 新增模块保存/读取状态
from modules.scaling_utils import compute_scaling_factors

# 根据配置自动拼接训练与测试数据路径

def evaluate_and_save(best_exprs, train_path, test_path, x_scales=None, y_scale=1.0, n=5, flag=0, base_path="", experience="", DATA_CONFIG=None, duration=0, output_dir=None):
    results = []

    # 筛选超过阈值的表达式
    valid_exprs = [expr for expr in best_exprs if expr["R2"] >= RUN_CONFIG["score_threshold"]]

    # 如果没有超过阈值的，就用前3个兜底
    if not valid_exprs:
        print("⚠️ 没有表达式满足阈值条件，使用前三个表达式进行评分。")
        valid_exprs = sorted(best_exprs, key=lambda x: x["R2"], reverse=True)[:3]

    for expr_info in valid_exprs:
        expr_str = expr_info["expression"]

        try:
            expr_str, train_r2, train_rmse, test_r2, test_rmse, train_score, test_score = evaluate_on_train_and_test(
                expr_str=expr_str,
                train_path=train_path,
                test_path=test_path,
                x_scales=x_scales,
                y_scale=y_scale
            )

            results.append({
                "expression": expr_str,
                "train_r2": train_r2,
                "train_rmse": train_rmse,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "train_score": train_score,
                "test_score": test_score
            })

        except Exception as e:
            print(f"❌ 评估失败: {expr_str}，错误：{e}")

    # 按 test_score 排序并保留前 n 个
    top_results = sorted(results, key=lambda x: x["test_score"], reverse=False)[:n]

    # 格式化输出
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output = f"时间: {now}\n运行时长: {duration:.2f}秒\n轮数: {flag}\n数据路径: {base_path}\n经验信息: {experience}\n缩放情况: x{x_scales}, y[{y_scale}]\n\n"

    for idx, res in enumerate(top_results, 1):
        output += f"""Top {idx} 表达式：{res['expression']}
Train R² = {res['train_r2']:.6f}, RMSE = {res['train_rmse']:.6f}
Test  R² = {res['test_r2']:.6f}, RMSE = {res['test_rmse']:.6f}
Train score = {res['train_score']:.4f}, Test score = {res['test_score']:.4f}

"""

    # 写入到txt文件
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = Path(DATA_CONFIG['group']).name
    file_name = output_dir / f"results_{group_name}_{DATA_CONFIG['dataset']}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)


def get_data_paths():
    base = PROJECT_ROOT / DATA_CONFIG['group'] / DATA_CONFIG['dataset']
    train_path = os.path.join(base, "train_points.npy")
    test_path = os.path.join(base, "test_points.npy")
    return base, train_path, test_path

# 主流程函数

def main(config=""):
    start_time = time.time()
    if config=="":
        config=DATA_CONFIG 
    provider, backend = get_backend_config()
    if backend["language"] == "zh":
        summarize = summarize_dataset
        make_prompt = build_prompt
        parse_result = extract_result
    else:
        summarize = summarize_dataset_en
        make_prompt = build_prompt_en
        parse_result = extract_result_en
        if provider == "llama3":
            def make_prompt(**kwargs):
                return build_prompt_en(
                    **kwargs, template_path=PROJECT_ROOT / "base_prompt with SRSD_feynman_llama.txt"
                )

    base_path = PROJECT_ROOT / config['group'] / config['dataset']
    metadata_files = {
        "srsd-feynman_easy": "easy-info.csv",
        "srsd-feynman_medium": "medium-info.csv",
        "srsd-feynman_hard": "hard-info.csv",
        "sr-bench": "sr-bench-info.csv",
    }
    group_name = Path(config['group']).name
    if group_name not in metadata_files:
        raise ValueError(f"No variable metadata configured for dataset group: {group_name}")
    metadata_path = PROJECT_ROOT / metadata_files[group_name]
    run_dir = (PROJECT_ROOT / "outputs" / provider / group_name / config['dataset']
               / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "state.json"
    train_path = os.path.join(base_path, "train_points.npy")
    test_path = os.path.join(base_path, "test_points.npy")
    data = np.load(train_path)
    X_raw, y_raw = data[:, :-1], data[:, -1]
    X, y, x_scales, y_scale = compute_scaling_factors(X_raw, y_raw)
    fitter = ExpressionFitter(X, y, x_scales=x_scales, y_scale=y_scale)
    flag=0
    # 第一步：分析数据，提取 info 和关键样本
    summary = summarize(X, y, config['dataset'],RUN_CONFIG["n_key_points"],excel_path=metadata_path)
    info, samples = summary["info"], summary["samples"]

    # 保存 info 和 samples 到文件
    with open(run_dir / "info.txt", "w", encoding="utf-8") as f:
        f.write(info)
    with open(run_dir / "samples.txt", "w", encoding="utf-8") as f:
        f.write(samples)

    # 构造初始 prompt 并调用大模型
    prompt = make_prompt(num_candidates=RUN_CONFIG["num_candidates"],
                          problem=info, samples=samples)
    llm_output = query_model(prompt, provider=provider)
    expressions, experience = parse_result(llm_output)
    if not expressions:
        raise RuntimeError("No expressions were parsed from the initial LLM response.")

    # 表达式拟合与评分
    fitter = ExpressionFitter(X, y)
    results = fitter.evaluate_batch(expressions)

    # 挑选最优表达式并保存
    best_exprs = fitter.select_best(results)
    if not best_exprs:
        raise RuntimeError("No fitted expressions are available after initialization.")
    save_state(best_exprs, experience, info, samples, path=state_path)

    if best_exprs[0]["R2"] >= RUN_CONFIG["score_threshold"]:
        print(f"🎯 满足终止条件：R² = {best_exprs[0]['R2']:.4f} ≥ {RUN_CONFIG['score_threshold']}")
        flag=1

    # 多轮迭代优化表达式
    if flag==0:
        flag=1
        for i in range(1, RUN_CONFIG["max_iterations"] + 1):
            best_exprs, experience, info, samples = load_state(path=state_path)
            flag = flag+1
            print(f"================第{flag}轮================")
            prompt = make_prompt(num_candidates=RUN_CONFIG["num_candidates"],
                                problem=info, samples=samples,
                                best_exprs=best_exprs, experience=experience)
            llm_output = query_model(prompt, provider=provider)
            expressions, new_experience = parse_result(llm_output)

            results = fitter.evaluate_batch(expressions)
            best_exprs = fitter.select_best(results + best_exprs)  # 合并去重 + 排序
            save_state(best_exprs, new_experience, info, samples, path=state_path)

            experience = new_experience

            if best_exprs[0]["R2"] >= RUN_CONFIG["score_threshold"]:
                print(f"🎯 满足终止条件：R² = {best_exprs[0]['R2']:.4f} ≥ {RUN_CONFIG['score_threshold']}")
                break
        
    duration = time.time() - start_time
    evaluate_and_save(
    best_exprs=best_exprs,
    train_path=train_path,
    test_path=test_path,
    x_scales=x_scales,
    y_scale=y_scale,
    n=5,
    flag=flag,
    base_path=base_path,
    experience=experience,
    DATA_CONFIG=config,
    duration=duration,
    output_dir=run_dir)
    print(f"Results saved to: {run_dir}")

