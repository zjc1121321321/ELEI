# config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Qwen3-Max is the strongest overall backend reported in the paper.
# Use ELEI_PROVIDER=deepseek for the alternative main-paper backend.
LLM_CONFIG = {
    "provider": os.getenv("ELEI_PROVIDER", "qwen"),
    "backends": {
        "qwen": {
            "model": "qwen3-max",
            "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "language": "zh",
        },
        "deepseek": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "language": "zh",
        },
        "openai": {
            "model": "o4-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "language": "en",
        },
        "llama3": {
            "model": "llama3",
            "base_url": "http://localhost:11434",
            "language": "en",
        },
    },
}
    
DATA_CONFIG_LIST =[
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.1" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.2" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.3" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.4" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.5" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.6" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.7" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.9" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.10" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.11" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.12" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.13" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.14" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.15" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.16" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.17" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.19" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-bonus.20" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.6.20" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.6.20a" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.6.20b" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.9.18" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.15.3t" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.15.3x" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.29.16" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.30.3" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.32.17" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.34.14" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.37.4" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.39.22" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.40.1" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.41.16" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.44.4" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-i.50.26" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.6.15a" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.6.15b" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.11.17" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.11.20" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.11.27" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.11.28" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.13.23" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.13.34" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.24.17" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.35.18" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.35.21" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-ii.36.38" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-iii.4.33" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-iii.9.52" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-iii.10.19" },
{"group": "Data/srsd-feynman_hard", "dataset": "feynman-iii.21.20" },


]

# 默认一个当前 DATA_CONFIG（会被 main_batch.py 自动覆盖）
DATA_CONFIG = DATA_CONFIG_LIST[0]

RUN_CONFIG = {
    "score_threshold": 0.999,
    "max_iterations": 9,
    "num_candidates": 5,
    "n_key_points": 30
}

