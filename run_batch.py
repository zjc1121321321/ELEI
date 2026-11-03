# main_batch.py

from main_loop import main  
import os
from config import DATA_CONFIG_LIST, PROJECT_ROOT

def run_batch():
    for idx, data_conf in enumerate(DATA_CONFIG_LIST):
        print(f"\n==============================")
        print(f"🚀 运行第 {idx + 1} 个数据集：{data_conf['dataset']}")
        print(f"📁 数据路径：{data_conf['group']}/{data_conf['dataset']}")

        base_path = PROJECT_ROOT / data_conf["group"] / data_conf["dataset"]
        train_file = os.path.join(base_path, "train_points.npy")
        test_file = os.path.join(base_path, "test_points.npy")

        assert os.path.exists(train_file), f"❌ 训练集文件不存在：{train_file}"
        assert os.path.exists(test_file), f"❌ 测试集文件不存在：{test_file}"
        print("✅ 数据检查通过，开始执行符号回归主流程...")

        main(data_conf)  # 运行一次
        print("✅ 当前数据集完成，可检查输出文件。")

if __name__ == "__main__":
    run_batch()
    print("\n✅ 所有实验已执行完毕，可以统一查看 results_*.txt 输出。")
