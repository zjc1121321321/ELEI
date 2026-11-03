from main_loop import main
import os

if __name__ == "__main__":
    from config import DATA_CONFIG, PROJECT_ROOT
    base_path = PROJECT_ROOT / DATA_CONFIG["group"] / DATA_CONFIG["dataset"]
    train_file = os.path.join(base_path, "train_points.npy")
    test_file = os.path.join(base_path, "test_points.npy")

    assert os.path.exists(train_file), f"训练集文件不存在：{train_file}"
    assert os.path.exists(test_file), f"测试集文件不存在：{test_file}"

    print("✅ 数据准备完成，开始执行符号回归主流程...\n")
    main()
    print("\n✅ 测试流程执行完毕，可以检查输出结果文件和终端日志。")

#本次实验用到的数据集来自真实的物理实验，请利用这一点。
