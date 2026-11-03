import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# -------------------------
# 变量含义读取函数
# -------------------------
def get_variable_description(dataset_name, excel_path="data-info.csv"):
    """
    给定数据集名称，从 Excel 中读取变量含义，返回一个字符串列表。
    """
    try:
        df = pd.read_csv(excel_path)
        df.set_index("dataset", inplace=True)

        if dataset_name not in df.index:
            return [f"[{dataset_name}] 未在 Excel 中找到。"]

        row = df.loc[dataset_name]
        lines = []
        if pd.notna(row.get("y", "")):
            lines.append(f"y —— {row['y']}")

        for col in row.index:
            if col.startswith("x") and pd.notna(row[col]):
                lines.append(f"{col} —— {row[col]}")
        lines.append("")
        return lines
    except Exception as e:
        return [f"[{dataset_name}] 无法读取变量说明：{str(e)}"]

# -------------------------
# 关键点提取器
# -------------------------
class KeyPointsExtractor:
    def __init__(self, X, y, n_points=30):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_points = n_points

    def extract_key_points(self):
        if len(self.X) <= self.n_points:
            return self.X, self.y

        kmeans = KMeans(n_clusters=self.n_points, random_state=42).fit(self.X)
        centers = kmeans.cluster_centers_

        indices = []
        for center in centers:
            distances = np.linalg.norm(self.X - center, axis=1)
            indices.append(np.argmin(distances))

        key_X = self.X[indices]
        key_y = self.y[indices]
        return key_X, key_y

# -------------------------
# 数据集概括函数
# -------------------------
def summarize_dataset(X, y, dataset_name=None, n_key_points=30, excel_path="data-info.csv"):
    assert len(X.shape) == 2 and len(y.shape) == 1

    info_lines = []

    # 变量含义
    if dataset_name is not None:
        info_lines += get_variable_description(dataset_name, excel_path=excel_path)

    # 每个变量的取值范围
    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_min, xi_max = np.min(xi), np.max(xi)
        info_lines.append(f"x{i} 的取值范围为 [{xi_min:.4e}, {xi_max:.4e}]")
    y_min, y_max = np.min(y), np.max(y)
    info_lines.append(f"y 的取值范围为 [{y_min:.4e}, {y_max:.4e}]")


    # 提取关键点
    extractor = KeyPointsExtractor(X, y, n_points=n_key_points)
    key_X, key_y = extractor.extract_key_points()

    # 格式化关键点
    all_values = np.concatenate([X.flatten(), y.flatten()])
    use_sci = (np.abs(all_values).max() > 1e5) or ((np.abs(all_values) < 1e-5) & (all_values != 0)).any()
    fmt = "{:.6e}" if use_sci else "{:.6f}"

    sample_lines = []
    for xi, yi in zip(key_X, key_y):
        x_str = ", ".join(fmt.format(v) for v in xi)
        y_str = fmt.format(yi)
        sample_lines.append(f"[{x_str}] => {y_str}")

    return {
        "info": "\n".join(info_lines),
        "samples": "\n".join(sample_lines)
    }


# -------------------------
# English version functions
# -------------------------
def get_variable_description_en(dataset_name, excel_path="data-info.csv"):
    """
    Given a dataset name, read variable descriptions from Excel and return a list of strings (English version).
    """
    try:
        df = pd.read_csv(excel_path)
        df.set_index("dataset", inplace=True)

        if dataset_name not in df.index:
            return [f"[{dataset_name}] not found in Excel."]

        row = df.loc[dataset_name]
        lines = []
        if pd.notna(row.get("y", "")):
            lines.append(f"y -- {row['y']}")

        for col in row.index:
            if col.startswith("x") and pd.notna(row[col]):
                lines.append(f"{col} -- {row[col]}")
        lines.append("")
        return lines
    except Exception as e:
        return [f"[{dataset_name}] Unable to read variable description: {str(e)}"]


def summarize_dataset_en(X, y, dataset_name=None, n_key_points=30, excel_path="data-info.csv"):
    """
    Summarize the dataset and extract key information (English version).
    """
    assert len(X.shape) == 2 and len(y.shape) == 1

    info_lines = []

    # Variable descriptions
    if dataset_name is not None:
        info_lines += get_variable_description_en(dataset_name, excel_path=excel_path)

    # Value range for each variable
    for i in range(X.shape[1]):
        xi = X[:, i]
        xi_min, xi_max = np.min(xi), np.max(xi)
        info_lines.append(f"x{i} value range: [{xi_min:.4e}, {xi_max:.4e}]")
    y_min, y_max = np.min(y), np.max(y)
    info_lines.append(f"y value range: [{y_min:.4e}, {y_max:.4e}]")

    # Extract key points
    extractor = KeyPointsExtractor(X, y, n_points=n_key_points)
    key_X, key_y = extractor.extract_key_points()

    # Format key points
    all_values = np.concatenate([X.flatten(), y.flatten()])
    use_sci = (np.abs(all_values).max() > 1e5) or ((np.abs(all_values) < 1e-5) & (all_values != 0)).any()
    fmt = "{:.6e}" if use_sci else "{:.6f}"

    sample_lines = []
    for xi, yi in zip(key_X, key_y):
        x_str = ", ".join(fmt.format(v) for v in xi)
        y_str = fmt.format(yi)
        sample_lines.append(f"[{x_str}] => {y_str}")

    return {
        "info": "\n".join(info_lines),
        "samples": "\n".join(sample_lines)
    }
