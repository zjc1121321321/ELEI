import numpy as np
import re
from sympy import symbols, sympify, lambdify, preorder_traversal, Pow, simplify, Symbol
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import threading


# -------------------------
# 超时保护工具
# -------------------------
def run_with_timeout(func, args=(), kwargs=None, timeout=10):
    """
    在指定超时时间内运行函数，超时则返回 None。
    适用于 Windows 系统（不依赖 signal）。
    """
    if kwargs is None:
        kwargs = {}
    
    result = [None]
    exception = [None]
    
    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # 超时了，线程仍在运行
        return None, TimeoutError(f"函数执行超时 ({timeout}s)")
    
    if exception[0] is not None:
        return None, exception[0]
    
    return result[0], None


def curve_fit_with_timeout(fit_func, X_flat, y, p0, timeout=5, **kwargs):
    """
    带超时保护的 curve_fit 调用
    """
    def do_fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return curve_fit(fit_func, X_flat, y, p0=p0, **kwargs)
    
    result, error = run_with_timeout(do_fit, timeout=timeout)
    if error is not None:
        raise error
    return result


# -------------------------
# 表达式复杂度评分类
# -------------------------
class ComplexityScorer:
    def __init__(self, alpha=1.0, beta=0.5):
        self.alpha = alpha  # 精度权重
        self.beta = beta    # 复杂度权重

    def count_ops(self, expr):
        """统计表达式中的操作符数量"""
        op_count = 0
        for node in preorder_traversal(expr):
            if node.is_Add or node.is_Mul:
                op_count += len(node.args) - 1
            elif isinstance(node, Pow):
                try:
                    exp = node.args[1]
                    if abs(float(exp) - 1) < 0.01:
                        continue
                except:
                    pass
                op_count += 1
            elif node.func.__name__ in {"log", "sin", "cos", "exp", "sqrt", "tanh", "atan"}:
                op_count += 1
        return op_count

    def score(self, func, X, y):
        """计算综合评分：精度 + 复杂度"""
        try:
            y_pred = func(X)
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return float('inf')
            mse = mean_squared_error(y, y_pred)
            r2 = 1 - mse / np.var(y)
            r2 = max(min(r2, 1.0), -1.0)
        except:
            return float('inf')

        try:
            expr_str = func.__doc__
            expr = sympify(expr_str)
            complexity = self.count_ops(expr)
        except:
            return float('inf')

        return self.alpha * (1 - r2) * 2 + self.beta * np.log1p(complexity)


# -------------------------
# 表达式拟合器类
# -------------------------
class ExpressionFitter:
    def __init__(self, X, y, mse_threshold=0.1, x_scales=None, y_scale=1.0):
        self.X = X
        self.y = y
        self.mse_threshold = mse_threshold
        self.scorer = ComplexityScorer()
        self.x_scales = x_scales if x_scales else [1.0] * X.shape[1]
        self.y_scale = y_scale

    def parse_expression(self, expr_str):
        """解析表达式字符串，提取系数和变量"""
        if '=' in expr_str:
            expr_str = expr_str.split('=')[1].strip()

        expr_str = expr_str.strip().rstrip('`~;')
        
        # 将独立的 C 替换为带编号的 C0, C1...
        matches = list(re.finditer(r'\bC\b', expr_str))
        new_expr_str = expr_str
        offset = 0
        for idx, match in enumerate(matches):
            start, end = match.span()
            new_expr_str = new_expr_str[:start + offset] + f"C{idx}" + new_expr_str[end + offset:]
            offset += len(f"C{idx}") - (end - start)
        
        coef_symbols = sorted(set(re.findall(r'C\d+', new_expr_str)), key=lambda x: int(x[1:]))
        var_symbols = sorted(set(re.findall(r'x\d+', new_expr_str)), key=lambda x: int(x[1:]))
        
        coefs = symbols(coef_symbols)
        vars = symbols(var_symbols)
        
        try:
            # 使用超时保护解析表达式，防止复杂表达式卡死
            def do_sympify():
                return sympify(new_expr_str, locals={str(c): c for c in coefs})
            
            expr, error = run_with_timeout(do_sympify, timeout=10)  # 表达式解析超时10秒
            if error is not None or expr is None:
                print(f"[表达式解析超时或失败] {new_expr_str}")
                return None, None, None
        except Exception as e:
            print(f"[表达式解析失败] {new_expr_str} -> {e}")
            return None, None, None
            
        return expr, coefs, vars

    def make_callable_function(self, expr, vars):
        """将 SymPy 表达式转为可调用函数"""
        # 确保 vars 是列表/元组形式
        if not isinstance(vars, (list, tuple)):
            vars = [vars]
        func = lambdify(vars, expr, modules='numpy')
        n_vars = len(vars)
        def wrapped(x):
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return func(*x[:, :n_vars].T)
        wrapped.__doc__ = str(expr)
        return wrapped

    def fit(self, expr_str):
        """拟合表达式到数据"""
        parsed = self.parse_expression(expr_str)
        if parsed[0] is None:
            return {
                "original": expr_str,
                "expression": "解析失败",
                "RMSE": 1e5,
                "R2": -1.0
            }
        
        expr, coefs, vars = parsed
        
        # 确保 coefs 和 vars 是元组/列表形式
        if not isinstance(coefs, (list, tuple)):
            coefs = (coefs,) if coefs else ()
        if not isinstance(vars, (list, tuple)):
            vars = (vars,) if vars else ()
        
        n_coefs = len(coefs)
        n_vars = len(vars)
        
        # 创建用于 curve_fit 的函数
        # curve_fit 要求函数签名为 f(x, *params)
        base_func = lambdify(list(vars) + list(coefs), expr, modules='numpy')
        
        def fit_func(X_flat, *params):
            """curve_fit 兼容的函数"""
            try:
                with np.errstate(all='ignore'):  # 抑制所有数值警告
                    # X_flat 是展平的数据，需要重塑
                    X = X_flat.reshape(n_vars, -1)
                    args = list(X) + list(params)
                    result = base_func(*args)
                    # 处理无效值
                    result = np.where(np.isfinite(result), result, 1e10)
                    return result
            except:
                return np.full(X_flat.shape[0] // n_vars, 1e10)

        # 准备数据：curve_fit 需要 xdata 格式为 (n_vars, n_samples)
        X_flat = self.X.T.flatten()
        
        # 多次随机初始化优化
        best_params = None
        best_loss = float('inf')
        total_timeout_count = 0
        max_timeouts = 3  # 连续超时3次则放弃
        
        for trial in range(10):
            if total_timeout_count >= max_timeouts:
                print(f"[优化超时] 连续{max_timeouts}次超时，跳过剩余尝试")
                break
                
            init_params = np.random.uniform(0.5, 2.0, size=n_coefs)
            try:
                # 使用带超时保护的 curve_fit
                popt, _ = curve_fit_with_timeout(
                    fit_func, 
                    X_flat, 
                    self.y,
                    p0=init_params,
                    timeout=10,  # 单次主优化超时10秒
                    maxfev=10000,
                    method='lm'  # Levenberg-Marquardt 算法
                )
                total_timeout_count = 0  # 成功则重置计数
                # 计算损失
                y_pred = fit_func(X_flat, *popt)
                loss = np.mean((y_pred - self.y) ** 2)
                
                if np.isfinite(loss) and loss < best_loss:
                    best_loss = loss
                    best_params = popt
            except TimeoutError:
                total_timeout_count += 1
                continue
            except Exception as e:
                # 尝试 trf 方法（带边界约束）
                try:
                    popt, _ = curve_fit_with_timeout(
                        fit_func, 
                        X_flat, 
                        self.y,
                        p0=init_params,
                        timeout=10,  # 单次主优化超时10秒
                        maxfev=10000,
                        method='trf',
                        bounds=(-1e6, 1e6)
                    )
                    total_timeout_count = 0
                    y_pred = fit_func(X_flat, *popt)
                    loss = np.mean((y_pred - self.y) ** 2)
                    
                    if np.isfinite(loss) and loss < best_loss:
                        best_loss = loss
                        best_params = popt
                except TimeoutError:
                    total_timeout_count += 1
                    continue
                except:
                    continue

        if best_params is None or best_loss > 1e9:
            return {
                "original": expr_str,
                "expression": "结构非常差",
                "RMSE": 1e5,
                "R2": -1.0
            }

        simplified_params = best_params.copy()

        # 剪枝逻辑：尝试将小系数置0
        if best_loss <= self.mse_threshold:
            for i in np.argsort(np.abs(simplified_params)):
                test_params = simplified_params.copy()
                test_params[i] = 0
                mask = test_params != 0
                if not np.any(mask):
                    continue

                # 创建剪枝后的拟合函数
                def fit_func_prune(X_flat, *sub_params):
                    full_params = test_params.copy()
                    full_params[mask] = sub_params
                    return fit_func(X_flat, *full_params)

                sub_init = test_params[mask]
                if len(sub_init) == 0:
                    continue
                    
                try:
                    # 剪枝优化也使用超时保护
                    popt_prune, _ = curve_fit_with_timeout(
                        fit_func_prune,
                        X_flat,
                        self.y,
                        p0=sub_init,
                        timeout=10,  # 剪枝优化超时10秒
                        maxfev=5000,
                        method='lm'
                    )
                    y_pred_prune = fit_func_prune(X_flat, *popt_prune)
                    new_loss = np.mean((y_pred_prune - self.y) ** 2)
                    
                    if np.isfinite(new_loss) and new_loss - best_loss < 0.01 * best_loss:
                        simplified_params = test_params.copy()
                        simplified_params[mask] = popt_prune
                        best_loss = new_loss
                except (TimeoutError, Exception):
                    continue

        # 构造最终表达式
        expr_fitted = expr
        for coef, val in zip(coefs, simplified_params):
            expr_fitted = expr_fitted.subs(coef, round(val, 8))

        # 计算最终指标
        func_final = self.make_callable_function(expr_fitted, vars)
        try:
            y_pred_final = func_final(self.X)
            final_loss = np.mean((y_pred_final - self.y) ** 2)
            rmse = np.sqrt(final_loss)
            r2 = 1 - final_loss / np.var(self.y)
        except:
            rmse = 1e5
            r2 = -1.0

        # 还原缩放
        for i, scale in enumerate(self.x_scales):
            if scale != 1.0:
                var = Symbol(f'x{i}')
                expr_fitted = expr_fitted.subs(var, var * scale)

        if self.y_scale != 1.0:
            expr_fitted = simplify(expr_fitted * self.y_scale)

        result = {
            "original": expr_str,
            "expression": str(expr_fitted),
            "RMSE": round(rmse, 6),
            "R2": round(r2, 6),
        }
        gc.collect()

        return result

    def evaluate_batch(self, expr_list, single_expr_timeout=90):
        """批量拟合多个表达式，每个表达式有独立的超时保护（默认90秒）"""
        results = []
        for idx, expr_str in enumerate(expr_list):
            try:
                # 为每个表达式的拟合添加总体超时保护
                result, error = run_with_timeout(self.fit, args=(expr_str,), timeout=single_expr_timeout)
                
                if error is not None:
                    if isinstance(error, TimeoutError):
                        print(f"[表达式拟合超时] ({idx+1}/{len(expr_list)}) {expr_str[:50]}...")
                        results.append({
                            "original": expr_str,
                            "expression": "拟合超时",
                            "RMSE": 1e5,
                            "R2": -1.0
                        })
                    else:
                        print(f"[表达式拟合失败] {expr_str} -> {error}")
                        results.append({
                            "original": expr_str,
                            "expression": "拟合失败",
                            "RMSE": 1e5,
                            "R2": -1.0
                        })
                elif result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[表达式拟合异常] {expr_str} -> {e}")
        return results

    def select_best(self, result_list, top_k=10):
        """根据结构去重，选出 Top-K 最佳表达式"""
        from modules.expression_utils import extract_structure
        expr_dict = {}
        for r in result_list:
            key = extract_structure(r['expression'])
            if key not in expr_dict or r['R2'] > expr_dict[key]['R2']:
                expr_dict[key] = r
        return sorted(expr_dict.values(), key=lambda x: (-x['R2'], x['RMSE']))[:top_k]


# -------------------------
# 训练/测试集评估
# -------------------------
def evaluate_on_train_and_test(expr_str, train_path, test_path, x_scales=None, y_scale=1.0):
    """在训练集和测试集上评估表达式"""
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # 对输入和输出数据进行缩放
    if x_scales:
        x_scales = np.array(x_scales)
        X_train = X_train / x_scales
        X_test = X_test / x_scales

    if y_scale != 1.0:
        y_train = y_train / y_scale
        y_test = y_test / y_scale

    # 解析表达式
    if '=' in expr_str:
        expr_str = expr_str.split('=')[1].strip()

    expr = sympify(expr_str)
    vars = sorted(expr.free_symbols, key=lambda x: str(x))
    func = lambdify(vars, expr, modules='numpy')

    def wrapped_func(X):
        return func(*X.T)

    wrapped_func.__doc__ = str(expr)

    scorer = ComplexityScorer()
    train_score = scorer.score(wrapped_func, X_train, y_train)
    test_score = scorer.score(wrapped_func, X_test, y_test)

    try:
        y_pred_train = wrapped_func(X_train)
        y_pred_test = wrapped_func(X_test)

        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    except:
        train_r2 = test_r2 = -1.0
        train_rmse = test_rmse = 1e5

    return (
        str(expr), 
        round(train_r2, 6), round(train_rmse, 6), 
        round(test_r2, 6), round(test_rmse, 6), 
        round(train_score, 4), round(test_score, 4)
    )


# -------------------------
# 测试与可视化
# -------------------------
def test_expression_fit(expr_str, X, y):
    """单个表达式的拟合测试 + 可视化"""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print(f"🎯 测试表达式骨架：{expr_str}")

    fitter = ExpressionFitter(X, y)
    result = fitter.fit(expr_str)

    print(f"✅ 拟合后表达式：{result['expression']}")
    print(f"📈 R²：{result['R2']:.6f}")
    print(f"📉 RMSE：{result['RMSE']:.6f}")

    # 检查拟合是否成功
    if result['R2'] < -0.99 or result['expression'] in ['解析失败', '结构非常差']:
        print("⚠️ 拟合失败，跳过可视化")
        print("=" * 60)
        return result

    # 可视化结果
    try:
        var_matches = re.findall(r'x\d+', result["expression"])
        if not var_matches:
            print("⚠️ 表达式中没有变量，跳过可视化")
            print("=" * 60)
            return result
            
        func = fitter.make_callable_function(
            sympify(result["expression"]), 
            symbols(sorted(set(var_matches), key=lambda x: int(x[1:])))
        )

        y_pred = func(X)
        
        # 检查预测值是否有效
        if not np.all(np.isfinite(y_pred)):
            print("⚠️ 预测值包含无效数值，跳过可视化")
            print("=" * 60)
            return result

        plt.figure(figsize=(8, 5))
        plt.plot(y, label="Ground Truth", linewidth=2)
        plt.plot(y_pred, label="Predicted", linestyle='--')
        plt.title("Expression Fitting Visualization")
        plt.xlabel("Sample Index")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

    print("=" * 60)
    return result


def main():
    data_path = "Data/srsd-feynman_medium/feynman-iii.17.37/train_points.npy"  
    expr = "f(x) = C * x0 + C*x1*cos(x2)"

    data = np.load(data_path)
    X, y = data[:, :-1], data[:, -1]

    test_expression_fit(expr, X, y)


if __name__ == "__main__":
    main()
