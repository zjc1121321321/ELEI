from sympy import sympify
import threading
from sympy import simplify
def safe_sympify(expr_str, locals_dict=None, timeout=10):
    result = {}

    def run():
        try:
            result['res'] = sympify(expr_str, locals=locals_dict)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⏰ sympify 超时，表达式被跳过：", expr_str)
        return None
    if 'error' in result:
        print(f"❌ sympify 失败: {type(result['error']).__name__}: {result['error']}")
        return None

    return result['res']

def safe_simplify(expr, timeout=10):
    result = {}

    def run():
        try:
            result['res'] = simplify(expr)
        except Exception as e:
            result['error'] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⏰ simplify 超时，直接跳过此表达式")
        return expr  # 超时，返回原始表达式
    if 'error' in result:
        print(f"❌ simplify 失败：{type(result['error']).__name__}: {result['error']}")
        return expr
    return result['res']

# -----------------------------------------
# 表达式结构提取函数
# -----------------------------------------
def extract_structure(expr_str):
    """
    去除表达式中的具体常数（如 C1、C2），仅保留结构骨架。
    用于结构去重时的键。
    """
    try:
        expr = safe_sympify(expr_str)

        # 将所有常数替换为 1（只要是以 C 开头的符号）
        consts = [s for s in expr.free_symbols if str(s).startswith("C")]
        expr_structure = expr.subs({c: 1 for c in consts})

        # 返回简化后的结构字符串
        return str(safe_simplify(expr_structure, timeout=5))
    except Exception as e:
        print(f"[结构提取失败] {expr_str} -> {e}")
        return expr_str  # 如果失败，则返回原表达式字符串作为 fallback