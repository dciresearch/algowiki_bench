from sklearn.metrics import r2_score
import numpy as np
from sympy import lambdify
from sympy.parsing.latex import parse_latex
from pylatexenc.latexencode import unicode_to_latex
import re

vars_pattern = r"([^A-Za-z\\\\\{]|^)([a-z])"
vars_pattern = re.compile(vars_pattern)


def fix_vars(text):
    return vars_pattern.sub(r"\1{\2}", text)


o_notation_pattern = "O\((.+)\)"
o_notation_pattern = re.compile(o_notation_pattern)

latex_fix_dict = {
    "\\text{log}": "\\log",
    "\\log_2": "\\log",
    "log(": "\\log(",
    "max(": "\\max(",
    "min(": "\\min(",
}


def fix_latex_min_max(latex_str):
    pattern = r'\\m(in|ax)'
    cleaned = re.sub(pattern, r'M\1', latex_str)
    return cleaned


def extract_o_notation(text, isolate_vars=True):
    for k, v in latex_fix_dict.items():
        text = text.replace(k, v)
    text = text.replace("\\\\", "\\")
    candidate = o_notation_pattern.findall(text)
    res = text.replace("$", "").lower()
    if candidate:
        res = candidate[0].lower()
    res = fix_latex_min_max(res)
    if isolate_vars:
        res = fix_vars(res)
    return res


def logx(array: float, base: float = 2):
    return np.log(array) / np.log(2)


def frac(nom, denom):
    return nom/denom


def np_lambdify(varname, func):
    # print(func.free_symbols)
    subdict = {vn: varname for vn in func.free_symbols}
    func = func.subs(subdict)
    # print(func)
    lamb = lambdify(varname, func, modules=[{'log': logx}, 'numpy'])
    if func.is_constant():
        return lambda t: np.full_like(t, lamb(t))
    else:
        return lambda t: lamb(np.array(t))


def o_notation_trajectory(formula, variable='n', eval_set=np.arange(2, 10+1, 0.5, dtype=np.float64)):
    expr = parse_latex(formula)
    try:
        calc = np_lambdify('n', expr)
        return calc(eval_set)
    except Exception as e:
        print(e, formula)
        return np.zeros_like(eval_set)


def o_notation_score(pr, gt):
    pr_ext = extract_o_notation(pr)
    gt_ext = extract_o_notation(gt)
    try:
        pr = o_notation_trajectory(pr_ext)
        gt = o_notation_trajectory(gt_ext)

        mask = np.isfinite(pr) & np.isfinite(gt)
        pr = pr[mask]
        gt = gt[mask]
    except Exception as e:
        print("Parsing error")
        print(e)
        print(pr_ext, gt_ext)
        score = 0
        return score

    # print(pr, gt)
    try:
        score = np.clip(r2_score(gt, pr), 0, 1)
    except Exception as e:
        print("R2 error")
        print(e)
        print(pr_ext, gt_ext)
        print(pr, gt)
        score = 0
    return score
