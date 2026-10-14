import ast
import pdb
import re

import pandas as pd
import sympy
from read_file import read_file

# from sympy import *
from sympy import Float, Integer, Symbol, preorder_traversal, simplify
from sympy.parsing.sympy_parser import parse_expr
from yaml import Loader, load


###############################################################################
# fn definitions from the algorithms, written in sympy operators
def sub(x, y):
    return sympy.Add(x, -y)


# def division(x,y):
#     print('division')
#     if isinstance(y, sympy.Float):
#         if abs(y) < 0.00001:
#             return x
#             # result = sympy.Mod(x,1e-6+abs(y))
#             # if y < 0:
#             #     result = -result
#             # return result
#     return sympy.Mul(x,1/y)


# TODO: handle protected division
def div(x, y):
    return sympy.Mul(x, 1 / y)


def square(x):
    return sympy.Pow(x, 2)


def cube(x):
    return sympy.Pow(x, 3)


def quart(x):
    return sympy.Pow(x, 4)


def PLOG(x, base=None):
    if isinstance(x, sympy.Float):
        if x < 0:
            x = sympy.Abs(x)
    if base:
        return sympy.log(x, base)
    else:
        return sympy.log(x)


def PLOG10(x):
    return PLOG(x, 10)


def PSQRT(x):
    if isinstance(x, sympy.Float):
        if x < 0:
            return sympy.sqrt(sympy.Abs(x))
    return sympy.sqrt(x)


###############################################################################


def complexity(expr):
    c = 0
    for arg in preorder_traversal(expr):
        c += 1
    return c


def round_floats(ex1):
    ex2 = ex1

    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, Integer(0))
            else:
                ex2 = ex2.subs(a, Float(round(a, 3), 3))
    return ex2


################################################################################
# currently the MRGP model is put together incorrectly. this set of functions
# corrects the MRGP model form so that it can be fed to sympy and simplified.
################################################################################
def add_commas(model):
    return "".join([m + "," if not m.endswith("(") else m for m in model.split()])[:-1]


def decompose_mrgp_model(model_str):
    """split mrgp model into its betas and model parts"""
    # get betas
    betas = [
        float(b[0])
        for b in re.findall(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\*", model_str)
    ]
    print("betas:", betas)
    # get form
    submodel = re.sub(
        pattern=r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?\*", repl=r"", string=model_str
    )
    return betas, submodel  # new_model


def print_model(node):
    if hasattr(node, "func"):
        model_str = node.func.id + "("
    elif hasattr(node, "id"):
        # model_str = node.id
        return node.id
    else:
        pdb.set_trace()
    if hasattr(node, "args"):
        i = 0
        for arg in node.args:
            model_str += print_model(arg)
            i += 1
            if i < len(node.args):
                model_str += ","
        model_str += ")"

    # print('print_model::',model_str)
    return model_str


def add_betas(node, betas):
    beta = betas[0]
    betas.pop(0)
    if float(beta) > 0:
        model_str = str(beta) + "*" + print_model(node)
        i = 1
    else:
        # print('filtering fn w beta=',beta)
        model_str = ""
        i = 0
    if hasattr(node, "args"):
        for arg in node.args:
            submodel = add_betas(arg, betas)
            if submodel != "":
                model_str += "+" if i != 0 else ""
                model_str += submodel
                i += 1
    # print('add_betas::',model_str)
    return model_str


################################################################################


def clean_pred_model(model_str, dataset, est_name):
    mrgp = "MRGP" in est_name

    model_str = model_str.strip()

    if mrgp:
        model_str = model_str.replace("+", "add")
        model_str = add_commas(model_str)
        betas, model_str = decompose_mrgp_model(model_str)

    X, labels, features = read_file(dataset)

    local_dict = {k: Symbol(k) for k in features}
    new_model_str = model_str
    # Rename generic feature placeholders. Model strings are expected to use
    # dataset feature names, and some datasets genuinely have names such as
    # ``x1`` and ``x2``. In that case, treating the names as placeholders would
    # corrupt an already-correct model.
    placeholder_pattern = r"\b(?:x_|X_|x|X)(\d+)\b"
    placeholder_tokens = {
        match.group(0) for match in re.finditer(placeholder_pattern, new_model_str)
    }
    feature_names = {str(feature) for feature in features}
    index_offset = int( # Some methods start counting features at 1
        any(name in est_name.lower() for name in ["mrgp", "operon", "dsr"])
    )

    def feature_name(index):
        feature_index = index - index_offset
        if 0 <= feature_index < len(features):
            return str(features[feature_index])
        return None

    # If every placeholder-looking identifier is an actual feature name, the
    # string is already in the documented dataset-name form. This intentionally
    # favors preserving a compliant model in the otherwise ambiguous x1/x2 case.
    if not placeholder_tokens.issubset(feature_names):

        def replace_placeholder(match):
            replacement = feature_name(int(match.group(1)))
            return replacement if replacement is not None else match.group(0)

        new_model_str = re.sub(placeholder_pattern, replace_placeholder, new_model_str)

    # Array-style placeholders cannot be dataset feature names, so they can
    # always be normalized. Keep this separate from identifier replacement.
    def replace_array_placeholder(match):
        replacement = feature_name(int(match.group(1)))
        return replacement if replacement is not None else match.group(0)

    new_model_str = re.sub(
        r"x\[:,\s*(\d+)\s*\]", replace_array_placeholder, new_model_str
    )
    new_model_str = re.sub(
        r"x\[\s*(\d+)\s*\]", replace_array_placeholder, new_model_str
    )
    # operators
    new_model_str = new_model_str.replace("^", "**")
    # GP-GOMEA
    new_model_str = new_model_str.replace("p/", "/")
    new_model_str = new_model_str.replace("plog", "PLOG")
    new_model_str = new_model_str.replace("aq", "/")
    # MRGP
    new_model_str = new_model_str.replace("mylog", "PLOG")
    # ITEA
    new_model_str = new_model_str.replace("sqrtAbs", "PSQRT")
    # new_model_str = re.sub(pattern=r'sqrtAbs\((.*?)\)',
    #        repl=r'sqrt(abs(\1))',
    #        string=new_model_str
    #       )
    new_model_str = new_model_str.replace("np.", "")
    # ellyn & FEAT
    new_model_str = new_model_str.replace("|", "")
    new_model_str = new_model_str.replace("log", "PLOG")
    new_model_str = new_model_str.replace("sqrt", "PSQRT")

    # AIFeynman
    new_model_str = new_model_str.replace("pi", "3.1415926535")

    local_dict.update(
        {
            "add": sympy.Add,
            "mul": sympy.Mul,
            "max": sympy.Max,
            "min": sympy.Min,
            "sub": sub,
            "div": div,
            "square": square,
            "cube": cube,
            "quart": quart,
            "PLOG": PLOG,
            "PLOG10": PLOG,
            "PSQRT": PSQRT,
        }
    )
    # BSR
    # get rid of square brackets
    new_model_str = new_model_str.replace("[", "").replace("]", "")

    print("parsing", new_model_str)
    if mrgp:
        mrgp_ast = ast.parse(new_model_str, "", "eval")
        new_model_str = add_betas(mrgp_ast.body, betas)
        assert len(betas) == 0

    print(local_dict)
    model_sym = parse_expr(new_model_str, local_dict=local_dict)
    print("round_floats")
    model_sym = round_floats(model_sym)
    print("rounded:", model_sym)
    print("simplify...")
    model_sym = simplify(model_sym, ratio=1)
    print("simplified:", model_sym)
    return model_sym


def get_sym_model(dataset, return_str=True):
    """return sympy model from dataset metadata"""
    metadata = load(
        open("/".join(dataset.split("/")[:-1]) + "/metadata.yaml", "r"), Loader=Loader
    )
    df = pd.read_csv(dataset, sep="\t")
    features = [c for c in df.columns if c != "target"]
    #     print('features:',df.columns)
    description = metadata["description"].split("\n")
    model_str = [ms for ms in description if "=" in ms][0].split("=")[-1]
    model_str = model_str.replace("pi", "3.1415926535")
    if return_str:
        return model_str

    # pdb.set_trace()
    # handle feynman problem constants
    #     print('model:',model_str)
    model_sym = parse_expr(model_str, local_dict={k: Symbol(k) for k in features})
    model_sym = round_floats(model_sym)
    #     print('sym model:',model_sym)
    return model_sym


def get_sympy_model(model_str, dataset):
    model_str = model_str.replace("pi", "3.1415926535")

    df = pd.read_csv(dataset, sep="\t")
    features = [c for c in df.columns if c != "target"]

    model_sym = parse_expr(model_str, local_dict={k: Symbol(k) for k in features})
    model_sym = round_floats(model_sym)
    return model_sym


def rewrite_AIFeynman_model_size(model_str):
    """AIFeynman complexity was incorrect prior to version , update it here"""
    return complexity(parse_expr(model_str))
