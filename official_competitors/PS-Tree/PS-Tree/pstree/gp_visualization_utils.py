from sympy import parse_expr
from pstree.common_utils import gene_to_string

infix_map = {
    'add_2': '+',
    'sub_2': '-',
    'multiply': '*',
    'protect_divide': '/',
}


def multigene_gp_to_string(label, regr):
    pipes = regr.pipelines
    cur_gene = None
    for i, g in enumerate(regr.best_pop):
        coef = pipes[label]['Ridge'].coef_[i]
        mean = pipes[label]['Scaler'].mean_[i]
        var = pipes[label]['Scaler'].scale_[i]
        if cur_gene is None:
            cur_gene = coef * (parse_expr(gene_to_string(g)) - mean) / var
        else:
            cur_gene += coef * (parse_expr(gene_to_string(g)) - mean) / var
    cur_gene += pipes[label]['Ridge'].intercept_
    return cur_gene
