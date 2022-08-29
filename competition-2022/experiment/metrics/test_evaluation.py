# from .evaluation import (symbolic_equivalence, SEx_IR_VARS, EF_VARx_S,
#                          exact_formula_synthetic, seir_gts,
#                          simplicity, feature_absence_score
#                        )
from .evaluation import *

def test_seir_true():
    """SEIR models evaluate as solutions"""

    pred_models = [
        ("dS", '-x_gamma*x_rzero*x_S*x_I/x_N'),
        ("dE", 'x_gamma*x_rzero*x_S*x_I/x_N - x_sigma*x_E'),
        ("dI", 'x_sigma*x_E-x_gamma*x_I+x_c*x_R*x_I/x_N'),
        ("dR", 'x_gamma*x_I-x_c*x_R*x_I/x_N'),
        ("dE", 'x_E*(-x_sigma) + x_gamma*x_S*x_I/(x_N/x_rzero)'),
        ("dE", '(-x_E*x_sigma**2 + x_gamma*x_sigma*x_rzero*x_S*x_I/x_N)/x_sigma')
    ]

    for k,v in pred_models:
        result = symbolic_equivalence(seir_gts[k], v, SEIR_VARS)
        assert result['equivalent']

def test_seir_true_offset():
    """SEIR models evaluate as solutions when offset"""

    pred_models = [
        ("dS", '-x_gamma*x_rzero*x_S*x_I/x_N+1.80'),
        ("dE", 'x_gamma*x_rzero*x_S*x_I/x_N - x_sigma*x_E - 3.12'),
        ("dI", 'x_sigma*x_E-x_gamma*x_I+x_c*x_R*x_I/x_N + 678.0'),
        ("dR", 'x_gamma*x_I-x_c*x_R*x_I/x_N-0.043'),
    ]

    for k,v in pred_models:
        result = symbolic_equivalence(seir_gts[k], v, SEIR_VARS)
        assert result['equivalent']

def test_seir_true_scale():
    """SEIR models evaluate as solutions when scaled"""

    pred_models = {
        ("dS", '-x_gamma*x_rzero*x_S*x_I/x_N*1.80'),
        ("dE", '(x_gamma*x_rzero*x_S*x_I/x_N - x_sigma*x_E) * 3.12'),
        ("dI", '(x_sigma*x_E-x_gamma*x_I+x_c*x_R*x_I/x_N)/678.0'),
        ("dR", '(x_gamma*x_I-x_c*x_R*x_I/x_N)*0.043'),
        ("dE", 'x_E*(-3*x_sigma) + x_gamma*x_S*x_I/(0.33333*x_N/x_rzero)'),
    }

    for k,v in pred_models:
        result = symbolic_equivalence(seir_gts[k], v, SEIR_VARS)
        assert result['equivalent']

def test_seir_true_round():
    """SEIR models evaluate as solutions when rounded"""

    pred_models = {
        "dE": '(x_gamma*x_rzero*x_S*x_I/x_N*1.0003 - x_sigma*x_E*0.9999)',
        "dI": 'x_sigma*x_E*0.9999-x_gamma*x_I+x_c*x_R*x_I/x_N*1.0001',
        "dR": '(x_gamma*x_I*1.0004-x_c*x_R*x_I/x_N)*0.043',
    }

    for k,v in pred_models.items():
        result = symbolic_equivalence(seir_gts[k], v, SEIR_VARS)
        assert result['equivalent']


def test_exactformula_true():
    """EF models evaluate as solutions"""

    pred_models = exact_formula_synthetic_str

    for k,v in pred_models.items():
        result = symbolic_equivalence(exact_formula_synthetic[k], v, EF_VARS)

def test_exactformula_true_offset():
    """EF models evaluate as solutions with offset"""

    pred_models = { 
        "easy": '2.35 +0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1 + log(30 * x3**2)',
        "medium": '0.67-(0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))',
        "hard": '9+(5.5 * sin(x1 + x2) + 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))'
    }

    for k,v in pred_models.items():
        result = symbolic_equivalence(exact_formula_synthetic[k], v, EF_VARS)

def test_exactformula_true_scale():
    """Exactformula models evaluate as solutions when scaled"""

    pred_models = { 
        "easy": '2.35*( 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1 + log(30 * x3**2))',
        "medium": '0.67*(0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))',
        "hard": '9*(5.5 * sin(x1 + x2) + 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))'
    }

    for k,v in pred_models.items():
        result = symbolic_equivalence(exact_formula_synthetic[k], v, EF_VARS)

def test_simplicity_comparator():
    """Test comparison has the right sign"""

    big =  '2.35*( 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1 + log(30 * x3**2))'
    small =   '2.35*( 0.4 * x1 * x2 - 1.5 * x1 )'
    features = ['x1','x2','x3','x4','x5']
    assert simplicity(big, features) < simplicity(small, features) 

def test_featureselection():
    """Any odd numbered features count against score"""

    model_scores = {
        '2.35*(0.4*x1*x2-1.5*x1+2.5*x2+1+log(30*x3**2))': (2-1)/2,
        '2.35*( 0.4 * x1 * x2 - 1.5 * x1 )': (2-1)/2,
        '2.35*( 0.4 * x2 * x2 - 1.5 * x2 )': (2-1)/2,
        '2.35*( 0.4 * x2 * x2 - 1.5 * x4 )': 0,
        '2.35*( 0.4 * x1 * x3 - 1.5 * x5 )': 1,
# 
    }

    for m, s in model_scores.items():
        problem_name = 'featureselection'
        est = None
        X = np.random.rand(100,5)
        _, score = problem_specific_score(problem_name, est, X, pred_model=m) 
        assert score == s


def test_localopt():
    """Any odd numbered features count against score"""

    model_scores = {
        '2.35*(0.4*x1*x2-1.5*x1+2.5*x2+1+log(30*x3**2))': (6,1),
        '2.35*( 0.4 * x1 * x8 - 1.5 * x1 )': (10, (5-1)/5 ),
        '2.35*( 0.4 * x6 * x2 - 1.5 * x7 )': (10, (5-2)/5 ),
        '2.35*( 0.4 * x9 * x8 - 1.5 * x6 )': (10, (5-3)/5 ),
        '2.35*( 0.4 * x5 * x6 - 1.5 * x7 + sin(x8) - log(x9*x1) )': (9, 0)
    }

    for m, (nf,s) in model_scores.items():
        problem_name = 'localopt'
        est = None
        X = np.random.rand(100,nf)
        _, score = problem_specific_score(problem_name, est, X, pred_model=m) 
        assert score == s
