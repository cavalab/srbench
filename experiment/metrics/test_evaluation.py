from .evaluation import *

# define local namespace dictionary
SEIR_VARS = {}
for s in ['dS', 'dE', 'dI', 'dR', 
          'x_S', 'x_E', 'x_I', 'x_R', 'x_rzero', 'x_gamma',
                  'x_sigma', 'x_c', 'x_N', 'x_Aux']:
        SEIR_VARS.update({ s: sp.Symbol(s) })

seir_gts = {
    "dS": get_symbolic_model('-x_gamma*x_rzero*x_S*x_I/x_N',SEIR_VARS),
    "dE": get_symbolic_model('x_gamma*x_rzero*x_S*x_I/x_N - x_sigma*x_E', SEIR_VARS),
    "dI": get_symbolic_model('x_sigma*x_E-x_gamma*x_I+x_c*x_R*x_I/x_N', SEIR_VARS),
    "dR": get_symbolic_model('x_gamma*x_I-x_c*x_R*x_I/x_N', SEIR_VARS)
}

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

def test_simplicity_comparator():
    """Test comparison has the right sign"""

    big =  '2.35*( 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1 + log(30 * x3**2))'
    small =   '2.35*( 0.4 * x1 * x2 - 1.5 * x1 )'
    features = ['x1','x2','x3','x4','x5']
    assert simplicity(big, features) < simplicity(small, features) 