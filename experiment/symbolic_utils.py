import pdb
import pandas as pd
from yaml import load, Loader
import sympy
from sympy import Symbol, simplify, factor, Float, preorder_traversal
from sympy.parsing.sympy_parser import parse_expr
from read_file import read_file
import re

def complexity(expr):
    complexity=1
    for arg in expr.args:
        complexity += complexity(expr.args)
    return 1
        
def round_floats(ex1):
    ex2 = ex1
    for a in preorder_traversal(ex1):
        if isinstance(a, Float):
            ex2 = ex2.subs(a, round(a, 6))
    return ex2

def clean_pred_model(model_str, dataset, mrgp=False):
    if mrgp:
        model_str = process_mrgp(model_str)
        
    X, labels, features = read_file(dataset)
   
    local_dict = {k:Symbol(k) for k in features}
    new_model_str = model_str
    # rename features
    for i,f in enumerate(features): 
        new_model_str = new_model_str.replace('x'+str(i),f)
        new_model_str = new_model_str.replace('x_'+str(i),f)
        new_model_str = new_model_str.replace('X_'+str(i),f)
        new_model_str = new_model_str.replace('X'+str(i),f)
        new_model_str = new_model_str.replace('x[:,{}]'.format(i),f)
        new_model_str = new_model_str.replace('x[{}]'.format(i),f)
    # operators
    new_model_str = new_model_str.replace('^','**')
    #GP-GOMEA
    new_model_str = new_model_str.replace('p/','/') 
    new_model_str = new_model_str.replace('plog','log') 
    new_model_str = new_model_str.replace('aq','/') 
    # ITEA
    new_model_str = new_model_str.replace('sqrtAbs','sqrt')
    # new_model_str = re.sub(pattern=r'sqrtAbs\((.*?)\)',
    #        repl=r'sqrt(abs(\1))',
    #        string=new_model_str
    #       )
    new_model_str = new_model_str.replace('np.','') 
    # ellyn & FEAT
    # pdb.set_trace()
    new_model_str = new_model_str.replace('|','')

    # gplearn
    for op in ('add', 'sub', 'mul', 'div'):
        new_model_str = new_model_str.replace(op,op.title()) 

#         new_model_str = new_model_str.replace('Sqrt','/') 

    # replace floating point digits with constants
#     constants = re.findall(r'[-+]?\d*\.*\d+',new_model_str)
#     print('constants:',constants)
#     for i,c in enumerate(constants):
#         new_model = new_model_str.replace(c,'C'+str(i))
        
    print('parsing',new_model_str)
    model_sym = parse_expr(new_model_str, local_dict = local_dict)
#     simp = model_sym
#     print('factor...')
#     simp = factor(model_sym)
    print('simplify...')
    simp = simplify(model_sym, ratio=1)
    print('simplified:',simp)
#     simp = simplify(simp)
    # return model_sym, simp
    return round_floats(simp)


def get_sym_model(dataset, return_str=True):
    """return sympy model from dataset metadata"""
    metadata = load(
            open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
            Loader=Loader
    )
    df = pd.read_csv(dataset,sep='\t')
    features = [c for c in df.columns if c != 'target']
#     print('features:',df.columns)
    description = metadata['description'].split('\n')
    model_str = [ms for ms in description if '=' in ms][0].split('=')[-1]
    model_str = model_str.replace('pi','3.1415926535')
    if return_str:
        return model_str

    # pdb.set_trace()
    # handle feynman problem constants
#     print('model:',model_str)
    model_sym = parse_expr(model_str, 
			   local_dict = {k:Symbol(k) for k in features})
#     print('sym model:',model_sym)
    return model_sym
