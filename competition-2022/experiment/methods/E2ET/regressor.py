import torch
import os, sys
import symbolicregression
import requests

model_path = "methods/E2ET/model.pt" 
try:
    if not os.path.isfile(model_path): 
        raise ValueError
    #     url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
    #     r = requests.get(url, allow_redirects=True)
    #     open(model_path, 'wb').write(r.content)
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    print(model.device)
    print("Model successfully loaded!")

except Exception as e:
    print("ERROR: model not loaded! path was: {}".format(model_path))
    print(e)    

est = symbolicregression.model.SymbolicTransformerRegressor(
                        model=model,
                        max_input_points=200,
                        n_trees_to_refine=100,
                        rescale=True
                        )

def model(est, X=None):
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    model_str = est.retrieve_tree(tree_idx=0, with_infos=True)["relabed_predicted_tree"].infix()
    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
    # wgl: variable names are wrong. 

    mapping = {'x_'+str(i): k for i, k in enumerate(X.columns)}
    new_model = model_str
    for k, v in reversed(list(mapping.items())):
        new_model = new_model.replace(k, v)
    return new_model

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    return

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {
                                 },
                   DataFrame=False
                  )
