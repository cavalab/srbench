import torch
import os, sys
import symbolicregression

model_path = os.path.join("/".join(__file__.split("/")[:-1]), "model1.pt")
try:
    model = torch.load(model_path)
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
    model_str = est.retrieve_tree(tree_idx=0).infix()
    return model_str

def my_pre_train_fn(est, X, y):
    """In this example we adjust FEAT generations based on the size of X 
       versus relative to FEAT's batch size setting. 
    """
    return

# define eval_kwargs.
eval_kwargs = dict(
                   pre_train=my_pre_train_fn,
                   test_params = {
                                 }
                  )
