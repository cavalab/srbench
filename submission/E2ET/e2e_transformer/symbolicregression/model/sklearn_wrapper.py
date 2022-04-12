import math, time, copy
import numpy as np
import torch
from collections import defaultdict
from symbolicregression.metrics import compute_metrics
from sklearn.base import BaseEstimator
import symbolicregression.model.utils_wrapper as utils_wrapper
import traceback

class SymbolicTransformerRegressor(BaseEstimator):

    def __init__(self,
                model=None,
                max_input_points=10000,
                max_number_bags=-1,
                stop_refinement_after=1,
                n_trees_to_refine=1,
                rescale=True
                ):

        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.model = model
        self.stop_refinement_after = stop_refinement_after
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale

    def set_args(self, args={}):
        for arg, val in args.items():
            assert hasattr(self, arg), "{} arg does not exist".format(arg)
            setattr(self, arg, val)

    def fit(
        self,
        X,
        Y,
        verbose=False
    ):
        self.start_fit = time.time()

        if not isinstance(X, list):
            X = [X]
            Y = [Y]
        n_datasets = len(X)

        scaler = utils_wrapper.StandardScaler() if self.rescale else None
        scale_params = {}
        if scaler is not None:
            scaled_X = []
            for i, x in enumerate(X):
                scaled_X.append(scaler.fit_transform(x))
                scale_params[i]=scaler.get_params()
        else:
            scaled_X = X

        inputs, inputs_ids = [], []
        for seq_id in range(len(scaled_X)):
            for seq_l in range(len(scaled_X[seq_id])):
                y_seq = Y[seq_id]
                if len(y_seq.shape)==1:
                    y_seq = np.expand_dims(y_seq,-1)
                if seq_l%self.max_input_points == 0:
                    inputs.append([])
                    inputs_ids.append(seq_id)
                inputs[-1].append([scaled_X[seq_id][seq_l], y_seq[seq_l]])

        if self.max_number_bags>0:
            inputs = inputs[:self.max_number_bags]
            inputs_ids = inputs_ids[:self.max_number_bags]

        forward_time=time.time()
        outputs = self.model(inputs)  ##Forward transformer: returns predicted functions
        if verbose: print("Finished forward in {} secs".format(time.time()-forward_time))

        candidates = defaultdict(list)
        assert len(inputs) == len(outputs), "Problem with inputs and outputs"
        for i in range(len(inputs)):
            input_id = inputs_ids[i]
            candidate = outputs[i]
            candidates[input_id].extend(candidate)
        assert len(candidates.keys())==n_datasets
            
        self.tree = {}
        for input_id, candidates_id in candidates.items():
            if len(candidates_id)==0: 
                self.tree[input_id]=None
                continue
        
            refined_candidates = self.refine(scaled_X[input_id], Y[input_id], candidates_id, verbose=verbose)
            for i,candidate in enumerate(refined_candidates):
                if scaler is not None:
                    refined_candidates[i]["predicted_tree"]=scaler.rescale_function(self.model.env, candidate["predicted_tree"], *scale_params[input_id])
                else: 
                    refined_candidates[i]["predicted_tree"]=candidate["predicted_tree"]
            self.tree[input_id] = refined_candidates

    @torch.no_grad()
    def evaluate_tree(self, tree, X, y, metric):
        numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree)
        y_tilde = numexpr_fn(X)[:,0]
        metrics = compute_metrics({"true": [y], "predicted": [y_tilde], "predicted_tree": [tree]}, metrics=metric)
        return metrics[metric][0]

    def order_candidates(self, X, y, candidates, metric="_mse", verbose=False):
        scores = []
        for candidate in candidates:
            if metric not in candidate:
                score = self.evaluate_tree(candidate["predicted_tree"], X, y, metric)
                if math.isnan(score): 
                    score = np.infty if metric.startswith("_") else -np.infty
            else:
                score = candidates[metric]
            scores.append(score)
        ordered_idx = np.argsort(scores)  
        if not metric.startswith("_"): ordered_idx=list(reversed(ordered_idx))
        candidates = [candidates[i] for i in ordered_idx]
        return candidates

    def refine(self, X, y, candidates, verbose):
        refined_candidates = []
        
        ## For skeleton model
        for i, candidate in enumerate(candidates):
            candidate_skeleton, candidate_constants =  self.model.env.generator.function_to_skeleton(candidate, constants_with_idx=True)
            if "CONSTANT" in candidate_constants:
                candidates[i] = self.model.env.wrap_equation_floats(candidate_skeleton, np.random.randn(len(candidate_constants)))

        candidates = [{"refinement_type": "NoRef", "predicted_tree": candidate, "time": time.time()-self.start_fit} for candidate in candidates]
        candidates = self.order_candidates(X, y, candidates, metric="_mse", verbose=verbose)

        ## REMOVE SKELETON DUPLICATAS
        skeleton_candidates, candidates_to_remove = {}, []
        for i, candidate in enumerate(candidates):
            skeleton_candidate, _ = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=False)
            if skeleton_candidate.infix() in skeleton_candidates:
                candidates_to_remove.append(i)
            else:
                skeleton_candidates[skeleton_candidate.infix()]=1
        if verbose: print("Removed {}/{} skeleton duplicata".format(len(candidates_to_remove), len(candidates)))

        candidates = [candidates[i] for i in range(len(candidates)) if i not in candidates_to_remove]
        if self.n_trees_to_refine>0:
            candidates_to_refine = candidates[:self.n_trees_to_refine]
        else:
            candidates_to_refine = copy.deepcopy(candidates)

        for candidate in candidates_to_refine:
            refinement_strategy = utils_wrapper.BFGSRefinement()
            candidate_skeleton, candidate_constants = self.model.env.generator.function_to_skeleton(candidate["predicted_tree"], constants_with_idx=True)
            try:
                refined_candidate = refinement_strategy.go(env=self.model.env, 
                                                        tree=candidate_skeleton, 
                                                        coeffs0=candidate_constants,
                                                        X=X,
                                                        y=y,
                                                        downsample=1024,
                                                        stop_after=self.stop_refinement_after)

            except Exception as e:
                if verbose: 
                    print(e)
                    #traceback.format_exc()
                continue
            
            if refined_candidate is not None:
                refined_candidates.append({ 
                        "refinement_type": "BFGS",
                        "predicted_tree": refined_candidate,
                        })            
        candidates.extend(refined_candidates)  
        candidates = self.order_candidates(X, y, candidates, metric="r2")

        for candidate in candidates:
            if "time" not in candidate:
                candidate["time"]=time.time()-self.start_fit
        return candidates

    def __str__(self):
        if hasattr(self, "tree"):
            for tree_idx in range(len(self.tree)):
                for gen in self.tree[tree_idx]:
                    print(gen)
        return "Transformer"

    def retrieve_refinements_types(self):
        return ["BFGS", "NoRef"]

    def retrieve_tree(self, refinement_type=None, tree_idx=0, with_infos=False):
        if tree_idx == -1: idxs = [_ for _ in range(len(self.tree))] 
        else: idxs = [tree_idx]
        best_trees = []
        for idx in idxs:
            best_tree = copy.deepcopy(self.tree[idx])
            if best_tree and refinement_type is not None:
                best_tree = list(filter(lambda gen: gen["refinement_type"]==refinement_type, best_tree))
            if not best_tree:
                if with_infos:
                    best_trees.append({"predicted_tree": None, "refinement_type": None, "time": None})
                else:
                    best_trees.append(None)
            else:
                if with_infos:
                    best_trees.append(best_tree[0])
                else:
                    best_trees.append(best_tree[0]["predicted_tree"])
        if tree_idx != -1: return best_trees[0]
        else: return best_trees


    def predict(self, X, refinement_type=None, tree_idx=0, batch=False):        
        if not isinstance(X, list):
            X = [X]
        res = []
        if batch:
            tree = self.retrieve_tree(refinement_type=refinement_type, tree_idx = -1)
            for tree_idx in range(len(tree)):
                X_idx = X[tree_idx]
                if tree[tree_idx] is None: 
                    res.append(None)
                else:   
                    numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree[tree_idx])
                    y = numexpr_fn(X_idx)[:,0]
                    res.append(y)
            return res
        else:
            X_idx = X[tree_idx]
            tree = self.retrieve_tree(refinement_type=refinement_type, tree_idx = tree_idx)
            if tree is not None:
                numexpr_fn = self.model.env.simplifier.tree_to_numexpr_fn(tree)
                y = numexpr_fn(X_idx)[:,0]
                return y
            else:
                return None