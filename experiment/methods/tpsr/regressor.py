import time
import json
import os
import torch
import numpy as np
import sympy as sp
from sklearn import feature_selection 
from sklearn.base import BaseEstimator, RegressorMixin

# handling hardcored paths and no package setup 
import sys; sys.path.append("/opt/conda/bin/tpsr")

# TPSR stuff
import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, refine_for_sample, refine_for_sample_test
from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json
from rl_env import RLEnv
from default_pi import E2EHeuristic

def get_top_k_features(X, y, k=10):
    if y.ndim==2:
        y=y[:,0]
    if X.shape[1]<=k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])

class TPRS_params():
    def __init__(self):
        self.ablation_to_keep=None
        self.accumulate_gradients=1
        self.actor_lr=1e-06
        self.alg='mcts'
        self.amp=-1
        self.attention_dropout=0
        self.backbone_model='e2e'
        self.batch_load=False
        self.batch_size=1
        self.batch_size_eval=64
        self.beam_early_stopping=True
        self.beam_eval=True
        self.beam_eval_train=0
        self.beam_length_penalty=1
        self.beam_selection_metrics=1
        self.beam_size=1
        self.beam_temperature=0.1
        self.beam_type='sampling'
        self.clip_grad_norm=0.5
        self.collate_queue_size=2000
        self.cpu=True
        self.critic_lr=1e-05
        self.debug=False
        self.debug_slurm=False
        self.debug_train_statistics=False
        self.dec_emb_dim=512
        self.dec_positional_embeddings='learnable'
        self.device='gpu'
        self.dropout=0
        self.dump_path=''
        self.emb_emb_dim=64
        self.emb_expansion_factor=1
        self.embedder_type='LinearPoint'
        self.enc_emb_dim=512
        self.enc_positional_embeddings=None
        self.enforce_dim=True
        self.entropy_coef=0.01
        self.entropy_weighted_strategy='none'
        self.env_base_seed=0
        self.env_name='functions'
        self.eval_data=''
        self.eval_dump_path=None
        self.eval_from_exp=''
        self.eval_in_domain=False
        self.eval_input_length_modulo=-1
        self.eval_mcts_in_domain=False
        self.eval_mcts_on_pmlb=False
        self.eval_noise=0
        self.eval_noise_gamma=0.0
        self.eval_noise_type='additive'
        self.eval_on_pmlb=False
        self.eval_only=True
        self.eval_size=10000
        self.eval_verbose=0
        self.eval_verbose_print=False
        self.exp_id=''
        self.exp_name='debug'
        self.export_data=False
        self.extra_binary_operators=''
        self.extra_constants=None
        self.extra_unary_operators=''
        self.float_precision=3
        self.fp16=False
        self.gpu_to_use='0'
        self.horizon=200
        self.kl_coef=0.01
        self.kl_regularizer=0.001
        self.lam=0.1
        self.local_rank=-1
        self.lr=1e-05
        self.lr_patience=100
        self.mantissa_len=1
        self.master_port=-1
        self.max_binary_ops_offset=4
        self.max_binary_ops_per_dim=1
        self.max_centroids=10
        self.max_epoch=100000
        self.max_exponent=100
        self.max_exponent_prefactor=1
        self.max_generated_output_len=200
        self.max_input_dimension=10
        self.max_input_points=200
        self.max_int=10
        self.max_len=200
        self.max_number_bags=10
        self.max_output_dimension=1
        self.max_src_len=200
        self.max_target_len=200
        self.max_token_len=0
        self.max_trials=1
        self.max_unary_depth=6
        self.max_unary_ops=4
        self.min_binary_ops_per_dim=0
        self.min_input_dimension=1
        self.min_len_per_dim=5
        self.min_op_prob=0.01
        self.min_output_dimension=1
        self.min_unary_ops=0
        self.n_dec_heads=16
        self.n_dec_hidden_layers=1
        self.n_dec_layers=16
        self.n_emb_layers=1
        self.n_enc_heads=16
        self.n_enc_hidden_layers=1
        self.n_enc_layers=2
        self.n_prediction_points=200
        self.n_steps_per_epoch=3000
        self.n_trees_to_refine=10
        self.n_words=10292
        self.no_prefix_cache=True # setting these will disable cache
        self.no_seq_cache=True
        self.norm_attention=False
        self.num_beams=1
        self.num_workers=10
        self.nvidia_apex=False
        self.operators_to_downsample='div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3'
        self.operators_to_not_repeat=''
        self.optimizer='adam_inverse_sqrt,warmup_updates=10000'
        self.pad_to_max_dim=True
        self.pmlb_data_type='feynman'
        self.prediction_sigmas='1,2,4,8,16'
        self.print_freq=100
        self.prob_const=0.0
        self.prob_rand=0.0
        self.queue_strategy=None
        self.reduce_num_constants=True
        self.refinements_types='method=BFGS_batchsize=256_metric=/_mse'
        self.reload_checkpoint=''
        self.reload_data=''
        self.reload_model=''
        self.reload_size=-1
        self.required_operators=''
        self.rescale=True
        self.reward_coef=1
        self.reward_type='nmse'
        self.rl_alg='ppo'
        self.rollout=3
        self.run_id=1
        self.sample_only=False
        self.save_eval_dic='./eval_result'
        self.save_model=False
        self.save_periodic=25
        self.save_results=False
        self.seed=23
        self.share_inout_emb=True
        self.simplify=False
        self.stopping_criterion=''
        self.target_kl=1
        self.target_noise=0.0
        self.tasks='functions'
        self.test_env_seed=1
        self.tokens_per_batch=10000
        self.train_noise_gamma=0.0
        self.train_value=False
        self.ts_mode='best'
        self.ucb_base=10.0
        self.ucb_constant=1.0
        self.uct_alg='uct'
        self.update_modules='all'
        self.use_abs=False
        self.use_controller=True
        self.use_skeleton=False
        self.use_sympy=False
        self.validation_metrics='r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity'
        self.vf_coef=0.0001
        self.warmup_epoch=5
        self.width=3
        self.windows=False
        
class TPSRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        
        self.tpsr_params = TPRS_params()

        self.random_state = random_state
        self.tpsr_params.seed = self.random_state
        
    def fit(self, X, y):
        
        np.random.seed(self.tpsr_params.seed)
        torch.manual_seed(self.tpsr_params.seed)
        torch.cuda.manual_seed(self.tpsr_params.seed)

        # manually setting the hyperparameters
        self.tpsr_params.cpu    = False if torch.cuda.is_available() else True
        self.tpsr_params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        symbolicregression.utils.CUDA = self.tpsr_params.device
        if (self.tpsr_params.no_prefix_cache==False) and (self.tpsr_params.no_seq_cache == False):
            self.tpsr_params.cache = 'b' #both cachings are arctivated
        elif (self.tpsr_params.no_prefix_cache==True) and (self.tpsr_params.no_seq_cache ==False):
            self.tpsr_params.cache = 's' #only sequence caching
        elif (self.tpsr_params.no_prefix_cache==False) and (self.tpsr_params.no_seq_cache==True):
            self.tpsr_params.cache = 'k' #only top-k caching
        else:
            self.tpsr_params.cache = 'n' # no caching

        self.tpsr_params.backbone_model="e2e"
        # self.tpsr_params.reload_model = "/pretrained/model1.pt"
        # if not os.path.isfile(self.tpsr_params.reload_model):
        #     raise FileNotFoundError(self.tpsr_params.reload_model)

        self.equation_env = build_env(self.tpsr_params)
        modules = build_modules(self.equation_env, self.tpsr_params)
        # print(self.tpsr_params)

        # pre-processing the data to handle up to 10 features
        
        top_k_features = get_top_k_features(X, y, k=self.tpsr_params.max_input_dimension)
        X = X[:, top_k_features]

        samples = {'x_to_fit': [X], 'y_to_fit':[y],
                   'x_to_pred':0, 'y_to_pred':0}

        # --- main ---
        # Transformer loads the pre trained
        model = Transformer(params = self.tpsr_params, env=self.equation_env, samples=samples)
        model.to(self.tpsr_params.device) 

        rl_env = RLEnv(
            params = self.tpsr_params,
            samples = samples,
            equation_env = self.equation_env,
            model = model)

        dp = E2EHeuristic(
                equation_env=self.equation_env,
                rl_env=rl_env,
                model=model,
                k=self.tpsr_params.width,
                num_beams=self.tpsr_params.num_beams,
                horizon=self.tpsr_params.horizon,
                device=self.tpsr_params.device,
                use_seq_cache=not self.tpsr_params.no_seq_cache,
                use_prefix_cache=not self.tpsr_params.no_prefix_cache,
                length_penalty = self.tpsr_params.beam_length_penalty,
                train_value_mode=self.tpsr_params.train_value,
                debug=self.tpsr_params.debug)

        start = time.time()

        agent = UCT(
            action_space=[], # this will not be used as we have a default policy
            gamma=1., # no discounting
            ucb_constant=1.,
            horizon=self.tpsr_params.horizon,
            rollouts=self.tpsr_params.rollout,
            dp=dp,
            width=self.tpsr_params.width,
            reuse_tree=True,
            alg=self.tpsr_params.uct_alg,
            ucb_base=self.tpsr_params.ucb_base)

        agent.display()
        if self.tpsr_params.sample_only:
            # a bit hacky, should set a large rollout number so all
            # programs are saved in samples json file
            horizon = 1
        else:
            horizon = 200    
            
        done = False

        s = rl_env.state

        ret_all = []
        for t in range(horizon):
            if len(s) >= self.tpsr_params.horizon:
                print(f'Cannot process programs longer than {self.tpsr_params.horizon}. Stop here.')
                break

            if done:
                break

            act = agent.act(rl_env, done)
            s, r, done, _ = rl_env.step(act)

            if t ==0:
                real_root = agent.root

            update_root(agent, act, s)
            dp.update_cache(s)
        
        time_elapsed = time.time() - start
        print('TPSR+E2E Time Elapsed:', time_elapsed)

        # TPSR+E2E NMSE
        # y_mcts , mcts_str , mcts_tree = pred_for_sample_no_refine(
        #     model, self.equation_env, s ,samples['x_to_fit'])
    
        # TPSR+E2E NMSE after Refine
        y_mcts , mcts_str , mcts_tree = refine_for_sample(
            self.tpsr_params, model, self.equation_env, s, 
            samples['x_to_fit'], samples['y_to_fit'])

        self.mcts_str_ = mcts_str
        self.tree_     = mcts_tree
        self.model_    = model
        self.X_        = X
        self.y_        = y
        self.s_        = s
        
        return self


    def predict(self, X):
        # y_mcts , _ , _ = pred_for_sample_no_refine(
        #     self.model_, self.equation_env, self.s_ ,[X])
    
        _, y_mcts , _ , _ = refine_for_sample_test(
            self.model_, self.equation_env, self.s_, 
            [self.X_], [self.y_], [X])
        
        return y_mcts


est = TPSRRegressor()


def model(est, X=None):
    mcts_str = est.mcts_str_
    
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    for op, replace_op in replace_ops.items():
        mcts_str = mcts_str.replace(op,replace_op)

    mcts_eq = sp.parse_expr(mcts_str)

    return str(mcts_eq)


#  python regressor.py

# define eval_kwargs.
eval_kwargs = dict(use_dataframe=False)
