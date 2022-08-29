from HROCH import Hroch
import os

est = Hroch()


def model(est, X=None):
    if X is None or not hasattr(X, 'columns'):
        return str(est.sexpr)

    model_str = str(est.sexpr)
    print('model_str:',model_str)
    mapping = {'x_'+str(i): k for i, k in enumerate(X.columns)}
    new_model = model_str
    for k, v in reversed(mapping.items()):
        new_model = new_model.replace(k, v)

    return new_model


def my_pre_train_fn(est, X, y):
    try:
        est.numThreads = os.environ['OMP_NUM_THREADS']
    except Exception as e:
        print(e)

    max_time = 3600 - 60  # 60 second of slack
    if len(X) > 1000:
        max_time = 36000 - 60  # 60 second of slack
    est.timeLimit = max_time * 1000

    print(
        f'my_pre_train_fn: timeLimit={est.timeLimit} numThreads={est.numThreads}')


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn,
    scale_x=False,  # avoid poisonous scaling
    scale_y=False,
    test_params=dict(timeLimit=1000)
)
