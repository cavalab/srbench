import importlib
import sys
from os.path import abspath
from os.path import dirname as d

sys.path.append(d(d(abspath(__file__))))


from evaluate_model import evaluate_model


def test_evaluate_model(ml):
    print("running test_evaluate_model with ml=", ml)
    dataset = "test/192_vineyard_small.tsv.gz"
    results_path = "tmp_results"
    random_state = 42

    algorithm = importlib.__import__(
        f"methods.{ml}.regressor",
        globals(),
        locals(),
        ["est", "hyper_params", "complexity"],
    )

    print("algorithm imported:", algorithm)
    eval_kwargs = {}
    if "eval_kwargs" in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    evaluate_model(
        dataset=dataset,
        results_path=results_path,
        random_state=random_state,
        est_name=ml,
        est=algorithm.est,
        model=algorithm.model,
        algorithm=algorithm,
        test=True,  # testing
        **eval_kwargs,
    )
