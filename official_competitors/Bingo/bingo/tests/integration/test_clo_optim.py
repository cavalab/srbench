# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.evaluation.fitness_function \
    import FitnessFunction, VectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin, VectorGradientMixin
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization, MINIMIZE_SET, ROOT_SET
from bingo.chromosomes.multiple_floats import MultipleFloatChromosome

NUM_VALS = 10
NUM_OPT = 3


class MultipleFloatValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        return np.linalg.norm(individual.values)


class MultipleFloatValueFitnessFunctionWithGradient(GradientMixin, MultipleFloatValueFitnessFunction):
    def get_fitness_and_gradient(self, individual):
        full_gradient = individual.values / np.linalg.norm(individual.values)
        return self.__call__(individual), [full_gradient[i] for i in individual._needs_opt_list]


class FloatVectorFitnessFunction(VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        vals = individual.values
        return [x - 0 for x in vals]


class FloatVectorFitnessFunctionWithJacobian(VectorGradientMixin, FloatVectorFitnessFunction):
    def get_fitness_vector_and_jacobian(self, individual):
        jacobian = np.zeros((len(individual.values), len(individual._needs_opt_list)))
        for i, optimize_i in enumerate(individual._needs_opt_list):
            jacobian[optimize_i][i] = 1
        return self.evaluate_fitness_vector(individual), jacobian


@pytest.fixture
def opt_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals, [1, 3, 4])


@pytest.fixture
def reg_individual():
    vals = [1. for _ in range(NUM_VALS)]
    return MultipleFloatChromosome(vals)


@pytest.mark.parametrize("algorithm", MINIMIZE_SET)
def test_optimize_params_without_gradient(opt_individual, reg_individual, algorithm):
    if algorithm == "TNC":
        pytest.skip("This algorithm has convergence issues")
        
    np.random.seed(0)
    fitness_function = MultipleFloatValueFitnessFunction()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.sqrt(NUM_VALS - NUM_OPT),
                                             rel=5.e-6)
    assert reg_indv_fitness == pytest.approx(np.sqrt(NUM_VALS))


@pytest.mark.parametrize("algorithm", MINIMIZE_SET)
def test_optimize_params_with_gradient(opt_individual, reg_individual, algorithm):
    np.random.seed(0)
    fitness_function = MultipleFloatValueFitnessFunctionWithGradient()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.sqrt(NUM_VALS - NUM_OPT),
                                             rel=5.e-6)
    assert reg_indv_fitness == pytest.approx(np.sqrt(NUM_VALS))


@pytest.mark.parametrize("algorithm", ROOT_SET)
def test_optimize_fitness_vector_without_jacobian(opt_individual, reg_individual, algorithm):
    np.random.seed(0)
    reg_list = [1. for _ in range(NUM_VALS)]
    opt_list = [1. for _ in range(NUM_VALS)]
    opt_list[:3] = [0., 0., 0.]
    fitness_function = FloatVectorFitnessFunction()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.mean(opt_list))
    assert reg_indv_fitness == pytest.approx(np.mean(reg_list))


@pytest.mark.parametrize("algorithm", ROOT_SET)
def test_optimize_fitness_vector_with_jacobian(opt_individual, reg_individual, algorithm):
    np.random.seed(0)
    reg_list = [1. for _ in range(NUM_VALS)]
    opt_list = [1. for _ in range(NUM_VALS)]
    opt_list[:3] = [0., 0., 0.]
    fitness_function = FloatVectorFitnessFunctionWithJacobian()
    local_opt_fitness_function = ContinuousLocalOptimization(
        fitness_function, algorithm)
    opt_indv_fitness = local_opt_fitness_function(opt_individual)
    reg_indv_fitness = local_opt_fitness_function(reg_individual)
    assert opt_indv_fitness == pytest.approx(np.mean(opt_list))
    assert reg_indv_fitness == pytest.approx(np.mean(reg_list))

