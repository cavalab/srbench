"""This module defines a benchmark test used to automatically test an
evolutionary algorithm in a `BenchmarkSuite`
"""


class BenchmarkTest:
    """ A class for easy training and scoring of an evolutionary algorithm

    Parameters
    ----------
    train_function : callable
        The function used to train (run) the evolutionary algorithm based on
        some input data. The function should have the following signature:
        equ, aux = train_function(training_data) where equ is an `Equation`,
        aux is auxiliary training info (any type), and training_data is
        `TrainingData`.
    score_function : callable
        The function used to score an evolutionary algorithm based on some
        input data. The function should have the following signature:
        scores = score_function(equ, data, aux) where scores is a tuple of
        scores, equ is an `Equation`, data is `TrainingData` and aux is
        auxiliary training info from the train_function.
    """
    def __init__(self, train_function, score_function):
        self._train_function = train_function
        self._score_function = score_function
        self._best_equation = None
        self._aux_train_info = None

    def train(self, training_data):
        """Run the evolutionary optimization and save the best individual

        Parameters
        ----------
        training_data : TrainingData
            Data used for fitness in the evolutionary optimization
        """
        self._best_equation, self._aux_train_info = \
            self._train_function(training_data)

    def score(self, score_data):
        """Calculate scoring metrics

        Identify how well the evolutionary optimization performed in
        predicting results on given scoring data

        Parameters
        ----------
        score_data : `TrainingData`
            data used in scoring the evolutionary optimization

        Returns
        -------
        scores : tuple
            scores of the best individual in the evolutionary optimization

        """
        if self._best_equation is None:
            raise RuntimeError("BenchmarkTest must be trained before scoring")
        return self._score_function(self._best_equation, score_data,
                                    self._aux_train_info)