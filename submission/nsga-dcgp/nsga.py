from sklearn.base import RegressorMixin, BaseEstimator


class NSGA(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            indiv_class, indiv_param,
            pop_size=100, n_gen=10000
    ):
        self.indiv_class = indiv_class
        self.indiv_param = indiv_param
        self.pop_size = pop_size
        self.n_gen = n_gen

    def fit(self):
        pass

    def predict(self, X):
        pass