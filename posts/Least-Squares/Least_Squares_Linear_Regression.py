import numpy as np


class linearSquares():
    def __init__(self):
        self.w = None
        self.score_history = []

    def P_compute(self, X):
        return X.T@X

    def q_compute(self, X, y):
        return X.T@y

    def analytic(self, X, y):
        return np.linalg.inv(X.T@X)@X.T@y

    def fitAnalytic(self, X, y):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = self.analytic(X_, y)

    def fitGradient(self, X, y, alpha=0.01, max_iters=1000):
        self.score_history = []
        self.w = np.random.rand(X.shape[1]+1)
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        P = self.P_compute(X_)
        q = self.q_compute(X_, y)
        for i in range(max_iters):
            self.w = self.w - alpha * 2*(P@self.w-q)
            self.score_history.append(self.score(X, y))

    def predict(self, X):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return X_.dot(self.w)

    def score(self, X, y):

        return 1 - np.sum((y - self.predict(X))**2) / np.sum((y - np.mean(y))**2)
