"""Training-fold-only supervised feature selection with deterministic ties."""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class FoldAnovaSelector(TransformerMixin, BaseEstimator):
    """Deterministic top-k ANOVA selection, excluding training-constant columns."""
    def __init__(self, k=20):
        self.k = k

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() or self.k < 1:
            raise ValueError('Invalid feature selection inputs')
        self.n_features_in_ = x.shape[1]
        usable = np.flatnonzero(np.ptp(x, axis=0) > 0)
        if not len(usable):
            raise ValueError('No varying training features')
        with warnings.catch_warnings():
            # Perfect within-class separation can legitimately yield +infinity.
            warnings.simplefilter('ignore', RuntimeWarning)
            scores, _ = f_classif(x[:, usable], y)
        scores = np.where(np.isnan(scores), -np.inf, scores)
        ranking = np.lexsort((usable, -scores))
        self.selected_ = np.sort(usable[ranking[:min(self.k, len(usable))]])
        return self

    def transform(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Feature schema changed after fitting')
        return x[:, self.selected_]

