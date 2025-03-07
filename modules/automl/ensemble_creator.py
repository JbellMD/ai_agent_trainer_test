from sklearn.feature_selection import SelectKBest, f_classif
from ..utils.logging import AutoTrainerLogger

class FeatureSelector:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def select_features(self, X, y, k: int = 10):
        """Select top k features using statistical tests"""
        self.logger.log(f"Selecting top {k} features")
        selector = SelectKBest(f_classif, k=k)
        return selector.fit_transform(X, y)
        
    def recursive_feature_elimination(self, X, y, n_features: int = 10):
        """Perform recursive feature elimination"""
        self.logger.log(f"Performing RFE to select {n_features} features")
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        selector = RFE(
            estimator=RandomForestClassifier(),
            n_features_to_select=n_features
        )
        return selector.fit_transform(X, y)