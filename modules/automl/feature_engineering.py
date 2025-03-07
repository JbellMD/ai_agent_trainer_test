import numpy as np
import pandas as pd
from ..utils.logging import AutoTrainerLogger

class FeatureEngineer:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def create_interaction_features(self, data: pd.DataFrame):
        """Create interaction features between columns"""
        self.logger.log("Creating interaction features")
        for i, col1 in enumerate(data.columns):
            for col2 in data.columns[i+1:]:
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        return data
        
    def create_polynomial_features(self, data: pd.DataFrame, degree: int = 2):
        """Create polynomial features"""
        self.logger.log(f"Creating polynomial features with degree {degree}")
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return pd.DataFrame(poly.fit_transform(data), columns=poly.get_feature_names_out(data.columns))