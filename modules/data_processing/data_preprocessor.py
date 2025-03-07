import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..utils.logging import AutoTrainerLogger

class DataPreprocessor:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        self.scaler = StandardScaler()
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        self.logger.log("Cleaning data")
        # Handle missing values
        data = data.fillna(data.mean())
        # Remove duplicates
        data = data.drop_duplicates()
        return data
        
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data"""
        self.logger.log("Normalizing data")
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
        
    def split_data(self, data: pd.DataFrame, test_size: float = 0.2) -> tuple:
        """Split data into training and test sets"""
        self.logger.log(f"Splitting data with test size: {test_size}")
        from sklearn.model_selection import train_test_split
        return train_test_split(data, test_size=test_size)