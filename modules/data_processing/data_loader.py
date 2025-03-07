import pandas as pd
from typing import Any, Dict
from ..utils.logging import AutoTrainerLogger

class DataLoader:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        self.logger.log(f"Loading data from CSV: {file_path}")
        return pd.read_csv(file_path)
        
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        self.logger.log(f"Loading data from JSON: {file_path}")
        import json
        with open(file_path) as f:
            return json.load(f)
            
    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from database"""
        self.logger.log(f"Loading data from database with query: {query}")
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)