import pytest
import pandas as pd
from ai_agent_trainer.modules.data_processing import DataLoader, DataPreprocessor

class TestDataProcessing:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, None],
            'feature2': [4, 5, 5, 6],
            'target': [0, 1, 0, 1]
        })

    def test_data_loader(self, tmp_path):
        # Test CSV loading
        file_path = tmp_path / "test.csv"
        test_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        test_df.to_csv(file_path, index=False)
        
        loader = DataLoader()
        loaded_df = loader.load_csv(file_path)
        assert loaded_df.equals(test_df)

    def test_data_cleaning(self, sample_data):
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(sample_data)
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) == 3  # Removed duplicate

    def test_data_normalization(self, sample_data):
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.clean_data(sample_data)
        normalized_data = preprocessor.normalize_data(cleaned_data)
        for col in normalized_data.columns:
            assert abs(normalized_data[col].mean()) < 1e-6
            assert abs(normalized_data[col].std() - 1) < 1e-6