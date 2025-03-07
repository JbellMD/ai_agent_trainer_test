import yaml
from pathlib import Path
from typing import Any, Dict
from ..utils.logging import AutoTrainerLogger

class ConfigLoader:
    def __init__(self):
        self.logger = AutoTrainerLogger()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        self.logger.log(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        self.logger.log(f"Saving configuration to {config_path}")
        Path(config_path).parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)