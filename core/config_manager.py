import yaml
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
    def save_config(self, config_path: str):
        """Save current configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self.config.update(updates)
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()