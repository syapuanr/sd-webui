import yaml
import os
from pathlib import Path
from omegaconf import OmegaConf

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return OmegaConf.create(config)
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def get(self, key, default=None):
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update(self, key, value):
        """Update configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration back to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(OmegaConf.to_container(self.config), f, default_flow_style=False)

    def create_directories(self):
        """Create necessary directories for models"""
        base_path = Path(self.get('models.base_path', './models'))
        
        directories = [
            base_path,
            base_path / 'checkpoints',
            base_path / 'loras',
            base_path / 'vae',
            base_path / 'controlnet',
            base_path / 'embeddings'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Created model directories in {base_path}")