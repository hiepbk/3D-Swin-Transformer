import os
import importlib.util
from collections import OrderedDict

class Config:
    """
    A configuration class that loads from Python files and handles parameter overrides.
    """
    def __init__(self, cfg_file=None, cfg_dict=None, merge_from_args=None):
        assert cfg_file is not None or cfg_dict is not None, \
            "Either cfg_file or cfg_dict must be specified."
        
        if cfg_file is not None:
            # Load the Python file as a module
            spec = importlib.util.spec_from_file_location("config", cfg_file)
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        else:
            self.module = type('ConfigModule', (), cfg_dict)
        
        if merge_from_args is not None:
            self.merge_from_args(merge_from_args)
    
    def __getattr__(self, name):
        if hasattr(self.module, name):
            value = getattr(self.module, name)
            if isinstance(value, dict):
                # Convert nested dictionaries to Config objects
                return Config(cfg_dict=value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, name):
        return getattr(self.module, name)
    
    def get(self, name, default=None):
        """Get a value from the config."""
        return getattr(self.module, name, default)
    
    def __repr__(self):
        return str({k: v for k, v in self.module.__dict__.items() if not k.startswith('_')})
    
    def merge_from_args(self, args_list):
        """
        Merge config from a list of strings in format "key=value".
        """
        assert len(args_list) % 2 == 0, "Arguments must be key-value pairs"
        for key, value in zip(args_list[0::2], args_list[1::2]):
            key_list = key.split('.')
            current = self.module
            for subkey in key_list[:-1]:
                if not hasattr(current, subkey):
                    setattr(current, subkey, {})
                current = getattr(current, subkey)
            subkey = key_list[-1]
            try:
                # Try to interpret as a number or boolean
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except (ValueError, AttributeError):
                # Keep as string if conversion fails
                pass
            setattr(current, subkey, value) 