import os
import sys
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
            # Handle file path
            if not os.path.exists(cfg_file):
                raise FileNotFoundError(f"Config file not found: {cfg_file}")
            
            # Get the directory containing the config file
            cfg_dir = os.path.dirname(os.path.abspath(cfg_file))
            if cfg_dir not in sys.path:
                sys.path.insert(0, cfg_dir)
            
            # Get the module name without extension
            module_name = os.path.splitext(os.path.basename(cfg_file))[0]
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, cfg_file)
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
    
    def to_dict(self):
        """Convert Config object to dictionary for unpacking or other uses."""
        return {k: v for k, v in self.module.__dict__.items() if not k.startswith('_')}
    
    def merge_from_args(self, args):
        """
        Merge config from either:
        1. argparse.Namespace object (automatically matches attributes)
        2. List of strings in format ["key", "value", "key2", "value2"]
        """
        if hasattr(args, '__dict__'):
            # Handle argparse.Namespace object
            self._merge_from_namespace(args)
        else:
            # Handle list of key-value pairs
            self._merge_from_list(args)
    
    def _merge_from_namespace(self, args):
        """Merge from argparse.Namespace object"""
        args_dict = vars(args)
        
        for key, value in args_dict.items():
            # Skip None values and special argparse attributes
            if value is None or key.startswith('_'):
                continue
                
            # Check if this attribute exists in config
            if self._has_nested_attr(key):
                # Update existing attribute
                self._set_nested_attr(key, value)
                # print(f"Updated config.{key} = {value}")
            else:
                # Add new attribute if it doesn't exist
                setattr(self.module, key, value)
                # print(f"Added config.{key} = {value}")
    
    def _merge_from_list(self, args_list):
        """Merge from list of key-value pairs"""
        assert len(args_list) % 2 == 0, "Arguments must be key-value pairs"
        for key, value in zip(args_list[0::2], args_list[1::2]):
            # Convert string value to appropriate type
            value = self._convert_value(value)
            self._set_nested_attr(key, value)
    
    def _has_nested_attr(self, attr_name):
        """Check if nested attribute exists in config"""
        try:
            current = self.module
            for part in attr_name.split('.'):
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return False
            return True
        except:
            return False
    
    def _set_nested_attr(self, attr_name, value):
        """Set nested attribute in config"""
        parts = attr_name.split('.')
        current = self.module
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if not hasattr(current, part):
                setattr(current, part, type('ConfigModule', (), {}))
            current = getattr(current, part)
        
        # Set the final attribute
        setattr(current, parts[-1], value)
    
    def _convert_value(self, value):
        """Convert string value to appropriate type"""
        if isinstance(value, str):
            try:
                # Try to interpret as a number or boolean
                if value.lower() == 'true':
                    return True
                elif value.lower() == 'false':
                    return False
                elif '.' in value:
                    return float(value)
                else:
                    return int(value)
            except (ValueError, AttributeError):
                # Keep as string if conversion fails
                return value
        return value 