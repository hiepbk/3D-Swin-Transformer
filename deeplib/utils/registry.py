class Registry:
    """
    A registry to map strings to classes or functions.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={list(self._module_dict.keys())})'
        return format_str

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_module(self, cls=None, module_name=None):
        if cls is None:
            return lambda x: self.register_module(x, module_name)
        
        if module_name is None:
            module_name = cls.__name__
        
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        
        self._module_dict[module_name] = cls
        return cls

# Create registries
HOOK_REGISTRY = Registry('hook')
MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
TRANSFORM_REGISTRY = Registry('transform') 