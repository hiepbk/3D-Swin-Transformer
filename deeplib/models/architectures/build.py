from deeplib.utils.registry import ARCHITECTURE_REGISTRY


def build_architecture(cfg):
    """Build architecture from config."""
    architecture_type = cfg.model.name
    architecture = ARCHITECTURE_REGISTRY.get(architecture_type)(cfg)
    return architecture


