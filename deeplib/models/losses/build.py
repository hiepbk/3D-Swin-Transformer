from deeplib.utils.registry import Registry

LOSS_REGISTRY = Registry('loss')

def build_loss(cfg):
    return LOSS_REGISTRY.get(cfg.model.loss.name)(cfg)
