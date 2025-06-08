from deeplib.utils.registry import LOSS_REGISTRY
from deeplib.config.config import Config

def build_loss(cfg):
    """Build loss function from config."""
    loss_cfg = cfg.model.loss
    loss_dict = {}
    if isinstance(loss_cfg, list):
        for l_cfg in loss_cfg:
            # Convert dict to Config object for dot notation access
            l_cfg = Config(cfg_dict=l_cfg)
            loss_dict[l_cfg.name] = LOSS_REGISTRY.get(l_cfg.name)(l_cfg)
    elif isinstance(loss_cfg, dict):
        # Convert dict to Config object for dot notation access
        loss_cfg = Config(cfg_dict=loss_cfg)
        loss_dict[loss_cfg.name] = LOSS_REGISTRY.get(loss_cfg.name)(loss_cfg)
    else:
        raise TypeError(f'Loss config must be a list or dict, but got {type(loss_cfg)}')

    return LOSS_REGISTRY.get('ClassifyLoss')(loss_dict)
        
