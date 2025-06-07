import numpy as np
from collections import defaultdict
from deeplib.utils.registry import Registry

TRANSFORM_REGISTRY = Registry('transform')

@TRANSFORM_REGISTRY.register_module()
class MeanGridVoxelize:
    def __init__(self, grid_size, pc_range, num_feat):
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = (max(pc_range) - min(pc_range)) / grid_size
        self.num_feat = num_feat

    def __call__(self, pc):
        xyz = pc[:, :3]
        
        # Calculate voxel coordinates
        voxel_coords = (xyz - self.pc_range[:3]) / self.voxel_size
        voxel_coords = np.floor(voxel_coords).astype(int)

        # Clip coordinates to ensure they stay within grid bounds
        voxel_coords = np.clip(voxel_coords, 0, self.grid_size - 1)
        voxel_coords = voxel_coords.reshape(-1, 3)

        # Group points by voxel
        voxel_dict = defaultdict(list)
        for i in range(len(pc)):
            key = tuple(voxel_coords[i])
            voxel_dict[key].append(pc[i])

        # Initialize voxel grid
        voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.num_feat))

        # Fill voxel grid for non-empty voxels
        for coords, points in voxel_dict.items():
            if all(0 <= c < self.grid_size for c in coords):  # Double check bounds
                voxel_grid[coords] = np.mean(points, axis=0)

        return voxel_grid  # (grid_size, grid_size, grid_size, num_feat)

def build_transform(transform_cfg):
    """Build transform from config."""
    transform_type = transform_cfg.pop('type')
    transform = TRANSFORM_REGISTRY.get(transform_type)(**transform_cfg)
    return transform