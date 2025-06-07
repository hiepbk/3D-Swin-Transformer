import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import open3d as o3d
from collections import defaultdict


class ModelNetDataset(Dataset):
    def __init__(self, root_dir='./data', 
                 split="train", 
                 num_classes=10, 
                 num_feat=6,
                 grid_size=80,
                 pc_range=[-1.0,-1.0,-1.0,1.0,1.0,1.0]):
        self.root_dir = root_dir

        self.num_classes = num_classes
        self.num_feat = num_feat
        self.split = split
        self.grid_size = grid_size
        self.voxel_size = (max(pc_range) - min(pc_range)) / grid_size
        self.pc_range = np.array(pc_range)

        # Load class names
        self.classes = self.read_txt(os.path.join(root_dir, f'modelnet{num_classes}_shape_names.txt'))
        if len(self.classes) != num_classes:
            raise ValueError(f"Number of classes in file ({len(self.classes)}) doesn't match num_classes ({num_classes})")
        
        # Load split data
        self.train_paths, self.train_labels = self.get_split_path('train')
        self.test_paths, self.test_labels = self.get_split_path('test')
        
        # Validate labels
        self._validate_labels()
        
    def _validate_labels(self):
        """Validate that all labels are within the correct range"""
        if self.split == "train":
            labels = self.train_labels
        else:
            labels = self.test_labels
            
        invalid_labels = [l for l in labels if l < 0 or l >= self.num_classes]
        if invalid_labels:
            raise ValueError(f"Found invalid labels: {invalid_labels}. Labels must be in range [0, {self.num_classes-1}]")
            
        # Log class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution in {self.split} split:")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label} ({self.classes[label]}): {count} samples")
        print()

    def read_txt(self, file_path):
        with open(file_path, 'r') as f:
            return f.read().splitlines()
        
    def get_split_path(self, split):
        split_path = []
        split_label = []
        split_lst = self.read_txt(os.path.join(self.root_dir, f'modelnet{self.num_classes}_{split}.txt'))
        for split in split_lst:
            parts = split.split('_')
            class_name = '_'.join(parts[:-1])
            split_path.append(os.path.join(self.root_dir, class_name, f'{split}.txt'))
            label = self.classes.index(class_name)
            split_label.append(label)
        return split_path, split_label
        
    def __len__(self):
        if self.split == "train":
            return len(self.train_paths)
        else:
            return len(self.test_paths)
    
    def __getitem__(self, idx):
        if self.split == "train":
            pc_path, label = self.train_paths[idx], self.train_labels[idx]
        else:
            pc_path, label = self.test_paths[idx], self.test_labels[idx]
        
        # Load and process point cloud
        pc = np.loadtxt(pc_path, delimiter=',')[:, :self.num_feat]
        
        # Normalize point cloud coordinates
        pc[:, :3] = (pc[:, :3] - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
        
        # Normalize features if they exist
        if pc.shape[1] > 3:
            pc[:, 3:] = pc[:, 3:] / (np.linalg.norm(pc[:, 3:], axis=1, keepdims=True) + 1e-8)
        
        voxel_grid = self.mean_grid_voxelize(pc)
        
        # Convert to torch tensors
        torch_voxel_grid = torch.from_numpy(voxel_grid).float().permute(3, 0, 1, 2)
        torch_label = torch.tensor(label, dtype=torch.long)  # Use long dtype for classification labels
        
        return torch_voxel_grid, torch_label
    
    
    def mean_grid_voxelize(self, pc):
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
        voxel_grid = np.zeros((self.grid_size, self.grid_size, self.grid_size, pc.shape[1]))

        # Fill voxel grid for non-empty voxels
        for coords, points in voxel_dict.items():
            if all(0 <= c < self.grid_size for c in coords):  # Double check bounds
                voxel_grid[coords] = np.mean(points, axis=0)
    
        return voxel_grid  # (grid_size, grid_size, grid_size, num_feat)
    

    def analyze_dataset(self):
        min_x = 100000
        min_y = 100000
        min_z = 100000
        max_x = -100000
        max_y = -100000
        max_z = -100000
        for i, pc_path in enumerate(self.train_paths):
            pc = np.loadtxt(pc_path, delimiter=',')
            min_x = min(min_x, np.min(pc[:, 0]))
            min_y = min(min_y, np.min(pc[:, 1]))
            min_z = min(min_z, np.min(pc[:, 2]))
            max_x = max(max_x, np.max(pc[:, 0]))
            max_y = max(max_y, np.max(pc[:, 1]))
            max_z = max(max_z, np.max(pc[:, 2]))
            print(f"{i}/{len(self.train_paths)}: {min_x}, {min_y}, {min_z}, {max_x}, {max_y}, {max_z}", end="\r")
        self.pc_range = np.array([min_x, min_y, min_z, max_x, max_y, max_z])

    def visualize_point_cloud(self, index):

        pc_path = self.train_paths[index]
        pc = np.loadtxt(pc_path, delimiter=',')

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        
        # Set points (first 3 columns)
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
        
        # Set normals (last 3 columns)
        pcd.normals = o3d.utility.Vector3dVector(pc[:, 3:])
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

        
        # # Visualize
        # o3d.visualization.draw_geometries([pcd, coordinate_frame],
        #                                 window_name=f"Point Cloud: {os.path.basename(pc_path)}",
        #                                 point_show_normal=True)
        
        # Print some information
        print(f"Point cloud path: {pc_path}")
        print(f"Number of points: {len(pc)}")
        print(f"Point cloud shape: {pc.shape}")

    def visualize_voxel_grid(self, voxel_grid):
        x_max, y_max, z_max = voxel_grid.shape[:3]
        x_min, y_min, z_min = 0, 0, 0
        num_cuboids = np.prod(voxel_grid.shape)
        print(f"num_cuboids: {num_cuboids}")

        # Create list to store geometries
        geometries = []

        # Add non-empty voxels
        for i in range(x_max):
            for j in range(y_max):
                for k in range(z_max):
                    if np.all(voxel_grid[i, j, k]) != 0:
                        cuboid = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
                        cuboid.translate([i, j, k])
                        cuboid.compute_vertex_normals()
                        cuboid.paint_uniform_color([1, 0, 0])  # Red for non-empty voxels
                        geometries.append(cuboid)

        # Create bounding box for the voxel grid
        bbox = o3d.geometry.LineSet()
        bbox_points = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_min, y_max, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_min, y_max, z_max], [x_max, y_max, z_max]
        ]
        bbox_lines = [
            [0, 1], [0, 2], [1, 3], [2, 3],  # bottom face
            [4, 5], [4, 6], [5, 7], [6, 7],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # connecting lines
        ]
        bbox.points = o3d.utility.Vector3dVector(bbox_points)
        bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
        bbox.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in range(len(bbox_lines))])  # Black lines
        geometries.append(bbox)

        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(x_max, y_max, z_max) * 0.2,  # Scale the coordinate frame
            origin=[x_min, y_min, z_min]
        )
        geometries.append(coordinate_frame)

        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add all geometries to the visualizer
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        # Set background color to white for better visibility
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.point_size = 1.0
        
        # Enable transparency
        opt.mesh_show_back_face = True
        
        # Set rendering options
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()

        print(f"x_min: {x_min}, y_min: {y_min}, z_min: {z_min}, x_max: {x_max}, y_max: {y_max}, z_max: {z_max}")

if __name__ == "__main__":
    dataset = ModelNetDataset()
    # random index
    # for i in range(len(dataset)):
    #     #random index
    #     index = np.random.randint(0, len(dataset))
    #     dataset.visualize_point_cloud(index)

    for i in range(len(dataset)):
        index = np.random.randint(0, len(dataset))
        pc, label = dataset[index]
        print(pc.shape)
        print(label)
    print(dataset.classes)


    
        
