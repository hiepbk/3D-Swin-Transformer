import torch
from swin_transformer_3d import SwinTransformer
from dataset import ModelNetDataset

def test_swin_transformer():
    # # Create sequential test images
    # batch_size = 2
    # channels = 3
    # height = 224
    # width = 224
    
    # # Create first image with values starting from 1
    # img1 = torch.arange(1, height * width * channels + 1, dtype=torch.float32)
    # img1 = img1.view(channels, height, width)
    
    # # Create second image with values starting from 2
    # img2 = torch.arange(2, height * width * channels + 2, dtype=torch.float32)
    # img2 = img2.view(channels, height, width)
    
    # # Stack images into a batch
    # x = torch.stack([img1, img2]) # (2, 3, 224, 224) -> (B, C, H, W)


    # load a sample of voxel grid data
    dataset = ModelNetDataset()
    voxel_grid_0, label_0 = dataset[0]
    voxel_grid_1, label_1 = dataset[1]

    batch_voxel_grid = torch.stack([voxel_grid_0, voxel_grid_1])

    batch_size = batch_voxel_grid.shape[0]
    channels = batch_voxel_grid.shape[1]
    x_dim = dataset.grid_size
    y_dim = dataset.grid_size
    z_dim = dataset.grid_size
    grid_size = dataset.grid_size

    

    num_classes = len(dataset.classes)

    # Initialize the model
    model = SwinTransformer(
        grid_size= grid_size,
        patch_size=4,
        in_chans=channels,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=5,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1
    )

    # Forward pass
    output = model(batch_voxel_grid)

    print(output.shape)
    


if __name__ == "__main__":
    test_swin_transformer() 