import torch

from dataset.dataset import ParkingDataset
from train.resnet import train_parking_model
from utils.visual_tools import visualize_parking_sample
from utils.visual_tools import show_roi_patches
from utils.pooling import roi_pooling

# dataset = ParkingDataset()
#
# # Donnée 0
# x = dataset[0]
#
# image = x[1]
# rois = x[2]
#
# patches = roi_pooling(image=image, rois=rois, output_size=256)
# show_roi_patches(patches, save_path="outputs/roi_patches.png")
# # visualize_parking_sample(dataset, idx=0)

#
if __name__ == '__main__':
    torch.cuda.empty_cache()
    model = train_parking_model(
        data_root="data",
        batch_size=4,
        num_epochs=15,
        lr=1e-3,
        pretrained=True,
        freeze_backbone=True
    )
