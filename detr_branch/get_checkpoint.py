#https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth from Meta detr model zoo https://github.com/facebookresearch/detr#model-zoo
import torch,torchvision
import torchvision.transforms as T


# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)

# Remove class weights that are not needed from COCO dataset (basically all of them)
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]

# Save
torch.save(checkpoint,
           'detr/detr-r50_no-class-head.pth')

