#https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth from Meta detr model zoo https://github.com/facebookresearch/detr#model-zoo
import torch,torchvision
import torchvision.transforms as T


# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
            map_location='cpu',
            check_hash=True)


# Save
torch.save(checkpoint,
           'retinanet-r50_no-class-head.pth')



