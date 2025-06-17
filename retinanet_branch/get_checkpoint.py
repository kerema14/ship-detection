
import torch


# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
            map_location='cpu',
            check_hash=True)


# Save
torch.save(checkpoint,
           'pytorch-retinanet/retinanet-r50_no-class-head.pth')



