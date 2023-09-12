# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
import torch.nn as nn
import torchvision.models as models
from exceptions.exceptions import InvalidBackboneError
from vit_pytorch.vivit import ViT


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        # self.resnet_dict = {"mvit": models.video.mvit_v1_b(pretrained=False, num_classes=out_dim)}
        # self.backbone = self._get_basemodel(base_model)
        # dim_mlp = self.backbone.head[1].in_features
        # print(dim_mlp)
        # self.backbone.head[1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.head[1])

        # self.backbone = ViT(
        #     image_size=224,  # image size
        #     frames=30,  # number of frames
        #     image_patch_size=16,  # image patch size
        #     frame_patch_size=2,  # frame patch size
        #     num_classes=out_dim,
        #     dim=768,
        #     spatial_depth=6,  # depth of the spatial transformer
        #     temporal_depth=6,  # depth of the temporal transformer
        #     heads=8,
        #     mlp_dim=768
        # )
        # dim_mlp = self.backbone.mlp_head[1].in_features
        # self.backbone.mlp_head[1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), self.backbone.mlp_head[1])

        self.resnet_dict = {"mvit": models.video.r3d_18(pretrained=False, num_classes=out_dim)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
