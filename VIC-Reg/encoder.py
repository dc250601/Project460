import torch
import torch.nn as nn
import timm

name  =['resnet18',
 'resnet18d',
 'resnet26',
 'resnet26d',
 'resnet26t',
 'resnet32ts',
 'resnet33ts',
 'resnet34',
 'resnet34d',
 'resnet50',
 'resnet50_gn',
 'resnet50d',
 'resnet50t',
 'resnet51q',
 'resnet61q',
 'resnet101',
 'resnet101d',
 'resnet152',
 'resnet152d',
 'resnet200',
 'resnet200d',
 'efficientnet_b0',
 'efficientnet_b1',
 'efficientnet_b2',
 'efficientnet_b2a',
 'efficientnet_b3',
 'efficientnet_b3a',
 'efficientnet_b4',
 'efficientnet_b5',
 'efficientnet_b6',
 'efficientnet_b7',
 'efficientnet_b8',
 'efficientnet_cc_b0_4e',
 'efficientnet_cc_b0_8e',
 'efficientnet_cc_b1_8e',
 'efficientnet_el',
 'efficientnet_em',
 'efficientnet_es',
 'efficientnet_l2',
 'efficientnet_lite0',
 'efficientnet_lite1',
 'efficientnet_lite2',
 'efficientnet_lite3',
 'efficientnet_lite4',
 'efficientnetv2_l',
 'efficientnetv2_m',
 'efficientnetv2_rw_m',
 'efficientnetv2_rw_s',
 'efficientnetv2_rw_t',
 'efficientnetv2_s',
 'efficientnetv2_xl']


class Encoder(nn.Module):

    def __init__(self, model_name):
        super(Encoder, self).__init__()
        assert model_name in name, f"Currently only Resnet and efficientnet based models are supported, {model_name} not found"
        model = timm.models.create_model(model_name)
        lst = list(model.children())[0:-1]
        sample = torch.rand(1,3,224,224)
        if len(nn.Sequential(*lst)(sample).shape) == 4:
            lst.append(timm.models.layers.SelectAdaptivePool2d(pool_type="avg",
                                                               flatten=nn.Flatten(start_dim=1,end_dim=-1)))
        self.backbone = nn.Sequential(*lst)

    def forward(self, x):
        x = self.backbone(x)
        return x
