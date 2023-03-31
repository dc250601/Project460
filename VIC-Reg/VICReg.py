import encoder
import projector
import torch.nn as nn


class VICReg(nn.Module):
    def __init__(self,
                 backbone="resnet50",
                 projector_depth=2,
                 projector_width=8192):
        super().__init__()
        self.encoder_network = encoder.Encoder(model_name=backbone)
        self.projector_network = projector.Projector(Encoder=self.encoder_network,
                                                     Breadth=projector_width,
                                                     Depth=projector_depth)

    def forward(self, x, y):
        x = self.encoder_network(x)
        x = self.projector_network(x)

        y = self.encoder_network(y)
        y = self.projector_network(y)

        return x, y
