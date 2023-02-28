import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self, Encoder, Breadth=8192, Depth=2):
        super(Projector, self).__init__()
        assert Encoder is not None, "The Encoder network is needed to be passsed"
        sample = torch.rand(1,3,224,224)
        shape = Encoder(sample).shape[-1]

        layers = []
        layers.append(nn.Linear(shape,Breadth))
        layers.append(nn.BatchNorm1d(Breadth))
        layers.append(nn.ReLU(True))
        for i in range(Depth-1):
            layers.append(nn.Linear(Breadth,Breadth))
            layers.append(nn.BatchNorm1d(Breadth))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(Breadth,Breadth,bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.model(x)
        return x
