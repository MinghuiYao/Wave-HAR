import torch
from torch import nn
from waveconv import Waveconv

class HARModel(nn.Module):
    def __init__(self,dims=[64,128,256,512], kernel_size=(5,1), stride=(4,1), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            Waveconv(in_channels=1, out_channels=dims[0], kernel_size=kernel_size),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )
        self.conv2 = nn.Sequential(
            Waveconv(in_channels=dims[0], out_channels=dims[1], kernel_size=kernel_size),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )
        self.conv3 = nn.Sequential(
            Waveconv(in_channels=dims[1], out_channels=dims[2], kernel_size=kernel_size),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        )
        self.classifier = nn.Linear(dims[3], 6)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x.flatten(1))
        return x
        
if __name__ == '__main__':
    input   = torch.randn(3, 1, 128, 9)  
    wtconv  = HARModel()
    output  = wtconv(input)
    print(output.size())