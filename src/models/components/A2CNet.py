from typing import Tuple
import torch.nn as nn
import torch
class ActorNet(nn.Module):
    def __init__(
        self,
        m:int = 3, # number of assets
        f:int = 3, # number of features
        n:int = 5, # window length

        kernelSizeConv1:Tuple[int,int]=(1,3),
        # kernelDepthConv1:int=3,

        kernelSizeConv2:Tuple[int,int]=(1,5),
        # kernelDepthConv2:int=3,


        kernelSizeConv3:Tuple[int,int]=(1,1),
        # kernelDepthConv3:int=4,
    ):
        super().__init__()
    
        self.conv1 = nn.Conv2d(f, f, kernel_size=kernelSizeConv1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=kernelSizeConv2)
        self.conv3 = nn.Conv2d(f, f + 1, kernel_size=kernelSizeConv3)
        self.softmax = nn.Softmax(dim=1)

        

    def forward(self, Xt, prevWeights):
        # First Convolution
        Xt = self.conv1(Xt)
        Xt = nn.Tanh(Xt)

        # Second Convolution
        Xt = self.conv2(Xt)
        Xt = nn.Tanh(Xt)

        # Third Convolution
        # Concatenate with previous weights
        Xt = torch.cat((prevWeights, Xt), dim=1)
        Xt = self.conv3(Xt)
        Xt = nn.Tanh(Xt)

        # Flatten and add bias
        Xt = Xt.view(Xt.size(0), -1)  # Reshape to (batch_size, (f + 1))
        Xt = torch.cat((torch.ones(Xt.size(0), 1), Xt), dim=1)

        # Softmax Layer
        distributedWeights = self.softmax(Xt)

        return distributedWeights

class CriticNet(nn.Module):
    def __init__(
        self,
        inputSize: int = 11,
        encoder1Size: int = 10,
        encoder2Size: int = 9,
        encoderOutputSize: int=3,
        dropoutRate:float = 0.3
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputSize, encoder1Size),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True),
            nn.Linear(encoder1Size, encoder2Size),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True),
            nn.Linear(encoder2Size, encoderOutputSize),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoder  = self.encoder(x)
        return encoder



if __name__ == "__main__":
    _ = ActorNet()
    _ = CriticNet()





