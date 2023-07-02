from torch import nn


class RSENet(nn.Module):
    def __init__(
        self,
        inputSize: int = 11,
        encoder1Size: int = 10,
        encoder2Size: int = 9,

        decoderInputSize: int = 3,
        decoder1Size: int = 3,
        decoder2Size: int = 3,

        outputSize: int = 3,
         
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
            nn.Linear(encoder2Size, decoderInputSize),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(decoderInputSize, decoder1Size),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True),
            nn.Linear(decoder1Size,decoder2Size),
            nn.Dropout(dropoutRate,True),
            nn.ReLU(True),
            nn.Linear(decoder2Size, outputSize),
            nn.Dropout(dropoutRate,True),
            nn.Tanh()
        )
        

    def forward(self, x):

        encoder  = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder


if __name__ == "__main__":
    _ = RSENet()
