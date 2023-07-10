from torch import nn


class RSENetDecoder(nn.Module):
    def __init__(
        self,
        decoderInputSize: int = 3,
        decoder1Size: int = 3,
        decoder2Size: int = 3,
        outputSize: int = 3,
        dropoutRate:float = 0.3
    ):
        super().__init__()
        
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
        decoder = self.decoder(x)
        return decoder

class RSENetEncoder(nn.Module):
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
    _ = RSENetEncoder()
    _ = RSENetDecoder()
