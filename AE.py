import torch
import torch.nn as nn

from utils import Gaussian_Encoder, Gaussian_Decoder

class AE(nn.Module):

    def __init__(self,
                 n_channels_in,
                 images_size):
        
        super(AE, self).__init__()

        self.n_channels_in = n_channels_in
        self.image_size = images_size

        self.encoder = Gaussian_Encoder(channels_in=n_channels_in, 
                                        image_size=self.image_size)
        
        self.decoder = Gaussian_Decoder(channels_in=n_channels_in, 
                                        image_size=self.image_size)
    
    def forward(self, x):
        
        bs = x.size(0)
        seq_len = x.size(1)
        n_channels = x.size(2)
        side = x.size(3)

        x = self.encoder(x.view(-1,*self.x.shape[2:]))
        x = self.decoder(x).view(bs, seq_len, n_channels, side, side)

        return x

    
    
