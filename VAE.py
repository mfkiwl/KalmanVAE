import torch
import torch.nn as nn

from utils import Gaussian_Encoder, Gaussian_Decoder

class VAE(nn.Module):

    def __init__(self,
                 n_channels_in,
                 images_size, 
                 latent_dim):
        
        super(VAE, self).__init__()

        self.n_channels_in = n_channels_in
        self.image_size = images_size
        self.latent_dim = latent_dim

        self.encoder = Gaussian_Encoder(channels_in=n_channels_in, 
                                        image_size=self.image_size, 
                                        latent_dim=self.latent_dim)
        
        self.decoder = Gaussian_Decoder(channels_in=n_channels_in, 
                                        image_size=self.image_size, 
                                        latent_dim=self.latent_dim)

    def log_gaussian(x, mean, var):
        const_log_pdf = (- 0.5 * torch.log(2 * torch.pi)).astype('float32')
        return const_log_pdf - torch.log(var)
    
    def forward(self, x):
        mean, std = self.encoder(x)
        gaussian_sample = mean + std*torch.randn_like(std)
        reconstructions = self.decoder(gaussian_sample, recon_only=True, clip=True)

        return reconstructions, mean, std
