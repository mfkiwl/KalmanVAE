import torch
import torch.nn as nn

from utils import Gaussian_Encoder, Gaussian_Decoder, Bernoulli_Decoder

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
                                        latent_dim=self.latent_dim, 
                                        n_channels=[16])
        
        self.decoder = Bernoulli_Decoder(channels_in=n_channels_in, 
                                        image_size=self.image_size, 
                                        latent_dim=self.latent_dim, 
                                        n_channels=[16])

    def log_gaussian(x, mean, var):
        const_log_pdf = (- 0.5 * torch.log(2 * torch.pi)).astype('float32')
        return const_log_pdf - torch.log(var)
    
    def forward(self, x):

        # encode sample
        a_dist = self.encoder(x) 
        a_sample = a_dist.rsample()

        # get reconstruction i.e. get p_{theta} (x|a)
        x_dist = self.decoder(a_sample)

        return x_dist, a_dist.mean, a_dist.variance
