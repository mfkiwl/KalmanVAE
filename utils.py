import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as Distributions


def compute_convolution_output_size(image_size, n_channels, kernel_size):
    out_sizes = []
    dilation = 1
    for i, _ in enumerate(n_channels):
        if i == 0:
            out_size = image_size - (dilation*(kernel_size-1))
        else:
            out_size = out_sizes[i-1] - (dilation*(kernel_size-1))
        out_sizes.append(out_size)

    return out_sizes


class Gaussian_Encoder(nn.Module):
    def __init__(self, 
                 channels_in,
                 image_size,
                 latent_dim=0, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3):
        
        super(Gaussian_Encoder, self).__init__()

        self.n_channels = n_channels
        self.latent_dim = latent_dim

        # NOTE: - latent_dim = 0 --> Autoencoder
        #       - latent_dim > 0 --> Variational Autoencoder

        modules = []
        for i, n_channel in enumerate(self.n_channels):
            if i == 0:
                in_channels = channels_in
            else:
                in_channels = self.n_channels[i-1]

            modules.append(nn.Conv2d(in_channels=in_channels,
                                     out_channels=n_channel, 
                                     kernel_size=kernel_size))
        
        self.conv_modules = nn.ModuleList(modules=modules)
        
        self.conv_out_size = compute_convolution_output_size(image_size, self.n_channels, kernel_size)

        if latent_dim > 0:
            self.to_mean = nn.Linear(in_features=self.n_channels[-1]*self.conv_out_size[-1]*self.conv_out_size[-1], 
                                    out_features=latent_dim)
            
            self.to_std = nn.Linear(in_features=self.n_channels[-1]*self.conv_out_size[-1]*self.conv_out_size[-1], 
                                    out_features=latent_dim)
        
    def forward(self, x):
        for conv_layer in self.conv_modules:
            x = F.relu(conv_layer(x))
        
        if self.latent_dim > 0:
            x = x.view(-1, self.n_channels[-1]*(self.conv_out_size[-1]**2))
            mean = self.to_mean(x)
            std = F.softplus(self.to_std(x))
            return mean, std

        else:
            return x
    
class Gaussian_Decoder(nn.Module):
    def __init__(self, 
                 channels_in,
                 image_size,
                 latent_dim=0, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3):
        
        super(Gaussian_Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.conv_out_size = compute_convolution_output_size(image_size, n_channels, kernel_size)
        in_channels = n_channels[-1]

        # NOTE: - latent_dim = 0 --> Autoencoder
        #       - latent_dim > 0 --> Variational Autoencoder

        if self.latent_dim > 0: 
            self.to_conv = nn.Linear(in_features=latent_dim, 
                                     out_features=(self.conv_out_size[-1]**2)*in_channels)
        
        modules = []
        n_channels = n_channels[::-1]
        for i, n_channel in enumerate(n_channels):
            if i > 0:
                in_channels = n_channels[i-1]

            modules.append(nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=n_channel, 
                                              kernel_size=kernel_size, 
                                              stride=1, 
                                              padding=0, 
                                              output_padding=0))
        
        self.conv_tranpose_modules = nn.ModuleList(modules=modules)

        if self.latent_dim > 0:
            self.to_mean = nn.ConvTranspose2d(in_channels=n_channels[-1],
                                            out_channels=channels_in, 
                                            kernel_size=kernel_size,
                                            stride=1, 
                                            padding=1, 
                                            output_padding=0)
            
            self.to_std = nn.ConvTranspose2d(in_channels=n_channels[-1],
                                            out_channels=channels_in, 
                                            kernel_size=kernel_size, 
                                            stride=1, 
                                            padding=1, 
                                            output_padding=0)
        
        self.n_channels = n_channels # reversed wrt to input in constructor
        
    def forward(self, encodings, recon_only=False):
        if self.latent_dim > 0:
            x = F.relu(self.to_conv(encodings)).view(-1, 
                                                    self.n_channels[0], 
                                                    self.conv_out_size[-1], 
                                                    self.conv_out_size[-1])

        for deconv_layer in self.conv_tranpose_modules:
            x = F.relu(deconv_layer(x))

        if self.latent_dim > 0 and not recon_only:
            mean = self.to_mean(x)
            std = F.softplus(self.to_std(x))
            x_hat = mean + std*torch.randn_like(std)
            return x_hat, mean, std
        
        else:
            return x
        
        
            
        




