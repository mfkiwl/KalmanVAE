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
                 a_dim, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3):
        
        super(Gaussian_Encoder).__init__()

        modules = []
        for i, n_channel in enumerate(n_channels):
            if i == 0:
                in_channels = channels_in
            else:
                in_channels = n_channels[i-1]

            modules.append(nn.Conv2d(in_channels=in_channels,
                                     out_channels=n_channel, 
                                     kernel_size=kernel_size))
        
        self.conv_modules = nn.ModuleList(modules=modules)
        
        conv_out_size = compute_convolution_output_size(image_size, n_channels, kernel_size)
        
        self.to_mean = nn.Linear(in_features=n_channels[-1]*conv_out_size[-1]*2, 
                                 out_features=a_dim)
        
        self.to_std = nn.Linear(in_features=n_channels[-1]*conv_out_size[-1]*2, 
                                 out_features=a_dim)
        
    def forward(self, x):
        for conv_layer in self.conv_modules:
            x = F.relu(conv_layer(x))
        
        mean = self.to_mean(x)
        std = F.softplus(self.to_std(x))

        a_dist = Distributions.Normal(loc=mean, scale=std)

        return a_dist, mean, std
    

class Gaussian_Decoder(nn.Module):
    def __init__(self, 
                 channels_in,
                 image_size,
                 a_dim, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3):
        
        super(Gaussian_Decoder).__init__()

        self.conv_out_size = compute_convolution_output_size(image_size, n_channels, kernel_size)

        in_channels = n_channels[-1]

        self.to_conv = nn.Linear(in_features=a_dim, 
                                 out_features=self.conv_out_size[-1]*2*in_channels)
        
        modules = []
        n_channels = n_channels.reverse()
        for i, n_channel in enumerate(n_channels):
            if i > 0:
                in_channels = n_channels[i-1]

            modules.append(nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=n_channel, 
                                              kernel_size=kernel_size, 
                                              stride=2, 
                                              padding=1, 
                                              output_padding=1))
        
        self.conv_tranpose_modules = nn.ModuleList(modules=modules)

        self.to_mean = nn.ConvTranspose2d(in_channels=n_channels[-1],
                                          out_channels=channels_in, 
                                          kernel_size=kernel_size,
                                          stride=2, 
                                          padding=1, 
                                          output_padding=1)
        
        self.to_std = nn.ConvTranspose2d(in_channels=n_channels[-1],
                                          out_channels=channels_in, 
                                          kernel_size=kernel_size, 
                                          stride=2, 
                                          padding=1, 
                                          output_padding=1)
        
        self.n_channels = n_channels # reversed wrt to input in constructor
        
    def forward(self, encodings):
        x = F.relu(self.to_conv(encodings)).view(-1, 
                                                 self.n_channels[0], 
                                                 self.conv_out_size[-1], 
                                                 self.conv_out_size[-1])
        for deconv_layer in self.conv_tranpose_modules:
            x = F.relu(deconv_layer(x))
        
        mean = self.to_mean(x)
        std = F.softplus(self.to_std(x))

        x_dist = Distributions.Normal(loc=mean, scale=std)

        return x_dist, mean, std
        
        
            
        




