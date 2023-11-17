import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.distributions as Distributions
import torch.distributions as D

def log_likelihood(x, mean, var, device):
    c = (- 0.5 * torch.log(torch.tensor([2 * torch.pi]))).to(device)
    ll = c + (- torch.log(var)/2 - torch.square(x-mean)/(2*var))
    return ll

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
            
            # return mean, std
            return Distributions.Normal(loc=mean, scale=std)

        else:
            return x
    
class Gaussian_Decoder_(nn.Module):
    def __init__(self, 
                 channels_in,
                 image_size,
                 latent_dim=0, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3, 
                 out_var=0.1):
        
        super(Gaussian_Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.var = out_var
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
        
        self.n_channels = n_channels # reversed wrt to input in constructor
        
    def forward(self, encodings, recon_only=False, clip=False):
        if self.latent_dim > 0:
            x = F.relu(self.to_conv(encodings)).view(-1, 
                                                    self.n_channels[0], 
                                                    self.conv_out_size[-1], 
                                                    self.conv_out_size[-1])

        for deconv_layer in self.conv_tranpose_modules:
            x = F.relu(deconv_layer(x))
        
        if recon_only:
            x = self.to_mean(x)
            if clip:
                x = x.clamp(0., 1.)

        if self.latent_dim > 0 and not recon_only:
            mean = self.to_mean(x)
            x_hat = mean + torch.sqrt(self.var)*torch.randn_like(mean)
            if clip:
                x_hat = x_hat.clamp(0., 1.)

            return x_hat, mean
        
        else:
            return x
        
class Bernoulli_Decoder_(nn.Module):
    def __init__(self, 
                 channels_in,
                 image_size,
                 latent_dim=0, 
                 n_channels=[16, 32, 64], 
                 kernel_size=3):
        
        super(Bernoulli_Decoder, self).__init__()

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
        
        self.n_channels = n_channels # reversed wrt to input in constructor
        
    def forward(self, encodings, recon_only=False, clip=False):
        if self.latent_dim > 0:
            x = F.relu(self.to_conv(encodings)).view(-1, 
                                                    self.n_channels[0], 
                                                    self.conv_out_size[-1], 
                                                    self.conv_out_size[-1])

        for deconv_layer in self.conv_tranpose_modules:
            x = F.relu(deconv_layer(x))
        
        if recon_only:
            x = self.to_mean(x)
            if clip:
                x = x.clamp(0., 1.)

        if self.latent_dim > 0 and not recon_only:
            x = self.to_mean(x)
            return Distributions.Bernoulli(logits=x)
        
        else:
            return x


#########################################################################################################


def compute_conv2d_output_size(input_size, kernel_size, stride, padding):
    h, w = input_size
    h_out = (h - kernel_size + 2 * padding) // stride + 1
    w_out = (w - kernel_size + 2 * padding) // stride + 1

    return h_out, w_out


class Encoder(nn.Module):
    def __init__(self, image_size, image_channels, a_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=image_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )

        self.fc_mean = nn.Linear(
            in_features=32 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )
        self.fc_std = nn.Linear(
            in_features=32 * conv_output_size[0] * conv_output_size[1],
            out_features=a_dim,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_mean = self.fc_mean(x.view(x.shape[0], -1))
        x_std = F.softplus(self.fc_std(x.view(x.shape[0], -1)))

        return D.Normal(loc=x_mean, scale=x_std)


class BernoulliDecoder(nn.Module):
    def __init__(self, a_dim, image_size, image_channels):
        super(BernoulliDecoder, self).__init__()

        conv_output_size = image_size
        for _ in range(3):
            conv_output_size = compute_conv2d_output_size(
                conv_output_size, kernel_size=3, stride=2, padding=1
            )
        self.conv_output_size = conv_output_size

        self.fc = nn.Linear(
            in_features=a_dim,
            out_features=32 * conv_output_size[0] * conv_output_size[1],
        )
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=image_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 32, *self.conv_output_size)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)

        return D.Bernoulli(logits=x)
