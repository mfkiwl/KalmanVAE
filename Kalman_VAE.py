import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from Kalman_Filter import Kalman_Filter
from utils import Gaussian_Encoder, Gaussian_Decoder

class KalmanVAE(nn.Module):
    
    def __init__(self, 
                 n_channels_in,
                 image_size, 
                 dim_a, 
                 dim_z, 
                 K, 
                 T, 
                 recon_scale=0.6,
                 dim_u=0):
        
        super(KalmanVAE, self).__init__()
        
        ## initialize variables
        self.n_channels_in = n_channels_in
        self.dim_a = dim_a
        self.dim_z = dim_z
        self.K = K
        self.T = T
        self.recon_scale = recon_scale

        ## initialize Kalman Filter
        self.kalman_filter = Kalman_Filter(dim_z=self.dim_z, 
                                           dim_a=self.dim_a, 
                                           K=self.K, 
                                           T=self.T)

        ## declare model variables
        # encoder mean and covariance
        self.a_dist = None
        self.a_mean = None
        self.a_cov = None

        # decoder mean and covariance
        self.x_dist = None
        self.x_mean = None
        self.x_cov = None

        # smoothed mean and variance
        self.smoothed_means = None
        self.smoothed_covariances = None

        # dynamics matrices
        self.A = self.kalman_filter.A
        if dim_u > 0:
            self.B = self.kalman_filter.B
        self.C = self.kalman_filter.C

        # dynamics parameter 
        self.dynamics_net = self.kalman_filter.dyn_net

        # x sample (ground-truth) and a sample
        self.x = None
        self.a_sample = None

        # initialize encoder and decoder
        self.encoder = Gaussian_Encoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        a_dim=self.dim_a)
        
        self.decoder = Gaussian_Decoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        a_dim=self.dim_a)

    def forward(self, x):
        
        # get batch size and sequence length
        self.x = x
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #### VAE - part
        # encode samples i.e get q_{phi} (a|x)
        self.a_mean, self.a_std = self.encoder(x.view(-1,*self.x.shape[2:]))
        
        # sample from q_{phi} (a|x)
        self.a_sample = (self.a_mean + self.a_std*torch.normal(mean=torch.zeros_like(self.a_mean))).view(batch_size, seq_len, self.dim_a)

        # get reconstruction i.e. get p_{theta} (x|a)
        x_hat, self.x_mean, self.x_std = self.decoder(self.a_sample)

        #### LGSSM - part
        # get smoothing Distribution: p_{gamma} (z|a) 
        params = self.kalman_filter.filter(self.a_sample, device=self.A.get_device())
        _, _, _, _, _, _, A, C = params
        self.smoothed_means, self.smoothed_covariances = self.kalman_filter.smooth(self.a_sample, params)

        return x_hat, A, C

    def calculate_loss(self, A, C):

        num_el = self.x.size(0)*self.x.size(1)

        #### VAE - part
        # a_mean, a_cov and a_sample will be used to calculate
        # the log likelihood of q_{phi} (a|x) by essentially 
        # evaluating the Normal distribution with a=a_sample,
        # mean=a_mean (from encoder) and cov=a_cov (from encoder).
        self.a_dist = MultivariateNormal(loc=self.a_mean,   
                                         covariance_matrix=torch.diag_embed(self.a_std))
        log_q_a_given_x = self.a_dist.log_prob(self.a_sample.view(-1, self.dim_a)).sum().div(num_el)

        # x_mean and x_cov are used for calculating the 
        # log likelihood p_{theta} (x|a) where x=ground-truth
        # so the estimantion of log(p_{theta}(x|a)) is equal to 
        # the evaluation of a Normal distribution with x=ground-truth, 
        # mean=x_mean (from decoder) and covariance=x_cov (from decoder).
        # print((self.x_mean.view(self.x.size(0), self.x.size(1), self.x.size(2), self.x.size(3), self.x.size(4)) - self.x).sum())
        self.x_dist = MultivariateNormal(loc=self.x_mean.view(self.x.size(0)*self.x.size(1), -1), 
                                         covariance_matrix=torch.diag_embed(self.x_std.view(self.x_std.size(0), -1)).to(self.x.get_device()))
        log_p_x_given_a = self.x_dist.log_prob(self.x.view(self.x.size(0)*self.x.size(1), -1)).sum().div(num_el)


        #### LGSSM - part
        # first we create p_{gamma} (z|a) from the smoothed
        # means and covariances as a Multivariate Normal
        p_z_given_a = MultivariateNormal(loc=torch.cat(self.smoothed_means), 
                                         scale_tril=torch.linalg.cholesky(torch.cat(self.smoothed_covariances)))
        
        # sample z from smoothed posterior p_{gamma} (z|a) --> (bs, seq_len, dim_z)
        # and evaluate p_z_given_a using z_sample
        z_sample = p_z_given_a.sample() 
        log_p_z_given_a = p_z_given_a.log_prob(z_sample).sum().div(num_el)

        # create p_{gamma} (a|z) = C (bs, seq_len, dim_a, dim_z) * z (bs, seq_len, dim_z)
        # and evaluate it with sample from a_dist
        a_transition = torch.matmul(C, z_sample.view(self.x.size(0), self.x.size(1), -1).unsqueeze(-1)).squeeze(-1)
        to_sample = self.a_sample.to(self.x.get_device()) - a_transition.to(self.x.get_device())
        p_a_given_z = MultivariateNormal(loc=torch.zeros(self.dim_a).to(self.x.get_device()), 
                                         scale_tril=torch.linalg.cholesky(self.kalman_filter.R))
        log_p_a_given_z = p_a_given_z.log_prob(to_sample).sum().div(num_el)

        # create transitional distribution --> p(z_T|z_{T-1})p(z_{T-1}|z_{T-2}) ... p(z_2|z_1)p(z_1)
        z_transition = torch.matmul(A, z_sample.view(self.x.size(0), self.x.size(1), -1).unsqueeze(-1)).squeeze(-1)
        to_sample = z_sample.view(self.x.size(0), self.x.size(1), -1) - z_transition
        p_zT_given_zt = MultivariateNormal(loc=torch.zeros(self.dim_z).to(self.x.get_device()), 
                                           scale_tril=torch.linalg.cholesky(self.kalman_filter.Q))
        log_p_zT_given_zt = p_zT_given_zt.log_prob(to_sample).sum().div(num_el)
        
        loss_dict = {'reconstruction loss': self.recon_scale*log_p_x_given_a.detach().cpu().numpy(),
                     'encoder loss': log_q_a_given_x.detach().cpu().numpy(), 
                     'LGSSM observation log likelihood': log_p_a_given_z.detach().cpu().numpy(),
                     'LGSSM tranisition log likelihood': log_p_zT_given_zt.detach().cpu().numpy(), 
                     'LGSSM tranisition log posterior': log_p_z_given_a.detach().cpu().numpy()}

        return -self.recon_scale*log_p_x_given_a + log_q_a_given_x + log_p_z_given_a - log_p_a_given_z - log_p_zT_given_zt, loss_dict

