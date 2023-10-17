import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from Kalman_Filter import Kalman_Filter
from VAE import Guassian_Encoder, Guassian_Decoder

class KalmanVAE(nn.module):
    
    def __init__(self, 
                 n_channels_in,
                 image_size, 
                 dim_a, 
                 dim_z, 
                 K):
        
        ## initialize variables
        self.n_channels_in = n_channels_in
        self.dim_a = dim_a
        self.dim_z = dim_z
        self.K = K

        ## initialize Kalman Filter
        self.kalman_filter = Kalman_Filter(dim_z=self.dim_z, 
                                           dim_a=self.dim_a, 
                                           K=self.K)

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
        self.A = None
        self.B = None
        self.C = None

        # x sample (ground-truth) and a sample
        self.x = None
        self.a_sample = None

        # initialize encoder 
        self.encoder = Guassian_Encoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        a_dim=self.dim_a)
        
        self.decoder = Guassian_Decoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        a_dim=self.dim_a)

    def forward(self, x):

        self.x = x
        
        #### VAE - part
        # encode samples i.e get q_{phi} (a|x)
        self.a_dist, self.a_mean, self.a_cov = self.encoder(x)

        # sample from q_{phi} (a|x)
        self.a_sample = self.a_dist.sample()

        # get reconstruction i.e. get p_{theta} (x|a)
        self.x_dist, self.x_mean, self.x_cov = self.decoder(self.a_sample)

        #### LGSSM - part
        ## Smoothing Distribution: p_{gamma} (z|a) 
        # use Kalman filter and then smoother to get smoothed posterior p_{gamma} (z|a)
        params = self.kalman_filter.filter(self.a_sample)
        self.smoothed_means, self.smoothed_covariances, gamma = self.kalman_filter.smooth(self.a_sample, params)
        if len(gamma) == 2:
           self.A, self.B = gamma
        else:
           self.A, self.B, self.C = gamma

    def calculate_loss(self):

        #### VAE - part
        # a_mean, a_cov and a_sample will be used to calculate
        # the log likelihood of q_{phi} (a|x) by essentially 
        # evaluating the Normal distribution with a=a_sample,
        # mean=a_mean (from encoder) and cov=a_cov (from encoder). 
        log_q_a_given_x = self.a_dist.log_prob(self.a_sample)

        # x_mean and x_cov are used for calculating the 
        # log likelihood p_{theta} (x|a) where x=ground-truth
        # so the estimantion of log(p_{theta}(x|a)) is equal to 
        # the evaluation of a Normal distribution with x=ground-truth, 
        # mean=x_mean (from decoder) and covariance=x_cov (from decoder).
        log_p_x_given_a = self.x_dist.log_prob(self.x)


        #### LGSSM - part
        # first we create p_{gamma} (z|a) from the smoothed
        # means and covariances as a Multivariate Normal
        p_z_given_a = MultivariateNormal(loc=self.smoothed_means, 
                                         covariance_matrix=self.smoothed_covariances)
        
        # sample z from smoothed posterior p_{gamma} (z|a) --> (bs, seq_len, dim_z)
        # and evaluate p_z_given_a using z_sample
        z_sample = p_z_given_a.sample() 
        log_p_z_given_a = p_z_given_a.log_prob(z_sample)

        # create p_{gamma} (a|z) = C (bs, seq_len, dim_a, dim_z) * z (bs, seq_len, dim_z)
        # and evaluate it with sample from a_dist
        a_transition = torch.matmul(self.C, z_sample.unsqueeze(-1)).squeeze(-1)
        p_a_given_z = MultivariateNormal(loc=a_transition, 
                                         covariance_matrix=self.kalman_filter.R)
        log_p_a_given_z = p_a_given_z.log_prob(self.a_sample)

        # create transitional distribution --> p(z_T|z_{T-1})p(z_{T-1}|z_{T-2}) ... p(z_2|z_1)p(z_1)
        z_transition = torch.matmul(self.A, z_sample.unsqueeze(-1)).squeeze(-1)
        p_zT_given_zt = MultivariateNormal(loc=z_transition, 
                                           covariance_matrix=self.kalman_filter.Q)
        log_p_zT_given_zt = p_zT_given_zt.log_prob(self.z_sample)

        return log_p_x_given_a + log_q_a_given_x + log_p_z_given_a + log_p_a_given_z + log_p_zT_given_zt

