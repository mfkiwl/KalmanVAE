import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from Kalman_Filter import Kalman_Filter
from utils import Gaussian_Encoder, Gaussian_Decoder, Bernoulli_Decoder, log_likelihood

class KalmanVAE(nn.Module):
    
    def __init__(self, 
                 n_channels_in,
                 image_size, 
                 dim_a, 
                 dim_z, 
                 K, 
                 T, 
                 recon_scale=0.3,
                 dim_u=0, 
                 x_var=0.01, 
                 use_bernoulli=False,
                 use_MLP=True):
        
        super(KalmanVAE, self).__init__()
        
        ## initialize variables (for Kalman filter)
        self.dim_a = dim_a
        self.dim_z = dim_z
        self.K = K
        self.T = T
        self.use_MLP = use_MLP

        ## initialize Kalman Filter
        self.kalman_filter = Kalman_Filter(dim_z=self.dim_z, 
                                           dim_a=self.dim_a, 
                                           K=self.K, 
                                           T=self.T, 
                                           use_MLP=self.use_MLP)

        self.mu_0 = torch.zeros(self.dim_z).float()
        self.sigma_0 = torch.eye(self.dim_z).float()

        # initialize other variables
        self.n_channels_in = n_channels_in
        self.recon_scale = recon_scale
        self.x_var = torch.tensor(x_var)
        self.use_bernoulli = use_bernoulli

        # initialize encoder and decoder
        self.encoder = Gaussian_Encoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        latent_dim=self.dim_a)
        if use_bernoulli:
            self.decoder = Bernoulli_Decoder(channels_in=n_channels_in, 
                                            image_size=image_size, 
                                            latent_dim=self.dim_a)
        else:
            self.decoder = Gaussian_Decoder(channels_in=n_channels_in, 
                                        image_size=image_size, 
                                        latent_dim=self.dim_a,
                                        out_var=self.x_var)

    def calculate_loss(self, x, train_dyn_net=True, recon_only=False):

        ##### forward pass
        # get batch size and sequence length
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #### VAE - part
        # encode samples i.e get q_{phi} (a|x)
        a_mean, a_std = self.encoder(x.view(-1, *x.shape[2:]))
        
        # sample from q_{phi} (a|x)
        a_sample = (a_mean + a_std*torch.normal(mean=torch.zeros_like(a_mean))).view(batch_size, seq_len, self.dim_a)
        a_mean = a_mean.view(batch_size, seq_len, self.dim_a)
        a_std = a_std.view(batch_size, seq_len, self.dim_a)

        # get reconstruction i.e. get p_{theta} (x|a)
        if self.use_bernoulli:
            x_dist = self.decoder(a_sample.view(-1, self.dim_a))
            x_hat = x_dist.mean
        else:
            x_hat, x_mean = self.decoder(a_sample)

        if recon_only:
            return x_hat

        #### LGSSM - part
        # get smoothing Distribution: p_{gamma} (z|a) 
        params = self.kalman_filter.filter(a_sample, train_dyn_net, device=x.get_device())
        _, _, _, _, _, _, A, C, alpha = params
        smoothed_means, smoothed_covariances = self.kalman_filter.smooth(a_sample, params)

        #### VAE - part
        #### p_{theta} (x|a)
        if self.use_bernoulli:
            log_p_x_given_a = x_dist.log_prob(x.reshape(-1, *x.shape[2:])).sum(0).mean(0).sum()
        else:
            log_p_x_given_a = log_likelihood(x.view(x.size(0)*x.size(1), -1),
                                            x_mean.view(x.size(0)*x.size(1), -1),
                                            self.x_var, 
                                            device=x.get_device()).mean(1).sum()
        '''
        self.x_dist = MultivariateNormal(loc=self.x_mean.view(self.x.size(0)*self.x.size(1), -1), 
                                         covariance_matrix=(torch.eye(self.x.size(-1)**2)*self.x_var).to(self.x.get_device()))
        log_p_x_given_a = self.x_dist.log_prob(self.x.view(self.x.size(0)*self.x.size(1), -1)).sum().div(num_el)
        '''
        
        #### q_{phi} (a|x)
        a_dist = MultivariateNormal(loc=a_mean,   
                                    covariance_matrix=torch.diag_embed(a_std))
        log_q_a_given_x = a_dist.log_prob(a_sample).mean(1).sum()


        #### LGSSM - part
        #### p_{gamma} (z|a)
        p_z_given_a = MultivariateNormal(loc=torch.cat(smoothed_means).view(x.size(0), x.size(1), -1), 
                                         scale_tril=torch.linalg.cholesky(torch.cat(smoothed_covariances).view(x.size(0), x.size(1), self.dim_z, self.dim_z)))
        z_sample = p_z_given_a.sample() 
        log_p_z_given_a = p_z_given_a.log_prob(z_sample).mean(0).sum()

        #### p_{gamma} (a|z)        
        a_transition = torch.matmul(C, z_sample.view(x.size(0), x.size(1), -1).unsqueeze(-1)).squeeze(-1)
        to_sample = a_sample.to(x.get_device()) - a_transition.to(x.get_device())
        p_a_given_z = MultivariateNormal(loc=torch.zeros(self.dim_a).to(x.get_device()), 
                                         scale_tril=torch.linalg.cholesky(self.kalman_filter.R))
        log_p_a_given_z = p_a_given_z.log_prob(to_sample).mean(0).sum()

        #### p_{gamma} (z_T|z_T-1, .., z_1) 
        z_transition = torch.matmul(A, z_sample.view(x.size(0), x.size(1), -1).unsqueeze(-1)).squeeze(-1)
        to_sample = z_sample.view(x.size(0), x.size(1), -1)[:, 1:, :] - z_transition[:, :-1, :]
        p_zT_given_zt = MultivariateNormal(loc=torch.zeros(self.dim_z).to(x.get_device()), 
                                           scale_tril=torch.linalg.cholesky(self.kalman_filter.Q))
        log_p_zT_given_zt = p_zT_given_zt.log_prob(to_sample).mean(0).sum()

        #### p_{gamma} (z_0)
        p_z0 = MultivariateNormal(loc=self.mu_0.to(x.get_device()), scale_tril=torch.linalg.cholesky(self.sigma_0.to(x.get_device())))
        log_p_z0 = p_z0.log_prob(z_sample[:, 0, :]).mean(0)

        # create loss dictionary
        loss_dict = {'reconstruction loss': self.recon_scale*log_p_x_given_a.detach().cpu().numpy(),
                     'encoder loss': log_q_a_given_x.detach().cpu().numpy(), 
                     'LGSSM observation log likelihood': log_p_a_given_z.detach().cpu().numpy(),
                     'LGSSM tranisition log likelihood': log_p_zT_given_zt.detach().cpu().numpy(), 
                     'LGSSM tranisition log posterior': log_p_z_given_a.detach().cpu().numpy()}

        return x_hat, alpha, -self.recon_scale*(log_p_x_given_a) + log_q_a_given_x + log_p_z_given_a - log_p_a_given_z - log_p_zT_given_zt - log_p_z0, loss_dict

    def impute(self, x, mask, sample=False):

        # get dims
        bs = x.size(0)
        seq_len = x.size(1)
        
        # convert mask to torch tensor
        mask_t = torch.Tensor(mask).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(bs, 1, 1, x.size(3), x.size(4)).to(x.get_device())

        # mask input
        x_masked = mask_t*x

        # feed masked sample in encoder
        a_mean, a_std = self.encoder(x_masked.view(-1,*x_masked.shape[2:]))

        # sample from q_{phi} (a|x)
        if sample:
            a_sample = (a_mean + a_std*torch.normal(mean=torch.zeros_like(a_mean))).view(bs, seq_len, self.dim_a)
        else:
            a_sample = a_mean.view(bs, seq_len, self.dim_a)

        # estimate observation from filtered posterior
        for t, mask_el in enumerate(mask):
            if mask_el != 0:
                continue
            else:
                # get filtered distribution up to t=t-1
                _, _, _, _, next_means, _, A, C, alpha = self.kalman_filter.filter(a_sample, imputation_idx=t, device=x.get_device())
                a_sample[:, t, :] = torch.matmul(C[:, t, :, :], torch.cat(next_means).view(-1, seq_len, self.dim_z)[:, t, :].unsqueeze(-1)).squeeze(-1)
        
        # get filtered+smoothed distribution and smoothed observations
        params = self.kalman_filter.filter(a_sample, device=x.get_device())
        _, _, _, _, next_means, _, A, C, alpha = params
        smoothed_means, _ = self.kalman_filter.smooth(a_sample, params=params)
        smoothed_obs = torch.matmul(C, torch.cat(smoothed_means).view(-1, seq_len, self.dim_z).unsqueeze(-1)).squeeze(1)

        # decode smoothed observations
        if self.use_bernoulli:
            x_dist = self.decoder(smoothed_obs.view(bs*seq_len, -1))
            imputed_data = x_dist.mean
        else:
            imputed_data, _ = self.decoder(smoothed_obs.view(bs*seq_len, -1))

        return imputed_data.view(bs, seq_len, *x.shape[2:]), alpha   

    def generate(self, x, mask, sample=False):  

        # get dims
        bs = x.size(0)

        # convert mask to torch tensor
        mask_t = torch.Tensor(mask).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(bs, 1, 1, x.size(3), x.size(4)).to(x.get_device()) 

        # mask input
        x_masked = mask_t*x

        # feed masked sample in encoder
        a_mean, a_std = self.encoder(x_masked.view(-1,*x_masked.shape[2:]))

        # sample from q_{phi} (a|x)
        if sample:
            a_sample = (a_mean + a_std*torch.normal(mean=torch.zeros_like(a_mean))).view(bs, len(mask), self.dim_a)
        else:
            a_sample = a_mean.view(bs, len(mask), self.dim_a)
        
        # estimate observation from filtered posterior
        for t, mask_el in enumerate(mask):
            if mask_el != 0:
                continue
            else:
                # get filtered distribution up to t=t-1
                if t>=self.T:
                    start_idx = t-self.T + 1
                    end_idx = t+1
                    C_idx = -1
                else:
                    start_idx = 0
                    end_idx = self.T
                    C_idx = t
                _, _, _, _, next_means, _, A, C, alpha = self.kalman_filter.filter(a_sample[:, start_idx:end_idx, :], imputation_idx=t, device=x.get_device())
                a_sample[:, t, :] = torch.matmul(C[:, C_idx, :, :], torch.cat(next_means).view(-1, len(mask), self.dim_z)[:, C_idx, :].unsqueeze(-1)).squeeze(-1)

        # decode predicted observations
        if self.use_bernoulli:
            x_dist = self.decoder(a_sample.view(bs*len(mask), -1))
            generated_data = x_dist.mean
        else:
            generated_data, _ = self.decoder(a_sample.view(bs*len(mask), -1))

        return generated_data.view(bs, len(mask), *x.shape[2:])   