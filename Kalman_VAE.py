import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from Kalman_Filter import Kalman_Filter
# from utils import Gaussian_Encoder, Gaussian_Decoder, Bernoulli_Decoder, log_likelihood

from utils import Encoder, BernoulliDecoder

class KalmanVAE(nn.Module):
    
    def __init__(self, 
                 n_channels_in,
                 image_size, 
                 dim_a, 
                 dim_z, 
                 K, 
                 T, 
                 A_init=1., 
                 C_init=0.,
                 recon_scale=0.3,
                 dim_u=0, 
                 x_var=0.01, 
                 use_bernoulli=False,
                 use_MLP=True, 
                 symmetric_covariance=True,
                 R_noise_init=1.0,
                 Q_noise_init=1.0,
                 dtype=torch.float32,
                 train_VAE=True, 
                 device=None):
        
        super(KalmanVAE, self).__init__()
        
        # initialize variables (for Kalman filter)
        self.dim_a = dim_a
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.K = K
        self.T = T
        self.use_MLP = use_MLP
        self.symmetric_covariance = symmetric_covariance
        self.train_VAE = train_VAE
        self.device = device
        
        # initialize Kalman Filter
        self.kalman_filter = Kalman_Filter(dim_z=self.dim_z, 
                                           dim_a=self.dim_a, 
                                           K=self.K, 
                                           T=self.T, 
                                           init_weight_A=A_init, 
                                           init_weight_C=C_init,
                                           use_MLP=self.use_MLP, 
                                           dtype=dtype, 
                                           symmetric_covariance=symmetric_covariance, 
                                           R_init=R_noise_init,
                                           Q_init=Q_noise_init,
                                           device=self.device)

        self.mu_0 = torch.zeros(self.dim_z).float()
        self.sigma_0 = torch.eye(self.dim_z).float()

        # initialize other variables
        self.n_channels_in = n_channels_in
        self.recon_scale = recon_scale
        self.x_var = torch.tensor(x_var)
        self.use_bernoulli = use_bernoulli

        # initialize encoder and decoder
        if self.train_VAE:
            '''
            self.encoder = Gaussian_Encoder(channels_in=n_channels_in, 
                                            image_size=image_size, 
                                            latent_dim=self.dim_a, 
                                            n_channels=[32, 32, 32])
            if use_bernoulli:
                self.decoder = Bernoulli_Decoder(channels_in=n_channels_in, 
                                                image_size=image_size, 
                                                latent_dim=self.dim_a, 
                                                n_channels=[32, 32, 32])
            else:
                self.decoder = Gaussian_Decoder(channels_in=n_channels_in, 
                                            image_size=image_size, 
                                            latent_dim=self.dim_a,
                                            out_var=self.x_var, 
                                            n_channels=[32, 32, 32])
            '''

            self.encoder = Encoder(image_size=[image_size, image_size], 
                                   image_channels=1, 
                                   a_dim=self.dim_a)
            if use_bernoulli:
                self.decoder = BernoulliDecoder(image_size=[image_size, image_size], 
                                                image_channels=1,
                                                a_dim=self.dim_a)
                                  
    def calculate_loss(self, x, train_dyn_net=True, upscale_vae_loss=True, use_mean=False, recon_only=False):
        
        ######################
        ##### Variables ######
        ######################

        # get batch size and sequence length
        batch_size = x.size(0)
        seq_len = x.size(1)

        # decide data type based on type of training
        if not self.train_VAE:
            a_sample = x

        # make sure two modes of averages are not used simultaneously
        assert upscale_vae_loss + use_mean <= 1 



        ######################
        #### Forward pass ####
        ######################

        #### VAE - part
        if self.train_VAE:

            # encode samples i.e get q_{phi} (a|x)
            a_dist = self.encoder(x.view(-1, *x.shape[2:])) 
            a_sample = a_dist.rsample().view(batch_size, seq_len, self.dim_a) # (bs, T, dim_a)

            # get reconstruction i.e. get p_{theta} (x|a)
            if self.use_bernoulli:
                x_dist = self.decoder(a_sample.view(-1, self.dim_a))
                x_hat = x_dist.mean.view(batch_size, seq_len, *x.shape[2:]) # (bs, T, n_ch, dim, dim)
            else:
                x_hat, x_mean = self.decoder(a_sample)
            if recon_only:
                return x_hat
        
        #### LGSSM - part
        # get smoothing Distribution: p_{gamma} (z|a) 
        params = self.kalman_filter.filter(a_sample, 
                                           train_dyn_net=train_dyn_net, 
                                           imputation_idx=None)
        _, _, filtered_means, _, _, _, A, C, alpha = params
        smoothed_means, smoothed_covariances = self.kalman_filter.smooth(a_sample, params)



        ######################
        ######## ELBO ########
        ######################

        #### VAE - part
        if self.train_VAE:
            #### q_{phi} (a|x)
            log_q_a_given_x = a_dist.log_prob(a_sample.view(-1, self.dim_a)).view(batch_size, seq_len, self.dim_a)
            if upscale_vae_loss:
                log_q_a_given_x = log_q_a_given_x.sum(-1).sum(-1).sum()
            elif use_mean:
                log_q_a_given_x = log_q_a_given_x.sum(-1).mean(1).mean(0)
            else:
                log_q_a_given_x = log_q_a_given_x.sum(-1).sum(-1).mean()
            

            #### p_{theta} (x|a)
            if self.use_bernoulli:
                if upscale_vae_loss:
                    log_p_x_given_a = x_dist.log_prob(x.reshape(-1, *x.shape[2:])).view(batch_size, seq_len, -1).sum(-1).sum(-1).sum()
                elif use_mean:
                    log_p_x_given_a = x_dist.log_prob(x.reshape(-1, *x.shape[2:])).view(batch_size, seq_len, -1).sum(-1).mean(1).mean(0)
                else:
                    log_p_x_given_a = x_dist.log_prob(x.reshape(-1, *x.shape[2:])).view(batch_size, seq_len, -1).sum(-1).sum(-1).mean()
            else:
                if upscale_vae_loss:
                    log_p_x_given_a = log_likelihood(x.view(batch_size, seq_len, -1), 
                                                    x_mean.view(batch_size, seq_len, -1), 
                                                    var=self.x_var, 
                                                    device=self.device).sum(-1).sum(-1).sum()
                elif use_mean:
                    log_p_x_given_a = log_likelihood(x.view(batch_size, seq_len, -1), 
                                                    x_mean.view(batch_size, seq_len, -1), 
                                                    var=self.x_var, 
                                                    device=self.device).mean(1).mean(0).sum()
                else:
                    log_p_x_given_a = log_likelihood(x.view(batch_size, seq_len, -1), 
                                                    x_mean.view(batch_size, seq_len, -1), 
                                                    var=self.x_var, 
                                                    device=self.device).sum(-1).sum(-1).mean()
        
        #### LGSSM - part
        #### p_{gamma} (z|a)
        p_z_given_a = MultivariateNormal(loc=torch.stack(smoothed_means).permute(1,0,2), 
                                         covariance_matrix=(torch.stack(smoothed_covariances)).permute(1,0,2,3))
        z_sample = p_z_given_a.rsample().view(batch_size, seq_len, -1)

        #### p_{gamma} (a|z)
        a_transition = torch.matmul(C, z_sample.unsqueeze(-1)).squeeze(-1)
        p_a_given_z = MultivariateNormal(loc=a_transition.view(-1, self.dim_a), 
                                         covariance_matrix=self.kalman_filter.R)
        if use_mean:
            log_p_a_given_z = p_a_given_z.log_prob(a_sample.view(-1, self.dim_a)).view(batch_size, seq_len).mean(1).mean()
        else:
            log_p_a_given_z = p_a_given_z.log_prob(a_sample).sum(1).mean()

        #### p_{gamma} (z_T|z_T-1, .., z_0) 
        z_transition_mean = torch.cat(
            [
                self.kalman_filter.mu.repeat(batch_size, 1, 1),
                torch.matmul(A[:, 1:, :], z_sample[:, :-1, :].unsqueeze(-1)).squeeze(-1)
            ], 
            dim=1
        )
        z_transition_cov = torch.cat(
            [
                self.kalman_filter.sigma.repeat(batch_size, 1, 1, 1), 
                self.kalman_filter.Q.repeat(batch_size, seq_len - 1, 1, 1)
            ], 
            dim=1
        )
        p_zT_given_zt = MultivariateNormal(loc=z_transition_mean, 
                                           covariance_matrix=z_transition_cov)

        if use_mean:
            log_p_zT_given_zt = p_zT_given_zt.log_prob(z_sample).mean(1).mean()
        else:
            log_p_zT_given_zt = p_zT_given_zt.log_prob(z_sample).sum(1).mean()

        #### p_{gamma} (z|a)
        if use_mean:
            log_p_z_given_a = p_z_given_a.log_prob(z_sample).mean(1).mean()
        else:
            log_p_z_given_a = p_z_given_a.log_prob(z_sample).sum(1).mean()
        
        # create loss dictionary
        if self.train_VAE:
            loss_dict = {'reconstruction loss': self.recon_scale*log_p_x_given_a.detach().cpu().numpy(),
                        'encoder loss': log_q_a_given_x.detach().cpu().numpy(), 
                        'LGSSM observation log likelihood': log_p_a_given_z.detach().cpu().numpy(),
                        'LGSSM tranisition log likelihood': log_p_zT_given_zt.detach().cpu().numpy(),
                        'LGSSM tranisition log posterior': log_p_z_given_a.detach().cpu().numpy()}

            return [a_sample, filtered_means, smoothed_means, C], alpha, -self.recon_scale*log_p_x_given_a + log_q_a_given_x + log_p_z_given_a - log_p_a_given_z - log_p_zT_given_zt, loss_dict
        else:
            loss_dict = {'LGSSM observation log likelihood': log_p_a_given_z.detach().cpu().numpy(),
                        'LGSSM tranisition log likelihood': log_p_zT_given_zt.detach().cpu().numpy(), 
                        'LGSSM tranisition log posterior': log_p_z_given_a.detach().cpu().numpy()}
            return alpha, log_p_z_given_a - log_p_a_given_z - log_p_zT_given_zt, loss_dict
        
    def impute(self, x, mask, sample=False):

        # get dims
        batch_size = x.size(0)
        seq_len = x.size(1)

        # get encoded observations for visualization purposes
        gt_sample = self.encoder(x.view(-1, *x.shape[2:])).mean.detach().view(batch_size, seq_len, self.dim_a)
        
        # convert mask to torch tensor
        mask_t = torch.Tensor(mask).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(batch_size, 1, 1, x.size(3), x.size(4)).to(self.device)
        
        # mask input
        x_masked = mask_t*x
        x_masked = x_masked.to(x.dtype)

        # feed masked sample in encoder
        a_dist = self.encoder(x_masked.view(-1, *x.shape[2:]))

        # sample from q_{phi} (a|x)
        if sample:
            a_sample = a_dist.rsample().view(batch_size, seq_len, self.dim_a)
        else:
            a_sample = a_dist.mean.view(batch_size, seq_len, self.dim_a)

        # estimate observation from filtered posterior
        for t, mask_el in enumerate(mask):
            if mask_el != 0:
                continue
            else:
                # get filtered distribution up to t=t-1
                _, _, _, _, next_means, _, _, C, alpha = self.kalman_filter.filter(a_sample, imputation_idx=t)

                # get predicted observation 
                a_sample[:, t, :] = torch.matmul(C[:, t, :, :], torch.stack(next_means).permute(1,0,2)[:, t, :].unsqueeze(-1)).squeeze(-1)

        # filter estimated sequence (to include last 4 frames, or any observed frame that might be in the sequence)
        params = self.kalman_filter.filter(a_sample)
        _, _, filtered_means, _, _, _, _, C, alpha = params
        
        # get smoothed observations
        smoothed_means, _ = self.kalman_filter.smooth(a_sample, params=params)
        smoothed_obs = torch.matmul(C, torch.stack(smoothed_means).permute(1,0,2).unsqueeze(-1)).squeeze(1)

        # get filtered observations  
        filtered_obs = torch.matmul(C, torch.stack(filtered_means).permute(1,0,2).unsqueeze(-1)).squeeze(1)
        
        # decode smoothed observations
        if self.use_bernoulli:
            x_dist_smooth = self.decoder(smoothed_obs.view(batch_size*seq_len, -1))
            imputed_smoothed_data = x_dist_smooth.mean.view(batch_size, seq_len, *x.shape[2:]).detach()

            x_dist_filtered = self.decoder(filtered_obs.view(batch_size*seq_len, -1))
            imputed_filtered_data = x_dist_filtered.mean.view(batch_size, seq_len, *x.shape[2:]).detach()
        else:
            imputed_smoothed_data, _ = self.decoder(smoothed_obs.view(batch_size*seq_len, -1)).view(batch_size, seq_len, *x.shape[2:]).detach()

        return imputed_smoothed_data, imputed_filtered_data, alpha.detach(), filtered_obs.detach(), smoothed_obs.detach(), gt_sample

    def generate(self, x, mask, sample=False, full_alpha=False, use_gt_weights=False):  

        # get dims
        batch_size = x.size(0)
        seq_len = x.size(1)
        if x.size(1) > self.T:
            seq_len = self.T

        # get ground-truth observations
        gt_obs = self.encoder(x.view(-1, *x.shape[2:])).mean.view(batch_size, x.size(1), self.dim_a).detach()

        # analysis purpose
        if use_gt_weights:
            print('Using Ground Truth Weights')
            gt_a = gt_obs
        else:
            gt_a = None

        # get masked input
        if len(mask) <= self.T:
            mask_t = torch.Tensor(mask).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(batch_size, 1, 1, x.size(3), x.size(4)).to(self.device)
            x_masked = mask_t*x
            x_masked = x_masked.to(x.dtype)
        else:
            x_masked = x[:, :self.T, :]

        # feed masked sample in encoder
        a_dist = self.encoder(x_masked.reshape(-1, *x_masked.shape[2:]))

        # sample from q_{phi} (a|x)
        if sample:
            a_sample = a_dist.rsample().view(batch_size, seq_len, self.dim_a)
        else:
            a_sample = a_dist.mean.view(batch_size, seq_len, self.dim_a)
        
        # create observations to be generated
        if len(mask) > self.T:      
            to_cat = torch.zeros(batch_size, len(mask) - self.T, self.dim_a).to(a_sample.device)
            a_sample = torch.cat([a_sample, to_cat], dim=1)

        # estimate observation from filtered posterior
        for t, mask_el in enumerate(mask):
            if mask_el != 0:
                continue
            else:
                if t>=self.T:
                    end_idx = t+1
                    C_idx = -1
                    imputation_idx = None
                    if full_alpha:
                        start_idx = 0
                    else:
                        start_idx = t-self.T + 1
                        assert (a_sample[:, start_idx:end_idx, :].size(1) == self.T)

                else:
                    start_idx = 0
                    end_idx = self.T
                    C_idx = t
                    imputation_idx = t
                
                # get filtered distribution up to t=t-1
                _, _, _, _, next_means, _, _, C, alpha = self.kalman_filter.filter(a_sample[:, start_idx:end_idx, :], 
                                                                                   imputation_idx=imputation_idx, 
                                                                                   gt_a=gt_a[:, start_idx:end_idx, :])
                a_sample[:, t, :] = torch.matmul(C[:, C_idx, :, :], \
                                                  torch.stack(next_means).permute(1,0,2)[:, C_idx, :].unsqueeze(-1)).squeeze(-1)

        # decode predicted observations
        if self.use_bernoulli:
            x_dist = self.decoder(a_sample.view(batch_size*len(mask), -1))
            generated_data = x_dist.mean.view(batch_size, len(mask), *x.shape[2:]).detach()
        else:
            generated_data, _ = self.decoder(a_sample.view(batch_size*len(mask), -1))

        return generated_data, a_sample.detach(), gt_obs, alpha.detach()
    
    