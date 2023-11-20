import torch
import torch.nn as nn
import torch.nn.functional as F

class Dynamics_Network(nn.Module):

    def __init__(self,
                 dim_a,
                 dim_hidden,
                 K,
                 num_layers=2, 
                 use_MLP=True):
        
        super(Dynamics_Network, self).__init__()

        self.dim_a = dim_a
        self.dim_hidden = dim_hidden
        self.K = K
        self.num_layers = num_layers
        self.use_MLP = use_MLP
        
        if self.use_MLP:
            dim = self.dim_hidden
            num_layers = 2
        else: 
            dim = self.K
            num_layers = 2

        self.lstm = nn.LSTM(self.dim_a, 
                            dim, 
                            self.num_layers, 
                            batch_first=True)
        if self.use_MLP:
            self.linear = nn.Linear(self.dim_hidden, self.K)

    def forward(self, input):
        input, _ = self.lstm(input)
        if self.use_MLP:
            input = self.linear(input)
        input = F.softmax(input, dim=-1)

        return input
    

class Kalman_Filter(nn.Module):

    def __init__(self, 
                 dim_z, 
                 dim_a, 
                 dim_u=0,
                 A_init=1,
                 B_init=1,
                 C_init=1,
                 R_init=1.,
                 Q_init=1.,
                 mu_init=0,
                 sigma_init=1,
                 K=3,
                 T=1,
                 use_KVAE=True, 
                 use_MLP=True, 
                 init_weight_A=1, 
                 init_weight_C=0,
                 mlp_dim=128, 
                 symmetric_covariance=True,
                 dtype=torch.float32, 
                 device=-1):

        super(Kalman_Filter, self).__init__()
        
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_u = dim_u
        self.K = K
        self.T = T
        self.use_KVAE = use_KVAE
        self.use_MLP = use_MLP
        self.symmetric_covariance = symmetric_covariance
        self.device = device
        self.A_init = A_init
        
        # initialize state transition matrix
        if isinstance(A_init, int) or isinstance(A_init, float):
            self.A = nn.Parameter((1-init_weight_A)*torch.randn(self.K, self.dim_z, self.dim_z)
                                   + init_weight_A*torch.eye(self.dim_z))
        else:
            self.A = A_init
        
        # initialize observation transition matrix
        if isinstance(C_init, int) or isinstance(C_init, float):
            self.C = nn.Parameter((1-init_weight_C)*torch.randn(self.K, self.dim_a, self.dim_z) 
                                   + init_weight_C*torch.eye(self.dim_a, self.dim_z))
        else:
            self.C = C_init
        
        # initialize matrix for additional input (if required)
        if self.dim_u > 0:
            if isinstance(B_init, int):
                self.B = nn.Parameter((1-init_weight_A)*torch.randn(self.K, self.dim_z, self.dim_u)
                                       + init_weight_A*torch.eye(self.dim_z, self.dim_u))
            else:
                self.B = B_init
        
        # initialize start code and Dynamics Network (if KVAE is used - else make parameters non-trainable)
        if self.use_KVAE:
            self.a_0 = nn.Parameter(torch.zeros(self.dim_a))
            self.dyn_net = Dynamics_Network(dim_a=self.dim_a, 
                                            dim_hidden=mlp_dim, 
                                            K=self.K, 
                                            use_MLP=self.use_MLP)
        else:
            self.A = self.A.detach().to(dtype).to(self.device)
            self.C = self.C.detach().to(dtype).to(self.device)
            if self.dim_u > 0:
                self.B = self.B.detach().to(dtype).to(self.device)

        # initialize noise variables
        self.R = R_init*torch.eye(self.dim_a).to(dtype).to(self.device)
        self.Q = Q_init*torch.eye(self.dim_z).to(dtype).to(self.device)

        # initialize start state
        if isinstance(mu_init, int):
            self.mu = torch.zeros(self.dim_z).to(dtype).to(self.device)
        else:
            self.mu = mu_init

        # initialize start covariance
        if isinstance(sigma_init, int):
            self.sigma = torch.eye(self.dim_z).to(dtype).to(self.device)
        else:
            self.sigma = sigma_init

    def filter(self, 
               a, 
               train_dyn_net=None, 
               imputation_idx=None):

        '''
        This method carries out Kalman filtering based 
        on the observation vector a=(batch-size, L, dim_a).
        '''

        # get batch size, sequence length
        bs = a.size(0)
        sequence_len = a.size(1)  

        # handle situation where we generate starting from 50 frames
        if sequence_len > self.T:
            sequence_len = self.T      

        # define mean and covariance for initial state z_0 = N(0, I)
        mu = self.mu.unsqueeze(0).repeat(bs, 1) # (bs, dim_z)
        sigma = self.sigma.unsqueeze(0).repeat(bs, 1, 1) # (bs, dim_z, dim_z)

        # adjust A,B,C depending on whether we are using KVAE or not           
        if not self.use_KVAE:
            A = self.A.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_z, dim_z)
            if self.dim_u > 0:
                B = self.B.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_z, dim_u)
            C = self.C.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_a, dim_z)

        # collect means and covariances for smoothing i.e mu_{t|t} --> E(z_t|y_1:t)
        means = [] # [(bs, dim_z), ..., (bs, dim_z)]
        covariances = [] # [(bs, dim_z, dim_z), ..., (bs, dim_z, dim_z)]

        # collect estimated means and covariances i.e mu_{t|t+1} --> E(z_t+1|y_1:t) 
        # i.e. multiply mean by transitional matrix A
        next_means = [] # [(bs, dim_z). ..., (bs, dim_z)]
        next_covariances = [] # [(bs, dim_z, dim_z). ..., (bs, dim_z, dim_z)]

        # compute mixture of A and C in case we are use Kalman filter in KVAE
        if self.use_KVAE:
            A, C, alpha = self.compute_transition_matrices(a, train_dyn_net, imputation_idx)
        
        # initialize predicted mean and variance
        mu_pred = mu
        sigma_pred = sigma

        # iterate through the length of the sequence

        for t_step in range(sequence_len):
            
            # collect predicted mean and predicted covariance
            next_means.append(mu_pred)
            next_covariances.append(sigma_pred)
            
            # get predicted observation
            a_pred = torch.matmul(C[:, t_step, :, :], mu_pred.unsqueeze(2)).squeeze(2)

            if imputation_idx is not None:
                if t_step == imputation_idx: 
                    break

            # get residual
            if a.size(1) > self.T:                    
                r = a[:, t_step + (a.size(1) - self.T), :] - a_pred
                if t_step ==  sequence_len -1:
                    print(a[:, t_step + (a.size(1) - self.T), :])
            else:
                r = a[:, t_step, :] - a_pred

            # get Kalman gain
            S = torch.matmul(torch.matmul(C[:, t_step, :, :], sigma_pred), torch.transpose(C[:, t_step, :, :],1,2)) + self.R
            S_inv = torch.inverse(S)
            K = torch.matmul(torch.matmul(sigma_pred, torch.transpose(C[:, t_step, :, :],1,2)), S_inv)   

            # update mean
            mu = mu_pred + torch.matmul(K, r.unsqueeze(2)).squeeze(2)
            
            # update covariance
            KC = torch.matmul(K, C[:, t_step, :, :])
            I = torch.eye(self.dim_z).repeat(bs, 1, 1)
            if self.device != -1:
                I = I.to(self.device)
            sigma = torch.matmul((I - KC), sigma_pred)

            if self.symmetric_covariance:
                sigma = (sigma + sigma.transpose(1, 2))/2.0

            # get predicted mean and covariances
            if t_step != sequence_len -1:
                mu_pred = torch.matmul(A[:, t_step+1, :, :], mu.unsqueeze(2)).squeeze(2)
                sigma_pred = torch.matmul(torch.matmul(A[:, t_step+1, :, :], sigma), torch.transpose(A[:, t_step+1, :, :],1,2)) + self.Q
                
                if self.symmetric_covariance:
                    sigma_pred = (sigma_pred + sigma_pred.transpose(1, 2))/2.0

            # collect mean and covariance
            means.append(mu)
            covariances.append(sigma)
        
        if self.use_KVAE:
            return mu, sigma, means, covariances, next_means, next_covariances, A, C, alpha
        else:
            return mu, sigma, means, covariances, next_means, next_covariances, A, C

    def smooth(self, a, params):

        '''
        This method carries out Kalman smoothing based 
        on the state z_T obtained via Kalman filtering
        on the observation vector a. Note that smoothing
        does not require to access a and it is initialized
        from the filtered mean and covariance at time t=T
        '''

        sequence_len = a.size(1)

        # get filtered mean and covariances for initialization of Kalman smoother
        if self.use_KVAE:
            _, _, filtered_means, filtered_covariances, next_means, next_covariances, A, _, _ = params
        else:
             _, _, filtered_means, filtered_covariances, next_means, next_covariances, A, _ = params

        # collect smoothed means and covariance
        means = [filtered_means[-1]]
        covariances = [filtered_covariances[-1]]

        # iterate through the sequence in reverse order and start from penultimate item
        for t in reversed(range(sequence_len-1)):

            # get backwards Kalman Gain
            J = torch.matmul(filtered_covariances[t], torch.matmul(torch.transpose(A[:, t+1, :, :], 1,2), \
                             torch.inverse(next_covariances[t+1])))
            
            # get smoothed mean and covariance
            mu_t_T = filtered_means[t] + \
                     torch.matmul(J, (means[0] - next_means[t+1]).unsqueeze(-1)).squeeze(-1)
            sigma_t_T = filtered_covariances[t] + \
                     torch.matmul(torch.matmul(J, covariances[0] - next_covariances[t+1]), torch.transpose(J, 1, 2))
            
            if self.symmetric_covariance:
                sigma_t_T = (sigma_t_T + sigma_t_T.transpose(1,2))/2.0

            means.insert(0, mu_t_T)
            covariances.insert(0, sigma_t_T)
        
        return means, covariances
            
    def compute_transition_matrices(self, a, train_dyn_net=False, imputation_idx=None):
        
        batch_size = a.size(0)
        a_0 = self.a_0.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1)

        code_and_obs = torch.cat([a_0, a], dim=1) # [(bs, 1, dim_a), (bs, T, dim_a)

        if imputation_idx is not None:
            code_and_obs = code_and_obs[:, :imputation_idx+1, :]
        else:
            code_and_obs = code_and_obs[:, :-1, :]

        if train_dyn_net:
            alpha = self.dyn_net(code_and_obs) # (bs, L, K)
        else:
            alpha = self.dyn_net(code_and_obs).detach() # (bs, L, K)

        if imputation_idx is not None:
            to_concat = torch.zeros(batch_size, self.T-imputation_idx-1, self.K).to(self.device)
            alpha = torch.cat([alpha, to_concat], dim=1) 

        
        # get mixture of As and Cs
        A = torch.einsum('blk,kij->blij', alpha[:, a.size(1)-self.T:, :], self.A)
        C = torch.einsum('blk,kij->blij', alpha[:, a.size(1)-self.T:, :], self.C)

        return A, C, alpha