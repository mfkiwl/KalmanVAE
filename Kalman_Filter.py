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
            num_layers = 5

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
                 R_init=0.01,
                 Q_init=0.01,
                 mu_init=0,
                 sigma_init=1,
                 K=3,
                 T=1,
                 use_KVAE=True, 
                 use_MLP=True):

        super(Kalman_Filter, self).__init__()
        
        self.dim_z = dim_z
        self.dim_a = dim_a
        self.dim_u = dim_u
        self.K = K
        self.T = T
        self.use_KVAE = use_KVAE
        self.use_MLP = use_MLP

        if isinstance(A_init, int):
            self.A = torch.eye(self.dim_z)
        else:
            self.A = A_init
        if self.dim_u > 0:
            if isinstance(B_init, int):
                self.B = torch.eye(self.dim_z, self.dim_u)
            else:
                self.B = B_init
        if isinstance(C_init, int):
            self.C = torch.eye(self.dim_a, self.dim_z)
        else:
            self.C = C_init

        if self.use_KVAE:
            self.dyn_net = Dynamics_Network(self.dim_a, 128, self.K, use_MLP=self.use_MLP)
            self.A = nn.Parameter(self.A.unsqueeze(0).unsqueeze(1).repeat(self.K, T, 1, 1))
            if self.dim_u > 0:
                self.B = nn.Parameter(self.B.unsqueeze(0).unsqueeze(1).repeat(self.K, T, 1, 1))
            self.C = nn.Parameter(self.C.unsqueeze(0).unsqueeze(1).repeat(self.K, T, 1, 1))
            self.a_0 = nn.Parameter(torch.zeros(self.dim_a))

        self.R = R_init*torch.eye(self.dim_a)
        self.Q = Q_init*torch.eye(self.dim_z)

        if isinstance(mu_init, int):
            self.mu = torch.zeros(self.dim_z)
        else:
            self.mu = mu_init
        if isinstance(sigma_init, int):
            self.sigma = torch.eye(self.dim_z)
        else:
            self.sigma = sigma_init

    def filter(self, a, train_dyn_net=True, imputation_idx=None, device=None):

        '''
        This method carries out Kalman filtering based 
        on the observation vector a=(batch-size, L, dim_a).
        '''

        # get batch size, sequence length
        bs, sequence_len = a.size(0), a.size(1)        

        # define mean and covariance for initial state z_0 = N(0, I)
        mu = self.mu.unsqueeze(0).repeat(bs, 1) # (bs, dim_z)
        sigma = self.sigma.unsqueeze(0).repeat(bs, 1, 1) # (bs, dim_z, dim_z)

        if device is not None:
            mu = mu.to(device)
            sigma = sigma.to(device)

        # adjust A,B,C depending on whether we are using KVAE or not
        if self.use_KVAE:
            A = self.A.unsqueeze(0).repeat(bs, 1, 1, 1, 1) # (bs, seq_len, K, dim_z, dim_z)
            if self.dim_u > 0:
                B = self.B.unsqueeze(0).repeat(bs, 1, 1, 1, 1) # (bs, seq_len, K, dim_z, dim_u)
            C = self.C.unsqueeze(0).repeat(bs, 1, 1, 1, 1) # (bs, seq_len, K, dim_a, dim_z)
            a_0 = self.a_0.unsqueeze(0).unsqueeze(1).repeat(bs, 1, 1)
        else:
            A = self.A.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_z, dim_z)
            if self.dim_u > 0:
                B = self.B.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_z, dim_u)
            C = self.C.unsqueeze(0).unsqueeze(1).repeat(bs, sequence_len, 1, 1) # (bs, seq_len, dim_a, dim_z)

        # collect means and covariances for smoothing i.e mu_{t|t} --> E(z_t|y_1:t)
        means = [] # [(bs, dim_z). ..., (bs, dim_z)]
        covariances = [] # [(bs, dim_z, dim_z). ..., (bs, dim_z, dim_z)]

        # collect estimated means and covariances i.e mu_{t|t+1} --> E(z_t+1|y_1:t) 
        # i.e. multiply mean by transitional matrix A
        next_means = [] # [(bs, dim_z). ..., (bs, dim_z)]
        next_covariances = [] # [(bs, dim_z, dim_z). ..., (bs, dim_z, dim_z)]

        # compute mixture of A and C in case we are use Kalman filter in KVAE
        if self.use_KVAE:

            # get alpha
            code_and_obs = torch.cat([a_0, a], dim=1)

            if imputation_idx is not None:
                code_and_obs = code_and_obs[:, :imputation_idx, :]
            else:
                code_and_obs = code_and_obs[:, :-1, :]

            if train_dyn_net:
                alpha = self.dyn_net(code_and_obs) # (bs, L, K)
            else:
                alpha = self.dyn_net(code_and_obs).detach() # (bs, L, K)

            if imputation_idx is not None:
                to_concat = torch.zeros(bs, self.T-imputation_idx, self.K).to(device)
                alpha = torch.cat([alpha, to_concat], dim=1)

            # get mixture of As and Cs
            A = torch.einsum('blk,bklij->blij', alpha, A)
            C = torch.einsum('blk,bklij->blij', alpha, C)
        
        if device is not None:
            self.R = self.R.to(device)
            self.Q = self.Q.to(device)

        # iterate through the length of the sequence
        for t_step in range(sequence_len):

            # account for initialization states
            if t_step == 0:
                #mu_pred = torch.matmul(A[:, t_step, :, :], mu.unsqueeze(2)).squeeze(2)
                #sigma_pred = torch.matmul(torch.matmul(A[:, t_step, :, :], sigma), torch.transpose(A[:, t_step, :, :],1,2)) + self.Q
                mu_pred = mu
                sigma_pred = sigma
            
            # collect predicted mean and predicted covariance
            next_means.append(mu_pred)
            next_covariances.append(sigma_pred)
            
            # get predicted observation
            a_pred = torch.matmul(C[:, t_step, :, :], mu_pred.unsqueeze(2)).squeeze(2)

            # get residual
            r = a[:, t_step, :] - a_pred

            # get Kalman gain
            S = torch.matmul(torch.matmul(C[:, t_step, :, :], sigma_pred), torch.transpose(C[:, t_step, :, :],1,2)) + self.R
            S_inv = torch.linalg.inv(S)
            K = torch.matmul(torch.matmul(sigma_pred, torch.transpose(C[:, t_step, :, :],1,2)), S_inv)   

            # update mean
            mu = mu_pred + torch.matmul(K, r.unsqueeze(2)).squeeze(2)
            
            # update covariance
            KC = torch.matmul(K, C[:, t_step, :, :])
            I = torch.eye(self.dim_z).repeat(bs, 1, 1)
            if device is not None:
                I = I.to(device)
            sigma = torch.matmul((I - KC), sigma_pred)

            # get predicted mean and covariances
            if t_step != sequence_len -1:
                mu_pred = torch.matmul(A[:, t_step+1, :, :], mu.unsqueeze(2)).squeeze(2)
                sigma_pred = torch.matmul(torch.matmul(A[:, t_step+1, :, :], sigma), torch.transpose(A[:, t_step+1, :, :],1,2)) + self.Q
        
            # collect mean and covariance
            means.append(mu)
            covariances.append(sigma)
        
        return mu, sigma, means, covariances, next_means, next_covariances, A, C, alpha

    def smooth(self, a, params):

        '''
        This method carries out Kalman smoothing based 
        on the state z_T obtained via Kalman filtering
        on the observation vector a. Note that smoothing
        does not require to access a and it is initialized
        from the filtered mean and covariance at time t=T
        '''

        bs, sequence_len = a.size(0), a.size(1)

        # get filtered mean and covariances for initialization of Kalman smoother
        _, _, filtered_means, filtered_covariances, next_means, next_covariances, A, _, _ = params

        # collect smoothed means and covariance
        means = [filtered_means[-1]]
        covariances = [filtered_covariances[-1]]

        # iterate through the sequence in reverse order and start from penultimate item
        for t_step_reversed in reversed(range(sequence_len-1)):
            
            # get backwards Kalman Gain
            J = torch.matmul(torch.transpose(A[:, t_step_reversed+1, :, :], 1,2), torch.linalg.inv(next_covariances[t_step_reversed+1]))
            J = torch.matmul(filtered_covariances[t_step_reversed], J)

            # get smoothed mean and covariance
            mu_t_T = filtered_means[t_step_reversed] + torch.matmul(J, (means[0] - next_means[t_step_reversed+1]).unsqueeze(2)).squeeze(2)
            sigma_t_T = filtered_covariances[t_step_reversed] + torch.matmul(torch.matmul(J, covariances[0] - next_covariances[t_step_reversed+1]), torch.transpose(J, 1,2))
            
            means.insert(0, mu_t_T)
            covariances.insert(0, sigma_t_T)
        
        return means, covariances
            