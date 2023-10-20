import torch
import matplotlib.pyplot as plt 
import torch.distributions as Distributions

from Kalman_Filter import Kalman_Filter

# test torch.distribution sizes
dist = Distributions.Normal(loc=torch.randn(64,50,1,16,16), scale=torch.ones(64,50,1,16,16))
print(dist.log_prob(torch.randn(16)).size())

# Test Kalman Filter with random data
dim_z = 2
dim_a = dim_z

sequence_len = 8
batch_size = 16

a = torch.randn(batch_size, sequence_len, dim_z)
kalman_filter = Kalman_Filter(dim_z, 
                              dim_a, 
                              dim_u=0, 
                              use_KVAE=False)

params = kalman_filter.filter(a)
means, covariances = kalman_filter.smooth(a, params)

# Test Kalman Filter with synthetic data
# create ground truth data
T = 15
start_Y = 10
end_Y = 9
step_Y = (start_Y - end_Y)/T

gt_seq_X = [x+1 for x in range(10, 10+T)]

gt_seq_Y = [start_Y]
for y in range(T-2):
    gt_seq_Y.append(gt_seq_Y[y] - step_Y)
gt_seq_Y.append(end_Y)

# create observed data
observations_Y = [11.8, 10.2, 9.7, 9.85, 8.5, 10.1, 9.6, 10.8, 8., 8.3, 9.3, 8.2, 10.1, 9.9, 6.5]
observations_X = [9., 11.8, 12.2, 12.3, 12.8, 15.3, 15.8, 17.7, 18.1, 19.5, 20.1, 20.6, 22.2, 23., 24.]
observations_seq = []
for obs_idx in range(T):
    observations_seq.append([observations_X[obs_idx], observations_Y[obs_idx]])
observations_seq = torch.tensor(observations_seq).unsqueeze(0)

# initialize Kalman filter
dim_z = 4
dim_a = observations_seq.size(2)
delta = 0.5
A_init = torch.FloatTensor([[1, 0, delta, 0], [0, 1, 0, delta], [0, 0, 1, 0], [0, 0, 0, 1]])
C_init = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0]])
R_init = 0.25
Q_init = 0.05
mu_init = torch.FloatTensor([8, 10, 1, 0])
sigma_init = 0.5*torch.eye(dim_z)
kalman_filter = Kalman_Filter(dim_z, 
                              dim_a, 
                              dim_u=0, 
                              A_init=A_init, 
                              C_init=C_init, 
                              R_init=R_init, 
                              Q_init=Q_init, 
                              mu_init=mu_init,
                              sigma_init=sigma_init,
                              use_KVAE=False)

# filter observations
mu, sigma, filtered_means, filtered_covariances, next_means, next_covariances, gamma = kalman_filter.filter(observations_seq)
filtered_X = []
filtered_Y = []
for _, loc in enumerate(filtered_means):
    filtered_X.append(loc[0, 0].numpy())
    filtered_Y.append(loc[0, 1].numpy())

# smooth observations
params = [mu, sigma, filtered_means, filtered_covariances, next_means, next_covariances, gamma]
smoothed_means, smoothed_covariances = kalman_filter.smooth(observations_seq, params)

# check for positive definite covariances in filtering and smoothing
print(torch.linalg.cholesky(torch.cat(filtered_covariances)) is not None)
print(torch.linalg.cholesky(torch.cat(smoothed_covariances)) is not None)

smoothed_X = []
smoothed_Y = []
for _, loc in enumerate(smoothed_means):
    print(loc[0, 0:2])
    smoothed_X.append(loc[0, 0].numpy())
    smoothed_Y.append(loc[0, 1].numpy())


# plot all figures together
fig_sum, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
fig_sum.suptitle('Kalman Filtering and Smoothing')
ax1.scatter(gt_seq_X, gt_seq_Y) 
ax1.scatter(observations_X, observations_Y, label='Observed')
ax1.plot(gt_seq_X, gt_seq_Y, label='Ground-Truth')
ax1.set_ylim([3,15])
ax1.legend(loc='upper right')
ax1.title.set_text('Observations and Ground Truth')

ax2.scatter(filtered_X, filtered_Y) 
ax2.scatter(observations_X, observations_Y, label='Observed')
ax2.plot(filtered_X, filtered_Y, label='Filtered')
ax2.set_ylim([3,15])
ax2.legend(loc='upper right')
ax2.title.set_text('Kalman Filtering')

ax3.scatter(smoothed_X, smoothed_Y) 
ax3.scatter(observations_X, observations_Y, label='Observed')
ax3.plot(smoothed_X, smoothed_Y, label='Smoothed')
ax3.set_ylim([3,15])
ax3.legend(loc='upper right')
ax3.title.set_text('Kalman Smoothing')

fig_sum.savefig('/data2/users/lr4617/Kalman_VAE/results/KalmanFilter/kalman_filter_all.png')


# plot ground truth data + observations
fig = plt.figure()
plt.scatter(gt_seq_X, gt_seq_Y) 
plt.scatter(observations_X, observations_Y, label='Observed')
plt.plot(gt_seq_X, gt_seq_Y, label='Ground-Truth')
plt.ylim([3,15])
plt.show()
fig.savefig('/data2/users/lr4617/Kalman_VAE/results/KalmanFilter/gt.png')


# plot filtered data + observation
fig_2 = plt.figure()
plt.scatter(filtered_X, filtered_Y) 
plt.scatter(observations_X, observations_Y, label='Observed')
plt.plot(filtered_X, filtered_Y, label='Filtered')
plt.ylim([3,15])
plt.legend()
plt.show()
fig_2.savefig('/data2/users/lr4617/Kalman_VAE/results/KalmanFilter/filtered.png')


# plot smoothed data + observation
fig_2 = plt.figure()
plt.scatter(smoothed_X, smoothed_Y) 
plt.scatter(observations_X, observations_Y, label='Observed')
plt.plot(smoothed_X, smoothed_Y, label='Smoothed')
plt.ylim([3,15])
plt.legend()
plt.show()
fig_2.savefig('/data2/users/lr4617/Kalman_VAE/results/KalmanFilter/smoothed.png')

