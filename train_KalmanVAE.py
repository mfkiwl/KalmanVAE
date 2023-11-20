import torch
import torch.nn as nn
import time
import wandb
import matplotlib.pyplot as plt 
import random
import numpy as np
import cv2


from Kalman_VAE import KalmanVAE
from datetime import datetime

from PIL import Image

from dataloaders.bouncing_data import BouncingBallDataLoader
from torch.utils.data import DataLoader
from moviepy.editor import ImageSequenceClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def train(train_loader, kvae, optimizer, train_dyn_net, args, dtype, upscale_vae_loss, use_mean, output_folder):

    loss_epoch = 0.
    idv_losses = {'reconstruction loss': 0,
                  'encoder loss': 0, 
                  'LGSSM observation log likelihood': 0,
                  'LGSSM tranisition log likelihood': 0, 
                  'LGSSM tranisition log posterior': 0}
    
    kvae.train()

    for n, sample in enumerate(train_loader, 1):
        
        optimizer.zero_grad()

        sample = sample > 0.5
        sample = sample.to(dtype).to('cuda:' + str(args.device))

        obs, alpha, loss, loss_dict = kvae.calculate_loss(sample, 
                                                          train_dyn_net=train_dyn_net, 
                                                          upscale_vae_loss=upscale_vae_loss, 
                                                          use_mean=use_mean)

        if n == 1:
            a_sample, filtered_means, smoothed_means, C = obs
            a_sample = a_sample.detach()
            C = C.detach()

            filtered_obs = torch.matmul(C, torch.stack(filtered_means).permute(1,0,2).unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            smoothed_obs = torch.matmul(C, torch.stack(smoothed_means).permute(1,0,2).unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            gt_obs = a_sample.view(sample.size(0), sample.size(1), -1).cpu().numpy()

            for i in range(5):
                fig = plt.figure()
                plt.plot(gt_obs[i, :, 0], gt_obs[i, :, 1], ".-", color='black', label='Ground-Truth')
                plt.plot(filtered_obs[i, :, 0], filtered_obs[i, :, 1], ".-", color='blue', label='Filtered')
                plt.plot(smoothed_obs[i, :, 0], smoothed_obs[i, :, 1], ".-", color='orange', label='Smoothed')
                plt.legend()
                plt.grid()
                fig.savefig(os.path.join(output_folder, 'trajectory_{}.png'.format(i)))

        loss.backward()
        optimizer.step()

        loss_epoch += loss

        for key in idv_losses.keys():
            idv_losses[key] += loss_dict[key]

        alphas = alpha.detach().cpu()

    for key in idv_losses.keys():
        idv_losses[key] = idv_losses[key]/len(train_loader)
    
    return loss_epoch/len(train_loader), idv_losses, alphas

def test_reconstruction(test_loader, kvae, output_folder, args, dtype, visualize=False):

    kvae.eval()
    with torch.no_grad():
        mse_error = 0
        for i, sample in enumerate(test_loader, 1):
            
            # prepare sample
            sample = sample > 0.5
            sample = sample.to(dtype).to('cuda:' + str(args.device))

            # get reconstruction
            x_hat = kvae.calculate_loss(sample, recon_only=True)

            # get mean-squared-error
            mse = nn.MSELoss()
            mse_error += mse(x_hat, sample)

            # visualize difference between sample and reconstruction
            if visualize and i == 1:
                
                # revert sample to showable format
                sample = sample.cpu().numpy()

                for sample_num in range(5):
                    fig, axs = plt.subplots(2, 6, figsize=(15, 8))
                    fig.suptitle('Reconstruction-Orginal Comparison')
                    for j, t in enumerate(range(0, args.T, 9)):
                        axs[0, j].title.set_text('Ground-Truth, t={}'.format(str(t)))
                        axs[0, j].imshow(sample[sample_num, t, 0, :, :]*255, cmap='gray', vmin=0, vmax=255)
                        axs[1, j].title.set_text('Reconstructions, t={}'.format(str(t)))
                        pred_to_plot = x_hat[sample_num, t, 0, :, :]*255
                        axs[1, j].imshow(pred_to_plot.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    
                    fig.savefig(output_folder + '/reconstruction_{}'.format(str(sample_num+1)))

        recon_error = mse_error/len(test_loader)
        print('Reconstruction Mean-Squared-Error: ', recon_error)

        return recon_error

def test_imputation(test_loader, kvae, mask, output_folder, args, dtype, visualize=False):
    
    kvae.eval()
    with torch.no_grad():
        mse_error_smoothed = 0
        mse_error_fitered = 0
        for i, sample in enumerate(test_loader, 1):
            
            # prepare sample
            sample = sample > 0.5
            sample = sample.to(dtype).to('cuda:' + str(args.device))

            # get imputations
            imputed_seq, imputed_filtered_seq, _, _, _, _ = kvae.impute(sample, mask)

            # get mean-squared-error for smoothed frames
            mse = nn.MSELoss()
            mse_error_smoothed += mse(imputed_seq, sample)

            # get mean-squared-error for filtered frames
            mse = nn.MSELoss()
            mse_error_fitered += mse(imputed_filtered_seq, sample)

            # visualize difference between sample and imputation
            if visualize and i == 1:

                zeros_idxs_in_mask = [i for i in range(len(mask)) if mask[i] == 0.]
                if args.masking_fraction > 0.25:
                    n_of_images_to_show = 12
                    idxs_to_show = np.sort(np.random.choice(range(0, len(zeros_idxs_in_mask)), size=n_of_images_to_show, replace=False))
                else:
                    n_of_images_to_show = len(zeros_idxs_in_mask)
                    idxs_to_show = [n for n in range(n_of_images_to_show)]

                # revert sample to showable format
                sample = sample.cpu().numpy()

                for sample_num in range(20):
                    fig, axs = plt.subplots(2, n_of_images_to_show, figsize=(8, 3))
                    fig.suptitle('Ground-Truth - Imputation Comparison ({}%)'.format(int(args.masking_fraction*100)))
                    for j, idx in enumerate(idxs_to_show):
                        t = zeros_idxs_in_mask[idx]
                        axs[0, j].set_title('GT, t={}'.format(str(t)), fontsize=7)
                        axs[0, j].imshow(sample[sample_num, t, 0, :, :]*255, cmap='gray', vmin=0, vmax=255)
                        axs[1, j].set_title('IM, t={}'.format(str(t)), fontsize=7)
                        pred_to_plot = imputed_seq[sample_num, t, 0, :, :]*255
                        axs[1, j].imshow(pred_to_plot.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
                        axs[0, j].grid(False)
                        axs[0, j].set_xticks([])
                        axs[0, j].set_yticks([])
                        axs[1, j].grid(False)
                        axs[1, j].set_xticks([])
                        axs[1, j].set_yticks([])

                    fig.savefig(output_folder + '/Imputation{}'.format(str(sample_num+1)))
        
        smoothing_error = mse_error_smoothed/len(test_loader)
        filtering_error = mse_error_fitered/len(test_loader)

        print('Imputation SMOOTHED Mean-Squared-Error: ', mse_error_smoothed/len(test_loader))
        print('Imputation FILTERED Mean-Squared-Error: ', mse_error_fitered/len(test_loader))

    return smoothing_error, filtering_error

def test_generation(test_loader, mask, kvae, output_folder, args, dtype, full_alpha=False, visualize=False):
    
    kvae.eval()
    with torch.no_grad():
        if args.n_of_starting_frames < args.T:
            mse_error = 0
            for i, sample in enumerate(test_loader, 1):
                
                # prepare sample
                sample = sample > 0.5
                sample = sample.to(dtype).to('cuda:' + str(args.device))

                # get imputations
                generated_seq, _, _ = kvae.generate(sample, mask)
                mse = nn.MSELoss()
                mse_error += mse(generated_seq, sample)

                # visualize difference between sample and generation
                if visualize and i == 1:
                    # get samples to show
                    zeros_idxs_in_mask = [i for i in range(len(mask)) if mask[i] == 0.]
                    n_of_images_to_show = 12
                    idxs_to_show = np.arange(0, n_of_images_to_show)

                    # revert sample to showable format
                    sample = sample.cpu().numpy()

                    for sample_num in range(20):
                        fig, axs = plt.subplots(2, n_of_images_to_show, figsize=(8, 3))
                        fig.suptitle('Ground-Truth - Generation Comparison')
                        for j, idx in enumerate(idxs_to_show):
                            t = zeros_idxs_in_mask[idx]
                            axs[0, j].set_title('GT, t={}'.format(str(t)), fontsize=7)
                            axs[0, j].imshow(sample[sample_num, t, 0, :, :]*255, cmap='gray', vmin=0, vmax=255)
                            axs[1, j].set_title('GN t={}'.format(str(t)), fontsize=7)
                            pred_to_plot = generated_seq[sample_num, t, 0, :, :]*255
                            axs[1, j].imshow(pred_to_plot.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
                            axs[0, j].grid(False)
                            axs[0, j].set_xticks([])
                            axs[0, j].set_yticks([])
                            axs[1, j].grid(False)
                            axs[1, j].set_xticks([])
                            axs[1, j].set_yticks([])

                        fig.savefig(output_folder + '/Generation{}'.format(str(sample_num+1)))

            print('Generation Mean-Squared-Error: ', mse_error/len(test_loader))

        else:
            distances = []
            for i, sample in enumerate(test_loader, 1):
                
                # get generated sequence
                sample = sample > 0.5
                sample = sample.to(dtype).to('cuda:' + str(args.device))
                generated_seq, generated_obs, _ = kvae.generate(sample, mask, full_alpha=full_alpha)

                # compute distance between consecutive datapoints in observation space
                a = generated_obs[:, 1:, :]
                b = generated_obs[:, :-1, :]

                assert a.size(1) == b.size(1)
                distance = torch.sqrt(torch.square(a-b).sum(dim=-1))
                distances.append(distance)

            distances_to_plot = torch.cat(distances, dim=0).cpu()

            fig = plt.figure()
            plt.errorbar(np.arange(len(mask)-1), distances_to_plot.mean(0).cpu(), yerr=distances_to_plot.var(0).cpu(), color="r")
            plt.plot(np.arange(len(mask)-1), distances_to_plot.mean(0).cpu())
            plt.xlabel('Time')
            plt.ylabel('COnsecutive Distance')
            plt.title('Consecutive Points Distance - Generation')
    
            fig.savefig(os.path.join(output_folder, 'distance.png'))

def plot(test_dl, 
         kvae, 
         mask, 
         output_folder, 
         args, 
         which, 
         dtype, 
         single_plots=False, 
         training=False, 
         n_samples_to_plot=5, 
         device=0, 
         full_alpha=False,
         return_paths=False):
    
    if single_plots:
        # show individual sequences, trajectories, and weights 
        
        # get root dir
        root_dir = os.path.join(output_folder, 'trajectories', '')
        if not os.path.isdir:
            os.mkdir(root_dir)
        
        # collect video paths if training=True
        if return_paths:
            video_paths = []

        # set kvae to eval mode
        kvae.eval()

        with torch.no_grad():
            print(mask)
            for sample_n in range(40, n_samples_to_plot + 40):

                # get sample
                sample = test_dl[sample_n]
                sample = sample > 0.5
                batched_sample = torch.Tensor(sample).unsqueeze(0).to(dtype).to('cuda:{}'.format(args.device))

                # create directories
                png_files = []
                sample_dir = os.path.join(root_dir, 'sample_{}'.format(sample_n), '')
                if not os.path.isdir(sample_dir):
                    os.makedirs(sample_dir)
                if which == 'imputation':
                    frames_dir = os.path.join(sample_dir, '', 'filt_vs_smooth')
                else:
                    frames_dir = os.path.join(sample_dir, '', 'generated_frames')
                if not os.path.isdir(frames_dir):
                    os.makedirs(frames_dir)
                path_out = os.path.join(sample_dir, 'trajectories.mp4')

                # settings for imputations
                if which == 'imputation':
                    # get imputations
                    imputed_seq, imputed_filtered_seq, alpha, filtered_obs, smoothed_obs, gt_smoothed_obs = kvae.impute(batched_sample, mask)
                    filtered_obs = filtered_obs.squeeze(-1).cpu()
                    smoothed_obs = smoothed_obs.squeeze(-1).cpu()
                    gt_smoothed_obs = gt_smoothed_obs.squeeze(-1).cpu()
                    alpha = alpha.detach().cpu()
                
                    # get max and min values for visualization
                    f_max = filtered_obs.max()
                    f_min = filtered_obs.min()
                    s_max = smoothed_obs.max()
                    s_min = smoothed_obs.min()
                    gt_max = gt_smoothed_obs.max()
                    gt_min= gt_smoothed_obs.min()
                    max_obs_vals = max([f_max, s_max, gt_max])
                    min_obs_vals = min([f_min, s_min, gt_min])

                    # prepare frames for visualization
                    batched_sample = batched_sample < 0.5
                    batched_sample = batched_sample.float()
                    imputed_seq = imputed_seq < 0.5
                    imputed_seq = imputed_seq.float()
                    imputed_filtered_seq = imputed_filtered_seq < 0.5
                    imputed_filtered_seq = imputed_filtered_seq.float()

                else:
                    # get generations
                    generated_seq, generated_obs, alpha = kvae.generate(batched_sample, mask, full_alpha=full_alpha)
                    generated_seq = generated_seq.cpu()
                    generated_obs = generated_obs.squeeze(-1).cpu()
                    alpha = alpha.detach().cpu()

                    print(alpha.size())

                    # get max and min values for visualization
                    max_obs_vals = generated_obs.max()
                    min_obs_vals = generated_obs.min()

                    # prepare frames for visualization
                    generated_seq = generated_seq < 0.5
                    generated_seq = generated_seq.float()

                # create video without saving all frames if training=True
                if training:
                    video_paths.append(path_out)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4")
                    frame_size = (1600, 400)
                    fps = 10
                    video = cv2.VideoWriter(path_out, fourcc, fps, frame_size)
                
                # for visualization purposes
                cmap = plt.get_cmap("tab10")
                if which == 'imputation':
                    length = batched_sample.size(1)
                else:
                    length = len(mask)

                for t in range(length):
                    
                    if which == 'imputation':
                        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
                    else:
                        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

                    if which == 'imputation':
                        fig.suptitle(f"$t = {t}$")
                    else:
                        if t < args.T:
                            fig.suptitle(f"$t = {t}$ (Ground Truth)")
                        else:
                            fig.suptitle(f"$t = {t}$ (Generation)", color='red')

                    # create frames for SMOOTHING + FILTERING
                    if which == 'imputation':
                        bg_ch = batched_sample[0, t, 0].cpu().unsqueeze(-1).repeat(1, 1, 2)
                        r_ch = torch.ones(batched_sample[0, t, 0].size(0), batched_sample[0, t, 0].size(0), 1)
                        gt_to_show = torch.cat([r_ch, bg_ch], dim=2)
                        im_s_to_show = imputed_seq[0, t, 0].unsqueeze(-1).repeat(1, 1, 3).float().detach().cpu().numpy()
                        im_f_to_show = imputed_filtered_seq[0, t, 0].unsqueeze(-1).repeat(1, 1, 3).float().detach().cpu().numpy()

                        axes[0].imshow(gt_to_show, vmin=0, vmax=1)
                        axes[0].imshow(im_s_to_show, vmin=0, vmax=1, alpha=0.5)
                        axes[0].set_adjustable('box') 
                        axes[0].set_title(r"imputation smoothed $\mathbf{x}_t$")

                        axes[1].imshow(gt_to_show, vmin=0, vmax=1)
                        axes[1].imshow(im_f_to_show, vmin=0, vmax=1, alpha=0.5)
                        axes[1].set_adjustable('box') 
                        axes[1].set_title(r"imputation filtered $\mathbf{x}_t$")

                        axes[2].bar(["0", "1", "2"], alpha[0, t, :].detach().cpu().numpy())
                        axes[2].set_ylim(0, 1)
                        axes[2].set_title(r"weight $\mathbf{k}_t$")

                        pos_img = axes[0].get_position()
                        pos_bar = axes[2].get_position()
                        axes[2].set_position([pos_bar.x0, pos_img.y0, pos_bar.width, pos_img.height])

                        axes[3].plot(filtered_obs[0, 0:t+1, 0], filtered_obs[0, 0:t+1, 1], ".-", color=cmap(0), label='Filtered')
                        axes[3].plot(smoothed_obs[0, 0:t+1, 0], smoothed_obs[0, 0:t+1, 1],".-", color=cmap(1), label='Smoothed')
                        axes[3].plot(gt_smoothed_obs[0, 0:t+1, 0], gt_smoothed_obs[0, 0:t+1, 1], ".-", color="black", label='Ground-Truth')
                        axes[3].set_xlim([min_obs_vals, max_obs_vals])
                        axes[3].set_ylim([min_obs_vals, max_obs_vals])
                        axes[3].legend()
                        axes[3].grid()
                    
                    else:
                        if t < args.T:
                            gen_to_show = generated_seq[0, t, 0].unsqueeze(-1).repeat(1, 1, 3)
                        else:
                            bg_ch = generated_seq[0, t, 0].cpu().unsqueeze(-1).repeat(1, 1, 2)
                            r_ch = torch.ones(generated_seq[0, t, 0].size(0), generated_seq[0, t, 0].size(0), 1)
                            gen_to_show = torch.cat([r_ch, bg_ch], dim=2)

                        axes[0].imshow(gen_to_show, vmin=0, vmax=1)
                        axes[0].set_adjustable('box') 
                        axes[0].set_title(r"generated $\mathbf{x}_t$")

                        axes[1].bar(["0", "1", "2"], alpha[0, t, :].detach().cpu().numpy())
                        axes[1].set_ylim(0, 1)
                        axes[1].set_title(r"weight $\mathbf{k}_t$")
                        pos_img = axes[0].get_position()
                        pos_bar = axes[1].get_position()
                        axes[1].set_position([pos_bar.x0, pos_img.y0, pos_bar.width, pos_img.height])

                        if t >= args.T:
                            if t == args.T:
                                print(generated_obs[0, args.T:t+1, 0].size())
                            color = 'red'
                            axes[2].plot(generated_obs[0, 0:args.T, 0], generated_obs[0, 0:args.T, 1], color='black')
                            axes[2].plot(generated_obs[0, args.T - 1:t+1, 0], generated_obs[0, args.T - 1:t+1, 1], color=color)
                        else:
                            color = 'black'
                            axes[2].plot(generated_obs[0, 0:t+1, 0], generated_obs[0, 0:t+1, 1], color=color)
                        axes[2].set_xlim([min_obs_vals, max_obs_vals])
                        axes[2].set_ylim([min_obs_vals, max_obs_vals])
                        axes[2].grid()
                    
                    plt.tight_layout()

                    if training:
                        canvas = FigureCanvas(fig)
                        canvas.draw()
                        matplotlib_image = np.array(canvas.renderer.buffer_rgba())
                        opencv_image = cv2.cvtColor(matplotlib_image, cv2.COLOR_RGBA2BGR)
                        video.write(opencv_image)
                    
                    else:
                        fig.savefig(os.path.join(frames_dir, 'visualization-{}.png'.format(t)))
                        png_files.append(os.path.join(frames_dir, 'visualization-{}.png'.format(t)))

                    plt.close(fig)

                if training:
                    video.release()

                else:
                    clip = ImageSequenceClip(png_files, fps=10)
                    clip.write_videofile(path_out, codec="libx264")
            
            if return_paths:
                return video_paths

    else:
        # show 16 sequences in the same plot    
        root_dir = os.path.join(output_folder, 'batch_visualizations', '')
        if not os.path.isdir:
            os.mkdir(root_dir)
        if which == 'imputation':
            name_joint = 'joint_gt_imputations'
        else:
            name_joint = 'joint_gt_generations'
        joint_dir = os.path.join(root_dir, '', name_joint)
        if not os.path.isdir(joint_dir):
            os.makedirs(joint_dir)
        
        kvae.eval()
        with torch.no_grad():

            imgs_to_show = 16
            batched_sample = torch.Tensor(test_dl).to('cuda:{}'.format(args.device))
            batched_sample = batched_sample > 0.5
            batched_sample = batched_sample.to(dtype)

            if which == 'imputation':
                x_hat, _ = kvae.impute(batched_sample[:imgs_to_show], mask)
            else:
                x_hat = kvae.generate(batched_sample[:imgs_to_show], mask)
            
            png_files = []

            for step in range(batched_sample.size(1)):
                
                image = batched_sample[:, step, :, :, :].squeeze(1).cpu()
                reconstruction = x_hat[:, step, :, :, :].detach().squeeze(1).cpu()

                fig_joint, axes_joint = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
                if args.consecutive_masking:
                    fig_joint.suptitle("Masking {}% - {}".format(int(args.masking_fraction*100), "Consecutive"))
                else:
                    fig_joint.suptitle("Masking {}% - {}".format(int(args.masking_fraction*100), "Random"))

                image = image < 0.5
                reconstruction = reconstruction < 0.5

                for sample_n in range(imgs_to_show):

                    row = int(sample_n/4)
                    col = sample_n%4

                    bg_ch = image[sample_n].unsqueeze(-1).repeat(1, 1, 2)
                    r_ch = torch.ones(image[sample_n].size(0), image[sample_n].size(0), 1)
                    img_to_show = torch.cat([r_ch, bg_ch], dim=2)
                    recon_to_show = reconstruction[sample_n].unsqueeze(-1).repeat(1, 1, 3).float().numpy()

                    axes_joint[row, col].imshow(img_to_show, vmin=0, vmax=1)
                    axes_joint[row, col].set_adjustable('box')
                    axes_joint[row, col].imshow(recon_to_show, vmin=0, vmax=1, alpha=0.5)

                fig_joint.savefig(os.path.join(joint_dir, 'join_gt_im_{}.png'.format(step)))

                png_files.append(os.path.join(joint_dir, 'join_gt_im_{}.png'.format(step)))

                plt.close()
            
            path_out = os.path.join(root_dir, 'trajectories.mp4')
            clip = ImageSequenceClip(png_files, fps=10)
            clip.write_videofile(path_out, codec="libx264")

def get_mask(masking_fraction, args, which='imputation', consecutive=True):
    if which == 'imputation':
        if isinstance(masking_fraction, int):
            masking_fractions = [masking_fraction]
        else:
            masking_fractions = masking_fraction

        mask_list = []
        for mask_frac in masking_fractions:
            mask = [1] * args.T
            n_of_samples_to_mask = int((args.T - 8)*mask_frac)
            if args.consecutive_masking:
                to_zero = np.arange(int(args.T/2) - int(n_of_samples_to_mask/2), int(args.T/2) + int(n_of_samples_to_mask/2), 1)
            else:
                to_zero = random.sample(range(4, args.T-4), n_of_samples_to_mask)
            for mask_idx in range(len(mask)):
                if mask_idx in to_zero:
                    mask[mask_idx] = 0

            mask_list.append(mask)

    if len(mask_list) == 1:
        return mask_list[0]
    else:
        return mask_list

def main(args):

    # set values for A and C matrices initialization
    if args.tune_hyperparams:
        if args.C_init == 0.:
            args.A_init = 1.
        elif args.C_init == 0.1 or args.C_init == 0.9:
            args.A_init = 0.9
    
    print('########################################')
    print('MODEL PATH', args.kvae_model)
    print('TRAIN: ', args.train)
    print('TEST: ', args.test)
    print('USE_MLP: ', args.use_MLP)
    print('A_init: ', args.A_init)
    print('C_init: ', args.C_init)
    print('R_init: ', args.R_noise_init)
    print('Q_init: ', args.Q_noise_init)
    print('########################################')

    # load data
    train_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'train')
    test_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'test')
    train_dl = BouncingBallDataLoader(train_dir, images=True)
    test_dl = BouncingBallDataLoader(test_dir, images=True)
    train_loader = DataLoader(train_dl, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dl, batch_size=args.batch_size, shuffle=True)

    # choose data format
    if args.use_double:
        dtype = torch.float64
    else:
        dtype = torch.float32

    # get image size
    it = iter(train_loader)
    first = next(it)
    _, T, n_channels_in, dim, dim = first.size()
    args.T = T

    # load model
    kvae = KalmanVAE(n_channels_in,
                     dim,
                     args.dim_a, 
                     args.dim_z, 
                     args.K, 
                     T=T, 
                     A_init=args.A_init, 
                     C_init=args.C_init,
                     recon_scale=args.recon_scale,
                     use_bernoulli=args.use_bernoulli,
                     use_MLP=args.use_MLP,
                     symmetric_covariance=args.symmetric_covariance,
                     R_noise_init=args.R_noise_init,
                     Q_noise_init=args.Q_noise_init,
                     dtype=dtype, 
                     device='cuda:' + str(args.device)).to('cuda:' + str(args.device)).to(dtype=dtype)
    
    # if already trained, load checkpoints
    if args.kvae_model is not None:
        kvae.load_state_dict(torch.load(args.kvae_model))

    if args.train:

        # define optimizer
        optimizer = torch.optim.Adam(kvae.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma_lr_schedule)

        # helper variables
        start = time.time()
        log_list = []        
        
        # set parameters for wandb
        if args.use_wandb:

            # define filename
            now = datetime.now()
            run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
            save_filename = args.output_folder + '/{}'.format(args.dataset) + '/{}'.format(run_name) 
            if not os.path.isdir(save_filename):
                os.makedirs(save_filename)

            # create specific filename when tuning hyperparams
            if args.tune_hyperparams:
                str_noise = str(args.R_noise_init).replace('.', '') # since R_init = Q_init always
                str_A = str(args.A_init).replace('.', '')
                str_C = str(args.C_init).replace('.', '')
                param_dir_name = "noise_{}_A_{}_C{}".format(str_noise, str_A, str_C)
                save_filename = os.path.join(args.output_folder,'', 
                                             args.dataset, '', 
                                             'tune_hyperparams', '', 
                                             param_dir_name, '', 
                                             run_name)
                if not os.path.isdir(save_filename):
                    os.makedirs(save_filename)

            # initialize wandb run
            run = wandb.init(project="KalmanVAE", 
                             config={"dataset" : args.dataset,
                                    "T": args.T,
                                    "use-double": args.use_double, 
                                    "batch-size" : args.batch_size,
                                    "iterations" : args.num_epochs,
                                    "n_epoch_initial": args.n_epoch_initial,
                                    "learning-rate" : args.lr, 
                                    "learning-rate-scheduler": args.lr_scheduler, 
                                    "gamma": args.gamma_lr_schedule,
                                    "a-dimensions": args.dim_a, 
                                    "z-dimensions": args.dim_z, 
                                    "A_init": args.A_init, 
                                    "C_init": args.C_init,
                                    "R_init": args.R_noise_init, 
                                    "Q_init": args.Q_noise_init,
                                    "number-of-dynamics-K": args.K,
                                    "use-MLP": args.use_MLP,
                                    "reconstruction-weight": args.recon_scale,
                                    "use_bernoulli": args.use_bernoulli,
                                    "upscale_vae_loss": args.upscale_vae_loss, 
                                    "use_mean": args.use_mean, 
                                    "symmetric-covariance": args.symmetric_covariance,
                                    "tune-hyperparams": args.tune_hyperparams,
                                    },
                            name=run_name)
    
        # define number of epochs when NOT to train Dynamic Parameter Network
        train_dyn_net = False
        for epoch in range(args.num_epochs):
            
            # delay training of Dynamics Parameter Network to epoch = n_epoch_initial
            if epoch >= args.n_epoch_initial:
                train_dyn_net = True

            # training
            output_folder = os.path.join(save_filename, '', 'training', '', 'trajectories', '', 'epoch_{}'.format(str(epoch)))
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            loss_train, loss_dict, _ = train(train_loader=train_loader, 
                                             kvae=kvae, 
                                             optimizer=optimizer, 
                                             train_dyn_net=train_dyn_net, 
                                             args=args, 
                                             dtype=dtype, 
                                             upscale_vae_loss=args.upscale_vae_loss, 
                                             use_mean=args.use_mean, 
                                             output_folder=output_folder)
            if args.use_wandb:
                run.log(loss_dict)
            if epoch % 20 == 0 and epoch > 0:
                scheduler.step()
            end = time.time()
            log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
            start = end
            print(log)
            log_list.append(log + '\n')

            # validation  
            output_folder = os.path.join(save_filename, '', 'validation', '', 'reconstructions', '', 'epoch_{}'.format(str(epoch)))
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            _ = test_reconstruction(test_loader=test_loader, 
                                    kvae=kvae, 
                                    output_folder=output_folder, 
                                    args=args, 
                                    dtype=dtype)

            # dynamics visualization
            if epoch % 10 == 0 and epoch > 0:
                fractions = [0.3, 0.5, 0.9, 1.0]
                consecutive_masking = True
                for masking_fraction in fractions:
                    mask = [1] * T
                    n_of_samples_to_mask = int((T - 8)*masking_fraction)
                    if consecutive_masking:
                        to_zero = np.arange(int(T/2) - int(n_of_samples_to_mask/2), int(T/2) + int(n_of_samples_to_mask/2), 1)
                    else:
                        to_zero = random.sample(range(4, T-4), n_of_samples_to_mask)
                    for mask_idx in range(len(mask)):
                        if mask_idx in to_zero:
                            mask[mask_idx] = 0
                    output_folder = os.path.join(save_filename, '', 'validation', '', 'dynamics', '', 'epoch_{}'.format(str(epoch)), 'mask_{}'.format(int(masking_fraction*100)))
                    if consecutive_masking:
                        output_folder = os.path.join(output_folder, '', 'consecutive')
                    else:
                        output_folder = os.path.join(output_folder, '', 'random')
                    if not os.path.isdir(output_folder):
                        os.makedirs(output_folder)
                    print('Plotting Imputation ...')
                    video_paths = plot(test_dl=test_dl, 
                                    kvae=kvae, 
                                    mask=mask, 
                                    output_folder=output_folder, 
                                    args=args, 
                                    which='imputation',
                                    dtype=dtype, 
                                    single_plots=True,
                                    training=False, 
                                    n_samples_to_plot=2, 
                                    device=args.device, 
                                    return_paths=True)
                    
                    if len(video_paths) > 0:
                        for n, video_path in enumerate(video_paths):
                            video = wandb.Video(
                            video_path,
                            f"idx_{n}_mask_length_{n_of_samples_to_mask}",
                            fps=10,
                            format="avi",
                            )

                            video_log = {"video": video,
                                        "batch_id": 0,
                                        "data_idx": n,
                                        "mask_length": n_of_samples_to_mask}

                            wandb.log(video_log)

            # save checkpoints 
            if epoch % 10 == 0 or epoch == args.num_epochs-1:
                with open(save_filename + '/kvae' + str(epoch+1) + '.pt', 'wb') as f:
                    torch.save(kvae.state_dict(), f)
            
            # save training log
            with open(save_filename + '/training.cklog', "a+") as log_file:
                log_file.writelines(log_list)
                log_list.clear()
    
    if args.test:

        #### get filename
        if not args.train:
            path_to_dir = args.kvae_model.split('/')[:-1]
            save_filename = "/".join(path_to_dir) 
        

        ################################
        ######## RECONSTRUCTION ########
        ################################
        if args.test_reconstruction:
            output_folder = os.path.join(save_filename, '', 'reconstructions')
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            recon_error = test_reconstruction(test_loader=test_loader, 
                                              kvae=kvae, 
                                              output_folder=output_folder, 
                                              args=args, 
                                              dtype=dtype)
            print("Reconstruction error: ", recon_error)
            run.log({'reconstruction error': recon_error})
        
        
        ############################
        ######## IMPUTATION ########
        ############################
        mask = [1] * T
        n_of_samples_to_mask = int((T - 8)*args.masking_fraction)
        if args.consecutive_masking:
            to_zero = np.arange(int(T/2) - int(n_of_samples_to_mask/2), int(T/2) + int(n_of_samples_to_mask/2), 1)
        else:
            to_zero = random.sample(range(4, T-4), n_of_samples_to_mask)
        for mask_idx in range(len(mask)):
            if mask_idx in to_zero:
                mask[mask_idx] = 0

        if args.test_imputation:
            output_folder = os.path.join(save_filename, '', 'imputations_{}'.format(str(int(args.masking_fraction*100))))
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            print('Testing Imputation ...')
            s_error, f_error = test_imputation(test_loader=test_loader, 
                                               kvae=kvae, 
                                               mask=mask, 
                                               output_folder=output_folder, 
                                               args=args, 
                                               dtype=dtype)
            print('Smoothing error: ', s_error)
            print('Filtering eroor: ', f_error)
            run.log({'smoothing error': s_error})
            run.log({'filtering error': f_error})
        
        if args.plot_imputation:
            output_folder = os.path.join(save_filename, '', 'dyn_analysis', '', 'imputations', '', 'mask_{}'.format(int(args.masking_fraction*100)))
            if args.consecutive_masking:
                output_folder = os.path.join(output_folder, '', 'consecutive')
            else:
                output_folder = os.path.join(output_folder, '', 'random')
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            print('Plotting Imputation ...')
            plot(test_dl=test_dl, 
                 kvae=kvae, 
                 mask=mask, 
                 output_folder=output_folder, 
                 args=args, 
                 which='imputation',
                 dtype=dtype, 
                 single_plots=args.plot_trajectories)
        

        ############################
        ######## GENERATION ########
        ############################
        mask = [1] * (args.n_of_starting_frames + args.n_of_frame_to_generate)
        to_zero = np.arange(args.n_of_starting_frames, args.n_of_starting_frames + args.n_of_frame_to_generate)
        for mask_idx in range(len(mask)):
            if mask_idx in to_zero:
                mask[mask_idx] = 0

        if args.test_generation:
            if args.n_of_starting_frames == args.T:
                output_folder = os.path.join(save_filename, '', 'generations', '', 'long_term')
            else:
                output_folder = os.path.join(save_filename, '', 'generations', '', 'gt_comparison')
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
                
            print('Testing Generation ...')
            test_generation(test_loader=test_loader, 
                            mask=mask, 
                            kvae=kvae, 
                            output_folder=output_folder, 
                            args=args, 
                            dtype=dtype, 
                            full_alpha=args.full_alpha)
        
        if args.plot_generation:
            output_folder = os.path.join(save_filename, '', 'dyn_analysis', '', 'generations')
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            print('Plotting Generation ...')
            plot(test_dl=test_dl, 
                 kvae=kvae, 
                 mask=mask, 
                 output_folder=output_folder, 
                 args=args, 
                 which='generation',
                 dtype=dtype, 
                 single_plots=args.plot_trajectories, 
                 n_samples_to_plot=10, 
                 device=args.device, 
                 full_alpha=args.full_alpha)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp
   
    parser = argparse.ArgumentParser(description='Kalman VAE')

    # data parameters
    parser.add_argument('--dataset', type=str, default='Bouncing_Ball',
        help='dataset used')
    parser.add_argument('--n_channels_in', type=int, default=1,
        help='number of color channels in the data')
    parser.add_argument('--train_with_masking', type=bool, default=False, 
        help='training with masked sequences')
    parser.add_argument('--use_double', type=int, default=1, 
        help='train with dtype=float64')
    
    # initialization parameters
    parser.add_argument('--A_init', type=float, default=1.,
        help='initialization weight for random component')
    parser.add_argument('--C_init', type=float, default=0.,
        help='initialization weight for random component')
    parser.add_argument('--R_noise_init', type=float, default=1., 
        help='amount of noise to add to the state transitions')
    parser.add_argument('--Q_noise_init', type=float, default=1., 
        help='amount of noise to add to the observation transitions')

    # encoder parameters
    parser.add_argument('--dim_a', type=int, default=2,
        help='dimensionality of encoded vector a')
    parser.add_argument('--dim_z', type=int, default=4,
        help='dimensionality of encoded vector z')
    parser.add_argument('--dim_u', type=int, default=0,
        help='dimensionality of encoded vector u')
    parser.add_argument('--K', type=int, default=3,
        help='number of LGSSMs to be mixed')
    parser.add_argument('--use_MLP', type=int, default=1,
        help='use MLP head in Dynamics Parameter Network')
    parser.add_argument('--T', type=int, default=50,
        help='number of timestep in the dataset')

    # training parameters
    parser.add_argument('--train', type=int, default=None,
        help='train model')
    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
        help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--use_grad_clipping', type=bool, default=False,
        help='use gradient clipping')
    parser.add_argument('--lr_scheduler', type=str, default='Exponential', 
        help='type of learning rate scheduler to use')
    parser.add_argument('--gamma_lr_schedule', type=float, default=0.85,
        help="learning rate decay multiplicative factor")
    parser.add_argument('--recon_scale', type=float, default=0.3, 
        help="importance given to reconstruction during training")
    parser.add_argument('--use_bernoulli', type=int, default=1,
        help='use bernoulli in the decoder')
    parser.add_argument('--n_epoch_initial', type=int, default=20,
        help='number of epochs to wait for to train dynamics parameter')
    parser.add_argument('--upscale_vae_loss', type=int, default=1, 
        help='decide whether to up-weigh the loss of encoder and decoder')
    parser.add_argument('--use_mean', type=int, default=0, 
        help='decide whether to average over sequences during training')
    parser.add_argument('--symmetric_covariance', type=int, default=1, 
        help='decide whether to symmetrize covariances in Kalman Filter')
    parser.add_argument('--tune_hyperparams', type=int, default=0, 
        help='decide whether to tune A,C,Q,R')

    # testing parameters
    parser.add_argument('--test', type=int, default=None,
        help='test model')
    parser.add_argument('--test_reconstruction', type=int, default=0,
        help='test model reconstruction')
    
    # test imputation
    parser.add_argument('--test_imputation', type=int, default=0,
        help='test model imputation')
    parser.add_argument('--plot_imputation', type=int, default=1,
        help='plot model imputation')
    parser.add_argument('--masking_fraction', type=float, default=0.3, 
        help='fraction fo sample being masked for testing imputation')
    parser.add_argument('--consecutive_masking', type=int, default=0, 
        help='decide whether masking is done on consecutive frames')
    parser.add_argument('--plot_trajectories', type=int, default=1, 
        help='decide whether to plot trajectories or batch_plots')
    
    # test generation
    parser.add_argument('--test_generation', type=int, default=0, 
        help='test model generation')
    parser.add_argument('--plot_generation', type=int, default=0,
        help='plot model generation')
    parser.add_argument('--n_of_starting_frames', type=int, default=50, 
        help='number of sample we want to generate from')
    parser.add_argument('--n_of_frame_to_generate', type=int, default=100, 
        help='number of sample we want to generate')
    parser.add_argument('--full_alpha', type=int,  default=1, 
        help='decide whether to compute alpha weights from whole sequence or last 50 frames')

    # logistics
    parser.add_argument('--datasets_root_dir', type=str, default="/data2/users/lr4617/data/",
        help='path to the root directory of datasets')
    parser.add_argument('--kvae_model', type=str, default=None,
        help='path to the kvae model dictionary')
    parser.add_argument('--output_folder', type=str, default='results',
        help='location to save kave model and results')
    parser.add_argument('--use_wandb', type=int, default=None,
        help='use weights and biases to track expriments')
    parser.add_argument('--device', type=int, default=None,
        help='cuda device to use')
    
    # get arguments
    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('{}'.format(args.output_folder)):
        os.makedirs('{}'.format(args.output_folder))
    
    main(args)


