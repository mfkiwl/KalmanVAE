import torch
import torch.nn as nn
import time
import wandb
import matplotlib.pyplot as plt 
import random
import numpy as np

from Kalman_VAE import KalmanVAE
from datetime import datetime

from PIL import Image

from dataloaders.bouncing_data import BouncingBallDataLoader
from torch.utils.data import DataLoader

def plot_dynamics(alphas, output_folder, n_samples=20):
    for n_sample in range(n_samples):
        sample_dyn = alphas[n_sample, :, :]
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
        bar_colors = ['tab:blue', 'tab:red', 'tab:orange']

        for k in range(args.K):
            axs[k].bar(np.arange(sample_dyn.size(0)), sample_dyn[:, k], color=bar_colors[k])
            axs[k].set_ylabel('K-value')
            axs[k].set_xlabel('Time-Step')

        fig.savefig(output_folder + 'dyn_bars_{}.png'.format(n_sample))

def train(train_loader, kvae, optimizer, train_dyn_net, args):

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
        sample = sample.cuda().float().to('cuda:' + str(args.device))

        _, alpha, loss, loss_dict = kvae.calculate_loss(sample, train_dyn_net)

        loss.backward()
        optimizer.step()

        loss_epoch += loss

        for key in idv_losses.keys():
            idv_losses[key] += loss_dict[key]

        alphas = alpha.cpu()

    for key in idv_losses.keys():
        idv_losses[key] = idv_losses[key]/len(train_loader)
    
    return loss_epoch/len(train_loader), idv_losses, alphas

def test_reconstruction(test_loader, kvae, output_folder, args):

    kvae.eval()
    with torch.no_grad():
        mse_error = 0
        for i, sample in enumerate(test_loader, 1):
            
            sample = sample > 0.5
            sample = sample.cuda().float().to('cuda:' + str(args.device))
            B, T, C, d1, d2 = sample.size()

            # get mean-squared-error on test data
            x_hat = kvae.calculate_loss(sample, recon_only=True)
            x_hat = x_hat.view(B, T, C, d1, d2)
            mse = nn.MSELoss()
            mse_error += mse(x_hat, sample)
            
            # revert sample to showable format
            sample = sample.cpu().numpy()

            # visualize difference between sample and reconstruction
            if i == 1:
                for sample_num in range(5):
                    fig, axs = plt.subplots(2, 6, figsize=(15, 8))
                    fig.suptitle('Reconstruction-Orginal Comparison')
                    for j, t in enumerate(range(0, T, 9)):
                        axs[0, j].title.set_text('Ground-Truth, t={}'.format(str(t)))
                        axs[0, j].imshow(sample[sample_num, t, 0, :, :]*255, cmap='gray', vmin=0, vmax=255)
                        axs[1, j].title.set_text('Reconstructions, t={}'.format(str(t)))
                        pred_to_plot = x_hat[sample_num, t, 0, :, :]*255
                        axs[1, j].imshow(pred_to_plot.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    
                    fig.savefig(output_folder + '/reconstruction_{}'.format(str(sample_num+1)))

        print('Reconstruction Mean-Squared-Error: ', mse_error/len(test_loader))

def test_imputation(test_loader, kvae, mask, output_folder, args):
    
    kvae.eval()
    with torch.no_grad():
        mse_error = 0
        for i, sample in enumerate(test_loader, 1):
            
            sample = sample > 0.5
            sample = sample.cuda().float().to('cuda:' + str(args.device))
            B, T, C, d1, d2 = sample.size()

            # get imputations
            imputed_seq, _ = kvae.impute(sample, mask)
            mse = nn.MSELoss()
            mse_error += mse(imputed_seq, sample)

            # revert sample to showable format
            sample = sample.cpu().numpy()

            zeros_idxs_in_mask = [i for i in range(len(mask)) if mask[i] == 0.]
            if args.masking_fraction > 0.25:
                n_of_images_to_show = 12
                idxs_to_show = np.sort(np.random.choice(range(0, len(zeros_idxs_in_mask)), size=n_of_images_to_show, replace=False))
            else:
                n_of_images_to_show = len(zeros_idxs_in_mask)
                idxs_to_show = [n for n in range(n_of_images_to_show)]

            # visualize difference between sample and reconstruction
            if i == 1:
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
            
            '''
            # visualize masked and neighbouring frames together
            idx_to_plot = [zeros_idxs_in_mask[0]]
            for a, mask_idx in enumerate(zeros_idxs_in_mask):
                if mask_idx - idx_to_plot[-1] > 0:
                    idx_to_plot.append(mask_idx)
                if len(idx_to_plot) == 4:
                    break

            n_of_images_to_show = int(len(idx_to_plot)*3)
            if i == 1:
                for sample_num in range(20):
                    
                    fig_2, axs_2 = plt.subplots(2, n_of_images_to_show, figsize=(8, 3))
                    fig_2.suptitle('Ground-Truth - Imputation Comparison ({}%)'.format(int(args.masking_fraction*100)))
                    
                    for j, idx in enumerate(idx_to_plot):
                        t = idx_to_plot[j]

                        ts = [-1, 0, 1]
                        for k, a in enumerate(range(j*3, (j*3)+3, 1)):

                            axs_2[0, a].set_title('GT, t={}'.format(str(t + ts[k])), fontsize=7)
                            axs_2[0, a].imshow(sample[sample_num, t + ts[k], 0, :, :]*255, cmap='gray', vmin=0, vmax=255)

                            axs_2[1, a].set_title('IM, t={}'.format(str(t + ts[k])), fontsize=7)
                            pred_to_plot = imputed_seq[sample_num, t + ts[k], 0, :, :]*255
                            axs_2[1, a].imshow(pred_to_plot.cpu().numpy(), cmap='gray', vmin=0, vmax=255)

                            axs_2[0, a].grid(False)
                            axs_2[0, a].set_xticks([])
                            axs_2[0, a].set_yticks([])

                            axs_2[1, a].grid(False)
                            axs_2[1, a].set_xticks([])
                            axs_2[1, a].set_yticks([])
                    
                    fig_2.savefig(output_folder + '/Imputation_and_Neighbours{}'.format(str(sample_num+1)))
            '''
        print('Imputation Mean-Squared-Error: ', mse_error/len(test_loader))

def test_generation(test_loader, mask, kvae, output_folder, args):
    kvae.eval()
    with torch.no_grad():

        mse_error = 0
        for i, sample in enumerate(test_loader, 1):
            
            sample = sample > 0.5
            sample = sample.cuda().float().to('cuda:' + str(args.device))

            # get imputations
            generated_seq = kvae.generate(sample, mask)
            mse = nn.MSELoss()
            mse_error += mse(generated_seq, sample)

            # revert sample to showable format
            sample = sample.cpu().numpy()

            # get samples to show
            zeros_idxs_in_mask = [i for i in range(len(mask)) if mask[i] == 0.]
            print(zeros_idxs_in_mask)
            n_of_images_to_show = 12
            idxs_to_show = np.arange(0, n_of_images_to_show)
            
            if i == 1:
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

def plot_imputation(test_dl, kvae, mask, output_folder, args, single_plots=False):

    # show 20 sequences individually
    if single_plots:
        for sample_n in range(20):

            sample = test_dl[sample_n]
            sample = sample > 0.5
            batched_sample = torch.Tensor(sample).unsqueeze(0).to('cuda:0')

            # get imputations
            imputed_seq, alpha = kvae.impute(batched_sample, mask)

            wieghts_dir = os.path.join(output_folder, '', 'sample_{}'.format(sample_n), '', 'weights')
            if not os.path.isdir(wieghts_dir):
                os.makedirs(wieghts_dir)

            for step, (image, reconstruction, weight) in enumerate(zip(batched_sample.squeeze(0).cpu(), imputed_seq.detach().cpu().squeeze(0).numpy(), alpha.squeeze(0))):
                
                image = image > 0.5
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
                fig.suptitle(f"$t = {step}$")

                axes[0].imshow(image[0], vmin=0, vmax=1, cmap="Greys", aspect='equal')
                axes[0].set_adjustable('box') 
                axes[1].imshow(reconstruction[0], vmin=0, vmax=1, cmap="Greys", aspect='equal')
                axes[2].bar(["0", "1", "2"], weight.detach().cpu().numpy())
                axes[2].set_ylim(0, 1)
                axes[0].set_title(r"image $\mathbf{x}_t$")
                axes[1].set_title(r"reconstruction $\hat{\mathbf{x}}_t$")
                axes[2].set_title(r"weight $\mathbf{k}_t$")
                pos_img = axes[0].get_position()
                pos_bar = axes[2].get_position()
                axes[2].set_position([pos_bar.x0, pos_img.y0, pos_bar.width, pos_img.height])
                
                fig.savefig(os.path.join(wieghts_dir, 'weight-{}.png'.format(step)))
                plt.close()
    else:
        # show 16 sequences in the same plot
        name_im = 'batch_imputations'
        imputations_dir = os.path.join(output_folder, '', name_im)
        if not os.path.isdir(imputations_dir):
                os.makedirs(imputations_dir)
        name_gt = 'batch_gt'
        gt_dir = os.path.join(output_folder, '', name_gt)
        if not os.path.isdir(gt_dir):
                os.makedirs(gt_dir)
        joint_dir = os.path.join(output_folder, '', 'joint_gt_imputations')
        if not os.path.isdir(joint_dir):
            os.makedirs(joint_dir)

        imgs_to_show = 16
        batched_sample = torch.Tensor(test_dl).to('cuda:0')
        batched_sample = batched_sample > 0.5
        imputed_seq, _ = kvae.impute(batched_sample[:imgs_to_show], mask)
        
        for step in range(batched_sample.size(1)):
        #for step, (image, reconstruction) in enumerate(zip(, imputed_seq.detach().squeeze(2).cpu().numpy())):
            
            image = batched_sample[:, step, :, :, :].squeeze(1).cpu()
            reconstruction = imputed_seq[:, step, :, :, :].detach().squeeze(1).cpu()

            fig_gt, axes_gt = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
            fig_im, axes_im = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))

            fig_joint, axes_joint = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))


            image = image < 0.5
            reconstruction = reconstruction < 0.5

            for sample_n in range(imgs_to_show):

                row = int(sample_n/4)
                col = sample_n%4

                bg_ch = image[sample_n].unsqueeze(-1).repeat(1, 1, 2)
                r_ch = torch.ones(image[sample_n].size(0), image[sample_n].size(0), 1)
                img_to_show = torch.cat([r_ch, bg_ch], dim=2)
                recon_to_show = reconstruction[sample_n].unsqueeze(-1).repeat(1, 1, 3).float().numpy()

                axes_gt[row, col].imshow(img_to_show, vmin=0, vmax=1, aspect='equal')
                axes_gt[row, col].set_adjustable('box') 
                axes_im[row, col].imshow(reconstruction[sample_n], vmin=0, vmax=1, cmap="Greys", aspect='equal')

                axes_joint[row, col].imshow(img_to_show, vmin=0, vmax=1)
                axes_joint[row, col].set_adjustable('box')
                axes_joint[row, col].imshow(recon_to_show, vmin=0, vmax=1, alpha=0.5)

            fig_im.savefig(os.path.join(imputations_dir, 'batched_imputations_{}.png'.format(step)))
            fig_gt.savefig(os.path.join(gt_dir, 'batched_gt_{}.png'.format(step)))
            fig_joint.savefig(os.path.join(joint_dir, 'join_gt_im_{}.png'.format(step)))

            plt.close()


def main(args):

    print('########################################')
    print('MODEL PATH', args.kvae_model)
    print('TRAIN: ', args.train)
    print('TEST: ', args.test)
    print('########################################')

    # load data
    train_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'train')
    test_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'test')
    train_dl = BouncingBallDataLoader(train_dir, images=True)
    test_dl = BouncingBallDataLoader(test_dir, images=True)
    train_loader = DataLoader(train_dl, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dl, batch_size=args.batch_size, shuffle=True)

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
                     recon_scale=args.recon_scale,
                     use_bernoulli=args.use_bernoulli).cuda().to('cuda:' + str(args.device))
    
    # if already trained, load checkpoints
    if args.kvae_model is not None:
        kvae.load_state_dict(torch.load(args.kvae_model))

    if args.train:
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

            if n_channels_in == 1:
                binary = True
            else:
                binary = False
            run = wandb.init(project="KalmanVAE", 
                            config={"dataset" : args.dataset,
                                    "binary" : binary,
                                    "train-with-masking": args.train_with_masking,
                                    "batch-size" : args.batch_size,
                                    "iterations" : args.num_epochs,
                                    "learning-rate" : args.lr, 
                                    "learning-rate-scheduler": args.lr_scheduler, 
                                    "gamma": args.gamma_lr_schedule,
                                    "grad-clipping": args.use_grad_clipping,
                                    "a-dimensions": args.dim_a, 
                                    "z-dimensions": args.dim_z, 
                                    "number-of-dynamics-K": args.K,
                                    "reconstruction-weight": args.recon_scale},
                            name=run_name)
    
        # define number of epochs when NOT to train Dynamic Parameter Network
        n_epoch_initial = 20
        train_dyn_net = False

        for epoch in range(args.num_epochs):
            
            # delay training of Dynamics Parameter Network to epoch = n_epoch_initial
            if epoch >= n_epoch_initial:
                train_dyn_net = True

            # train 
            loss_train, loss_dict, alphas = train(train_loader, kvae, optimizer, train_dyn_net, args)
            if args.use_wandb:
                run.log(loss_dict)
            if epoch % 20 == 0 and epoch > 0:
                scheduler.step()
            end = time.time()
            log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
            start = end
            print(log)
            log_list.append(log + '\n')

            # valid reconstruction 
            output_folder = os.path.join(save_filename, '', 'validation', '', 'epoch_{}'.format(str(epoch)))
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            test_reconstruction(test_loader, kvae, output_folder, args)

            # save checkpoints 
            if epoch % 10 == 0 or epoch == args.num_epochs-1:
                with open(save_filename + '/kvae' + str(epoch+1) + '.pt', 'wb') as f:
                    torch.save(kvae.state_dict(), f)
            
            # save training log
            with open(save_filename + '/training.cklog', "a+") as log_file:
                log_file.writelines(log_list)
                log_list.clear()
    
    if args.test:
        
        #### GET FILENAME
        if not args.train:
            path_to_dir = args.kvae_model.split('/')[:-1]
            save_filename = "/".join(path_to_dir) 
        
        #### RECONSTRUCTION
        if args.test_reconstruction:
            output_folder = os.path.join(save_filename, '', 'reconstructions')
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            test_reconstruction(test_loader, kvae, output_folder, args)
        
        #### IMPUTATION
        mask = [1] * T
        n_of_samples_to_mask = int((T - 8)*args.masking_fraction)
        to_zero = random.sample(range(4, T-4), n_of_samples_to_mask)
        for mask_idx in range(len(mask)):
            if mask_idx in to_zero:
                mask[mask_idx] = 0
        if args.test_imputation:
            output_folder = os.path.join(save_filename, '', 'imputations_{}'.format(str(int(args.masking_fraction*100))))
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            test_imputation(test_loader, kvae, mask, output_folder, args)
        else:
            output_folder = os.path.join(save_filename, '', 'dyn_analysis', '', 'mask_{}'.format(int(args.masking_fraction*100)))
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            plot_imputation(test_dl, kvae, mask, output_folder, args)
        
        #### GENERATION
        if args.n_of_frame_to_generate > 0:
            mask = [1] * args.n_of_frame_to_generate
            to_zero = np.arange(8, args.n_of_frame_to_generate)
            for mask_idx in range(len(mask)):
                if mask_idx in to_zero:
                    mask[mask_idx] = 0
            output_folder = os.path.join(save_filename, '', 'generations')
            if not os.path.exists('{}'.format(output_folder)):
                os.makedirs(output_folder)
            test_generation(test_loader, mask, kvae, output_folder, args)


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

    # encoder parameters
    parser.add_argument('--dim_a', type=int, default=2,
        help='dimensionality of encoded vector a')
    parser.add_argument('--dim_z', type=int, default=4,
        help='dimensionality of encoded vector z')
    parser.add_argument('--dim_u', type=int, default=0,
        help='dimensionality of encoded vector u')
    parser.add_argument('--K', type=int, default=3,
        help='number of LGSSMs to be mixed')
    parser.add_argument('--T', type=int, default=50,
        help='number of timestep in the dataset')

    # training parameters
    parser.add_argument('--train', type=int, default=None,
        help='train model')
    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--use_bernoulli', type=int, default=0,
        help='use bernoulli in the decoder')
    parser.add_argument('--n_epoch_initial', type=int, default=20,
        help='number of epochs to wait for to train dynamics parameter')
    
    # testing parameters
    parser.add_argument('--test', type=int, default=None,
        help='test model')
    parser.add_argument('--test_reconstruction', type=int, default=1,
        help='test model reconstruction')
    parser.add_argument('--test_imputation', type=int, default=1,
        help='test model imputation')
    parser.add_argument('--masking_fraction', type=float, default=0.3, 
        help='fraction fo sample being masked for testing imputation')
    parser.add_argument('--n_of_frame_to_generate', type=int, default=100, 
        help='number of sample we want to generate')

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


