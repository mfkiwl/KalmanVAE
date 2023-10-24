import torch
import torch.nn as nn
import time
import wandb
import matplotlib.pyplot as plt 

from Kalman_VAE import KalmanVAE
from datetime import datetime

from dataloaders.bouncing_data import BouncingBallDataLoader
from torch.utils.data import DataLoader

def train(train_loader, kvae, optimizer, args):

    loss_epoch = 0.
    idv_losses = {'reconstruction loss': 0,
                  'encoder loss': 0, 
                  'LGSSM observation log likelihood': 0,
                  'LGSSM tranisition log likelihood': 0, 
                  'LGSSM tranisition log posterior': 0}

    for _, sample in enumerate(train_loader, 1):

        optimizer.zero_grad()
        
        sample = sample.cuda().float()

        x_hat, A, C = kvae(sample)
        loss, loss_dict = kvae.calculate_loss(A, C)

        loss.backward()

        optimizer.step()

        # TODO: add option to train with masking
        # TODO: add calculation of MSE
        # TODO: add gradient clipping

        loss_epoch += loss

        for key in idv_losses.keys():
            idv_losses[key] += loss_dict[key]
    
    for key in idv_losses.keys():
        idv_losses[key] = idv_losses[key]/len(train_loader)

    return loss_epoch/len(train_loader), idv_losses

def test_reconstruction(test_loader, kvae, output_folder, args):

    kvae.eval()
    with torch.no_grad():
        mse_error = 0
        for i, sample in enumerate(test_loader, 1):

            sample = sample.cuda().float()
            B, T, C, d1, d2 = sample.size()

            # get mean-squared-error on test data
            x_hat, _, _ = kvae(sample)
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

            # visualize differences in trajectories between sample and reconstruction
            # TODO

        print('Test Mean-Squared-Error: ', mse_error/len(test_loader))


def test_generation(test_loader, kvae, args):
    pass

def test_imputation(test_loader, kvae, args):
    pass

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

    # load model
    kvae = KalmanVAE(n_channels_in,
                     dim,
                     args.dim_a, 
                     args.dim_z, 
                     args.K, 
                     T=T, 
                     recon_scale=args.recon_scale).cuda()
    
    # if already trained, load checkpoints
    if args.kvae_model is not None:
        kvae.load_state_dict(torch.load(args.kvae_model))

    if args.train:
        # define optimizer
        if args.dim_u == 0:
            params = [kvae.A, kvae.C] + list(kvae.encoder.parameters()) + list(kvae.decoder.parameters())
        else:
            params = [kvae.A, kvae.B, kvae.C] + list(kvae.encoder.parameters()) + list(kvae.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma_lr_schedule)

        # helper variables
        start = time.time()
        log_list = []

        # define filename
        now = datetime.now()
        run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
        save_filename = args.output_folder + '/{}'.format(args.dataset) + '/{}'.format(run_name) 
        if not os.path.isdir(save_filename):
            os.makedirs(save_filename)
        
        # set parameters for wandb
        if args.use_wandb:
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

        # training 
        # TODO: 
        # 1. include validation
        # 2. include training with masking
    
        # define number of epochs when NOT to train Dynamic Parameter Network
        n_epoch_initial = 20
        for epoch in range(args.num_epochs):
            
            # delay training of Dynamics Parameter Network to epoch = n_epoch_initial
            if epoch == n_epoch_initial:
                optimizer.add_param_group({'params': kvae.dynamics_net.parameters()})
            
            # train 
            loss_train, loss_dict = train(train_loader, kvae, optimizer, args)
            if args.use_wandb:
                run.log(loss_dict)
            if epoch % 20 == 0 and epoch > 0:
                scheduler.step()
            end = time.time()
            log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
            start = end
            print(log)
            log_list.append(log + '\n')

            # save checkpoints 
            if epoch % 10 == 0 or epoch == args.num_epochs-1:
                with open(save_filename + '/kvae' + str(epoch+1) + '.pt', 'wb') as f:
                    torch.save(kvae.state_dict(), f)
            
            # save training log
            with open(save_filename + '/training.cklog', "a+") as log_file:
                log_file.writelines(log_list)
                log_list.clear()
    
    # testing:
    # TODO:
    # 1. Test reconstruction
    #    - visual comparison (for both frames and whole trajectory)
    #    - compute MSE error for cross-model and cross-mode comparison
    # 2. Test long term generation 
    #    - visual comparison with different numbers of starting frames
    # 3. Test frame imputation
    #    - visual comparison for both random and consecutive frame omission
    #      including both initial and final frames 
    # 4. Reproduce results for Dyanmic Parameter Network
    #    - visual comparison of the the different K=3 dynamics
    if args.test:
        
        if not args.train:
            path_to_dir = args.kvae_model.split('/')[:-1]
            save_filename = "/".join(path_to_dir) 
        
        output_folder = os.path.join(save_filename + '/reconstructions')
        if not os.path.exists('{}'.format(output_folder)):
            os.makedirs(output_folder)
        test_reconstruction(test_loader, kvae, output_folder, args)

        output_folder = os.path.join(save_filename + '/generations')
        if not os.path.exists('{}'.format(output_folder)):
            os.makedirs(output_folder)
        test_generation(test_loader, kvae, args)

        output_folder = os.path.join(save_filename + '/imputations')
        if not os.path.exists('{}'.format(output_folder)):
            os.makedirs(output_folder)
        test_imputation(test_loader, kvae, args)


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
    
    # testing parameters
    parser.add_argument('--test', type=int, default=None,
        help='test model')
    # TODO: add args for testing generations and imputations

    # logistics
    parser.add_argument('--datasets_root_dir', type=str, default="/data2/users/lr4617/data/",
        help='path to the root directory of datasets')
    parser.add_argument('--kvae_model', type=str, default=None,
        help='path to the kvae model dictionary')
    parser.add_argument('--output_folder', type=str, default='results',
        help='location to save kave model and results')
    parser.add_argument('--use_wandb', type=int, default=None,
        help='use weights and biases to track expriments')
    
    # get arguments
    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('{}'.format(args.output_folder)):
        os.makedirs('{}'.format(args.output_folder))
    
    main(args)


