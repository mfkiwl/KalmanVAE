import torch
import torch.nn as nn
import time
import wandb

from Kalman_VAE import KalmanVAE
from datetime import datetime

from dataloaders.bouncing_data import BouncingBallDataLoader
from torch.utils.data import DataLoader

def train(train_loader, kvae, optimizer, args):

    loss_epoch = 0.

    for _, sample in enumerate(train_loader, 1):

        optimizer.zero_grad()
        
        sample = sample.cuda().float()

        x_hat, A, C = kvae(sample)
        loss = kvae.calculate_loss(A, C)

        loss.backward()

        optimizer.step()

        # TODO: add calculation of MSE
        # TODO: add gradient clipping

        loss_epoch += loss
    
    return loss_epoch/len(train_loader)

def test(valid_loader, kvae, optimizer, args):
    return None

def main(args):

    # load data
    train_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'train')
    test_dir = os.path.join(args.datasets_root_dir, '', args.dataset, '', 'test')
    train_dl = BouncingBallDataLoader(train_dir, images=True)
    test_dl = BouncingBallDataLoader(test_dir, images=True)
    train_loader = DataLoader(train_dl, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(test_dl, batch_size=args.batch_size, shuffle=True)

    it = iter(train_loader)
    first = next(it)
    _, T, n_channels_in, dim, dim = first.size()

    # load model
    kvae = KalmanVAE(n_channels_in,
                     dim,
                     args.dim_a, 
                     args.dim_z, 
                     args.K, 
                     T=T).cuda()
    
    # if already trained, load checkpoints
    if args.kvae_model is not None:
        kvae.load_state_dict(torch.load(args.kvae_model))

    # define optimizer
    if args.dim_u == 0:
        params = [kvae.A, kvae.C] + list(kvae.encoder.parameters()) + list(kvae.decoder.parameters())
    else:
        params = [kvae.A, kvae.B, kvae.C] + list(kvae.encoder.parameters()) + list(kvae.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # helper variables
    start = time.time()
    log_list = []

    # define filename
    now = datetime.now()
    run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
    save_filename = args.output_folder + '/{}'.format(args.dataset) + '/{}'.format(run_name) 
    if not os.path.isdir(save_filename):
        os.makedirs(save_filename)
    if args.use_wandb:
        run = wandb.init(project="autoregressive-trasformer", 
                        config={"dataset" : args.dataset,
                                "batch size" : args.batch_size,
                                "iterations" : args.num_epochs,
                                "learning rate" : args.lr},
                        name=run_name)

    # define number of epochs when NOT to train Dynamic Parameter Network
    n_epoch_initial = 20

    # training + validation loop (+ save checkpoints)
    for epoch in range(args.num_epochs):
        
        # delay training of Dynamics Parameter Network to epoch = n_epoch_initial
        if epoch < n_epoch_initial:
            for param in kvae.dynamics_net.parameters():
                param.requires_grad = False
        elif epoch == n_epoch_initial:
            for param in kvae.dynamics_net.parameters():
                param.requires_grad = True
        
        # train 
        loss_train = train(train_loader, kvae, optimizer, args)
        if args.use_wandb:
            run.log({"loss_train": loss_train})
        end = time.time()
        log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
        start = end
        print(log)
        log_list.append(log + '\n')

        # save checkpoints 
        if epoch % 10 == 0:
            with open(save_filename + '/kvae' + str(epoch+1) + '.pt', 'wb') as f:
                torch.save(kvae.state_dict(), f)
        
        # save training log
        with open(save_filename + '/training.cklog', "a+") as log_file:
            log_file.writelines(log_list)
            log_list.clear()


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
    parser.add_argument('--batch_size', type=int, default=64,
        help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
        help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--use_grad_clipping', type=bool, default=False,
        help='use gradient clipping')
    
    # logistics
    parser.add_argument('--datasets_root_dir', type=str, default="/data2/users/lr4617/data/",
        help='path to the root directory of datasets')
    parser.add_argument('--kvae_model', type=str, default=None,
        help='path to the kvae model dictionary')
    parser.add_argument('--output_folder', type=str, default='results',
        help='location to save kave model and results')
    parser.add_argument('--use_wandb', type=bool, default=False,
        help='use weights and biases to track expriments')
    
    # get arguments
    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./{}'.format(args.output_folder)):
        os.makedirs('./{}'.format(args.output_folder))
    
    main(args)


