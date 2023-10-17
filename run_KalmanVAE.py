import torch
import torch.nn as nn
import time
import wandb

from Kalman_VAE import Kalman_VAE
from datetime import datetime

def train():
    pass

def test():
    pass

def main(args):

    # load data
    ##############################
    # TODO: create ball simulation
    train_loader = None
    valid_loader = None
    ##############################

    # load model
    kvae = Kalman_VAE(args.n_channels_in,
                      args.dim_a, 
                      args.dim_z, 
                      args.K)

    # define optimizer
    optimizer = torch.optim.Adam(kvae.parameters(), lr=args.lr)

    # helper variables
    start = time.time()
    log_list = []

    # define filename
    now = datetime.now()
    run_name = 'run_' + now.strftime("%Y_%m_%d_%H_%M_%S")
    save_filename = args.output_folder + '/{}'.format(args.dataset) + '/{}'.format(run_name) 
    if not os.path.isdir(save_filename):
        os.makedirs(save_filename)
    run = wandb.init(project="autoregressive-trasformer", 
                     config={"dataset" : args.dataset,
                             "batch size" : args.batch_size,
                             "iterations" : args.num_epochs,
                             "learning rate" : args.lr},
                     name=run_name)

    # training + validation loop (+ save checkpoints)
    for epoch in range(args.num_epochs):
        
        # train 
        loss_train = train(train_loader, optimizer, args)
        if args.use_wandb:
            run.log({"loss_train": loss_train})
        end = time.time()
        log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
        start = end
        print(log)
        log_list.append(log + '\n')
        
        # validation
        loss_test = test(valid_loader, optimizer, args)
        if args.use_wandb:
            run.log({"loss_test": loss_test})
        log = 'epoch = {}, loss_test = {}'.format(epoch+1, loss_test)
        print(log)
        log_list.append(log + '\n')

        # save checkpoints 
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
    parser.add_argument('--dataset', type=str, default='Ball_Simulation',
        help='dataset used')
    parser.add_argument('--n_channels_in', type=int, default=3,
        help='number of color channels in the data')

    # encoder parameters
    parser.add_argument('--dim_a', type=int, default=4,
        help='dimensionality of encoded vector a')
    parser.add_argument('--dim_z', type=int, default=2,
        help='dimensionality of encoded vector z')
    parser.add_argument('--K', type=int, default=3,
        help='number of LGSSMs to be mixed')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
        help='batch size for training')
    parser.add_argument('--lr', type=float, default=None,
        help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--use_grad_clipping', type=bool, default=False,
        help='use gradient clipping')
    
    # logistics
    parser.add_argument('-kvae_model', type=str, default=None,
        help='path to the kvae model dictionary')
    parser.add_argument('--output_folder', type=str, default='results',
        help='location to save kave model and results')
    parser.add_argument('use_wandb', type=bool, default=False,
        help='use weights and biases to track expriments')
    
    # get arguments
    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./{}'.format(args.output_folder)):
        os.makedirs('./{}'.format(args.output_folder))
    
    main(args)


