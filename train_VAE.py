import torch
import torch.nn as nn
import time
import wandb
import matplotlib.pyplot as plt 
import os
from VAE import VAE
from datetime import datetime

from dataloaders.bouncing_data import BouncingBallDataLoader
from torch.utils.data import DataLoader

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD, [reproduction_loss, KLD]


def train(train_loader, vae, optimizer, args):
    loss_epoch = 0.
    idv_losses = {'reconstruction-loss': 0,
                  'KL-loss': 0}
    
    for _, sample in enumerate(train_loader, 1):

        optimizer.zero_grad()
        
        sample = sample.cuda().float()

        x_hat, mean, std = vae(sample)

        loss, idv_losses_list = loss_function(sample, x_hat, mean, std)

        idv_losses['reconstruction-loss'] += idv_losses_list[0]
        idv_losses['KL-loss'] += idv_losses_list[1]

        loss.backward()

        optimizer.step()

        loss_epoch += loss

    for key in idv_losses.keys():
        idv_losses[key] = idv_losses[key]/len(train_loader)

    return loss_epoch/len(train_loader), idv_losses

def test(test_loader, vae, args):
    pass

def generate(vae):
    pass

def main(args):
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
    vae = VAE(n_channels_in,
              dim,
              latent_dim=2).cuda()

    # if already trained, load checkpoints
    if args.vae_model is not None:
        vae.load_state_dict(torch.load(args.vae_model))
    
    if args.train:
        # define optimizer
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr) 
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
            run = wandb.init(project="VAE", 
                            config={"dataset" : args.dataset,
                                    "binary" : binary,
                                    "batch-size" : args.batch_size,
                                    "iterations" : args.num_epochs,
                                    "learning-rate" : args.lr, 
                                    "learning-rate-scheduler": args.lr_scheduler, 
                                    "gamma": args.gamma_lr_schedule,
                                    "grad-clipping": args.use_grad_clipping,
                                    "latent_dim": 2, 
                                    "beta": args.beta},
                            name=run_name)

        for epoch in range(args.num_epochs):

            # train and save log
            loss_train, train_loss_dict = train(train_loader, vae, optimizer, args)
            if args.use_wandb:
                run.log(train_loss_dict)
            if epoch % 20 == 0 and epoch > 0:
                scheduler.step()
            end = time.time()
            log = 'epoch = {}, loss_train = {}, time = {}'.format(epoch+1, loss_train, end-start)
            start = end
            print(log)
            log_list.append(log + '\n')

            # validate
            loss_test, test_loss_dict = test(test_loader, vae, args)
            if args.use_wandb:
                run.log(test_loss_dict)
            log = 'epoch = {}, loss_test = {}, time = {}'.format(epoch+1, loss_test, end-start)
            start = end
            print(log)
            log_list.append(log + '\n')

            # save checkpoints 
            if epoch % 10 == 0 or epoch == args.num_epochs-1:
                with open(save_filename + '/vae' + str(epoch+1) + '.pt', 'wb') as f:
                    torch.save(vae.state_dict(), f)
            
            # save training log
            with open(save_filename + '/training.cklog', "a+") as log_file:
                log_file.writelines(log_list)
                log_list.clear()
    
    if args.test:
        if not args.train:
            path_to_dir = args.kvae_model.split('/')[:-1]
            save_filename = "/".join(path_to_dir) 



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
    parser.add_argument('--beta', type=float, default=0.3, 
        help="importance given to reconstruction during training")
    
    # testing parameters
    parser.add_argument('--test', type=int, default=None,
        help='test model')

    # logistics
    parser.add_argument('--datasets_root_dir', type=str, default="/data2/users/lr4617/data/",
        help='path to the root directory of datasets')
    parser.add_argument('--vae_model', type=str, default=None,
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
