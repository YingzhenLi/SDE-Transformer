"""
Bayesian DL via SDEs

#SDEs as building blocks for Bayesian neural networks

Experiment with ViT
"""

import os
import argparse
import torch
from training.train import train_net, evaluate

EPS = 1e-5  # define a small constant for numerical stability control
DATA = ['mnist', 'cifar10'] # supported datasets
BLOCK_TYPE = ['nn', 'mfvi', 'attention', 'sgpa', 'sde', 'sparsecorrsde', 'mhsde'] # supported blocks
ARCH = ['vit']  # supported network architecture

def main(args):
    # load data
    if args.data == 'mnist':
        from data.load_data import mnist as load_data_func
        num_class = 10
    if args.data == 'cifar10':
        from data.load_data import cifar10 as load_data_func
        num_class = 10
    train_loader, val_loader, test_loader, input_shape, n_class = load_data_func(args.path, args.batch_size)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # build network
    torch.manual_seed(args.seed)
    if args.arch == 'vit':
        from models.vit import construct_net
        patch_size = min(args.patch_size, input_shape[-1])
        layer_kwargs = {'patch_size': (patch_size, patch_size),
                        'input_shape': input_shape,
                        'dim_emb': args.dim_emb,
                        'n_block': args.n_block,
                        'dim_out': n_class,
                        'device': device,
                        'block_type': args.block_type,
                        'inference_mode': 'marginal',
                        'r_min': 0.01,
                        'r_max': 0.2,
                        'n_inducing': 8,
                        'drop_rate': 0.1,
                        'learn_time': False,
                        'norm': True
                        }
        net = construct_net(**layer_kwargs)
    net.to(device)
    print(net)
    if len(args.ckpt) > 0:   # when loading path is provided
        net.load_state_dict(torch.load(args.ckpt))
        print("checkpoint loaded from {}".format(args.ckpt))
        epochs_total = int(args.ckpt.split('/')[-1][5:-3])
    else:
        epochs_total = 0
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("total number of parameters: {}".format(num_params))

    # now start training
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    logs = train_net(net, opt, train_loader, device, args.epochs, args.beta, args.kl, args.map)
    epochs_total += args.epochs

    # evaluate
    accuracy, ece_error = evaluate(net, val_loader, device, args.block_type)
    print('Validation Accuracy: {}%, ECE: {}'.format(accuracy, ece_error))

    # save network
    if args.save:
        save_path = './ckpt/{}_{}_{}'.format(args.arch, args.block_type, args.data)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = save_path + '/epoch{}.pt'.format(epochs_total)
        torch.save(net.state_dict(), save_path)
        print("checkpoint saved to {}".format(save_path))

if __name__ == '__main__':
    parser=argparse.ArgumentParser('Argument Parser')
    parser.add_argument('--seed',type=int, default=0)
    parser.add_argument('--data', choices=DATA, default='mnist')
    parser.add_argument('--arch', choices=ARCH, default='vit')
    parser.add_argument('--block_type', choices=BLOCK_TYPE, default='nn')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4) # learning rate
    parser.add_argument('--n_block', type=int, default=4)
    parser.add_argument('--dim_emb', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--beta', type=float, default=1.0) # kl annealing if applicable
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--kl', type=bool, default=False)
    parser.add_argument('--map', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--path', type=str, default='../data')
    args=parser.parse_args()

    main(args)

