import argparse


def get_config():

    parser = argparse.ArgumentParser(description='The vae with scm')

    # Data settings
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='tree', choices=['celeba', 'pendulum', 'human', 'tree', 'minist'])
    parser.add_argument('--data_dir', type=str, default='./data/c3dtree/', help='data directory')

    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--z_dim', type=int, default=20)
    parser.add_argument('--sup_prop', type=float, default=1, help='proportion of supervised labels')


    # Training settings
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()

    return args
