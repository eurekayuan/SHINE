import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default='strong_targeted', choices=['strong_targeted', 'weak_targeted', 'untargeted'])
    parser.add_argument('--subname', '-sn', type=str, default='rand0.1', choices=['3x3', '4x4', '5x5', 'cross', 'equal', 'rand0.1', 'rand0.2', 'rand0.3'])
    
    # ours: train poisoned policy with defense, nodef: train poisoned policy without defense, clean: train from scratch
    parser.add_argument('--mode', '-m', type=str, default='ours', choices=['ours', 'nc', 'direct', 'clean'], help='retraining method')

    parser.add_argument('--poisoned_policy', action='store_true', help='evaluate poisoned policy')
    parser.add_argument('--no_poison', action='store_true', help='evaluate in a clean environment')
    args = parser.parse_args()

    return args
