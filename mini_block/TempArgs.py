import argparse

def args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cifar10')
    parser.add_argument('--adv', type=str, default='PGD')
    parser.add_argument('--remove_batch', type=int, default=1000, help='using the mini batch remove method')
    parser.add_argument('--remove_numbers', type=int, default=5000, help='total number for delete')
    parser.add_argument('--device', type=int, default=0, help='the cuda device number')
    parser.add_argument('--epochs', type=int, default=300, help='custom the training epochs')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=128, help='the traning batch size')
    parser.add_argument('--times', type=int, default=0, help='do repeat experiments')
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--ngpu', default=1, type=int)

    # for remove type chose
    parser.add_argument('--adv_type', type=str, default='PGD', help='the adv training type')
    parser.add_argument('--isBatchRemove', type=int, default=1, help='0: no batch, Schur complement. 1: batch, Neumann')

    # for pretrain type
    parser.add_argument('--isPretrain', default=True, type=bool)
    parser.add_argument('--layers', default=28, type=int, help='total number of layers')
    parser.add_argument('--widen_factor', default=10, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
    parser.add_argument('--pretrain_path', default='data/model/pretrain_model/imagenet_wrn_baseline_epoch_', type=str)
    parser.add_argument('--pretrain_model_number', default=99, type=int)
    parser.add_argument('--tuning_epochs', default=50, type=int)
    parser.add_argument('--tuning_lr', default=0.001, type=float)
    parser.add_argument('--tuning_layer', default='linear', type=str)
    parser.add_argument('--isBias', default=False, type=bool)

    args = parser.parse_args([])

    return args
