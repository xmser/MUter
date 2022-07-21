from random import choices
from email.policy import default
import torch
import argparse
import os

from Train import Neter
from Remover import MUterRemover, NewtonRemover, InfluenceRemover, FisherRemover, SchurMUterRemover
from Recorder import Recorder
from data_utils import Dataer
from utils import get_layers
from SISA import SISA
from utils import get_random_sequence
"""
mainly code for machine unlearning, un see the detail of
the concrete code about how to calculate the matrix and its inverse
or sub operation or save or load matrix and so on.
"""

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cifar10')
parser.add_argument('--adv', type=str, default='PGD')
parser.add_argument('--remove_batch', type=int, default=2500, help='using the mini batch remove method')
parser.add_argument('--remove_numbers', type=int, default=10000, help='total number for delete')
parser.add_argument('--device', type=int, default=0, help='the cuda device number')
parser.add_argument('--epochs', type=int, default=300, help='custom the training epochs')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=128, help='the traning batch size')
parser.add_argument('--times', type=int, default=1, help='do repeat experiments')
parser.add_argument('--gpu_id', default=1, type=int)
parser.add_argument('--ngpu', default=1, type=int)

# for remove type chose
parser.add_argument('--adv_type', type=str, default='FGSN', help='the adv training type')
parser.add_argument('--isBatchRemove', type=int, default=0, help='0: no batch, Schur complement. 1: batch, Neumann')

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

# for repeat experiments
parser.add_argument('--seed', default=666, type=int, help='determate the remove data id')

args = parser.parse_args()


remove_squence_dict = {
    0: [1, 200, 500, 1000, 2000, 4000],
    1: [remain_head for remain_head in range(args.remove_batch, args.remove_numbers + 1, args.remove_batch)]
}

remove_squence = remove_squence_dict[args.isBatchRemove]

"""
1) traninig a roubust model for unlearning (adding SISA)
2) pre calculate the matrix, store and load
3) the unlearning request coming, do unlearning and measure the metrics.
4) post of unlearning 
"""

### pre work
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu_id)
recorder = Recorder(args=args)

pretrain_param = None
if args.isPretrain:
    pretrain_param = {
        'layers': args.layers,
        'widen_factor': args.widen_factor,
        'droprate': args.droprate,
        'root_path': args.pretrain_path + '{}'.format(args.pretrain_model_number) + '.pt',
        'epochs': args.tuning_epochs,
        'lr': args.tuning_lr,
        'new_last_layer': get_layers(args.tuning_layer, isBias=args.isBias),
    }

#####
# Stage 1) traninig a roubust model for unlearning (adding SISA)
#####

dataer = Dataer(dataset_name=args.dataset)
resort_sequence = get_random_sequence(dataer.train_data_lenth, resort_lenth=args.remove_numbers, seed=args.seed)
dataer.set_sequence(sequence=resort_sequence)

neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)

# after pre save model, we could load model
neter.load_model('FGSM_Batch_model_ten_0_times{}'.format(args.times))
# print('Train acc: {:.2f}%'.format(neter.test(isTrainset=True) * 100))
# print('Test acc: {:.2f}%'.format(neter.test(isTrainset=False) * 100))
# print('Adv Train test acc: {:.2f}%'.format(neter.test(isTrainset=True, isAttack=True)*100))
# print('Adv Test acc: {:.2f}%'.format(neter.test(isTrainset=False, isAttack=True)*100))

neter.initialization(isCover=True)  # init generate the adv samples, inner output files.

# sisaer = SISA(dataer=dataer, args=args, shards_num=5, slices_num=5)
# sisaer.Reload()
# sisaer.sisa_train(isAdv=True)
# sisaer.sisa_remove(sequence=[17211, ], isTrain=True, isAdv=True)


# test inner output acc 
# print('clean train acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=True, isAdv=False) * 100))
# print('adv train acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=True, isAdv=True) * 100))

# print('clean test acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=False, isAdv=False) * 100))
# print('adv test acc {:.2f}%'.format(neter.test_inner_out_acc(isTrain=False, isAdv=True) * 100))


# # pre save model
# time = neter.training(epochs=args.epochs, lr=args.lr, batch_size=args.batchsize, isAdv=True)
# print('time {:.2f}'.format(time))
# neter.save_model()


# # ########
# # ### stage 2) pre calculate the matrix, store and load
# # ########
if args.isBatchRemove == 1:
    muter = MUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)
else:
    muter = SchurMUterRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)

newton_delta = NewtonRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Newton_delta', args=args)
newton = NewtonRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Newton', args=args)
influence_delta = InfluenceRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Influence_delta', args=args)
influence = InfluenceRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Influence', args=args)
fisher_delta = FisherRemover(basic_neter=neter, dataer=dataer, isDelta=True, remove_method='Fisher_delta', args=args)
fisher = FisherRemover(basic_neter=neter, dataer=dataer, isDelta=False, remove_method='Fisher', args=args)


# ####
# stage 3) the unlearning request coming, do unlearning and measure the metrics.
# stage 4) post of unlearning 
# ####

for index, remain_head in enumerate(remove_squence):

    remove_head = 0
    if index > 0:
        remove_head = remove_squence[index - 1]
    print('Unlearning deomain [{} -- {})'.format(remove_head, remain_head))

    # ## 1) for retrain
    retrain_neter = Neter(dataer=dataer, args=args, isTuning=args.isPretrain, pretrain_param=pretrain_param)
    # # spending_time = retrain_neter.training(args.epochs, lr=args.lr, batch_size=args.batchsize, head=remain_head)
    retrain_neter.load_model('FGSM_Batch_model_ten_{}_times{}'.format(remain_head, args.times))

    # recorder.metrics_time_record(method='Retrain', time=spending_time)

    # 2) for SISA
    # under construction


    # 3) for MUter
    unlearning_time = muter.Unlearning(head=remove_head, rear=remain_head)

    recorder.metrics_time_record(method=muter.remove_method, time=unlearning_time)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=muter)

    # ## 4) for Newton_delta, Newton
    newton_delta.Unlearning(head=remove_head, rear=remain_head)
    newton.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton)


    # ## 5) for Influence_delta, Influence
    influence_delta.Unlearning(head=remove_head, rear=remain_head)
    influence.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence)


    # ## 6) for Fisher_delta, Fisher
    fisher_delta.Unlearning(head=remove_head, rear=remain_head)
    fisher.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher)


# save information
recorder.save()





