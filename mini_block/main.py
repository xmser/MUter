import torch
import argparse

from Train import Neter
from Remover import MUterRemover, NewtonRemover, InfluenceRemover, FisherRemover
from Recorder import Recorder
from datamarket import Dataer
"""
mainly code for machine unlearning, un see the detail of
the concrete code about how to calculate the matrix and its inverse
or sub operation or save or load matrix and so on.
"""

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='Cifar10')
parser.add_argument('--adv', type=str, default='PGD')
parser.add_argument('remove_batch', type=int, default=100, help='using the mini batch remove method')
parser.add_argument('--remove_numbers', type=int, default=2000, help='total number for delete')
parser.add_argument('--device', type=int, default=0, help='the cuda device number')
parser.add_argument('--epochs', type=int, default=100, help='custom the training epochs')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=128, help='the traning batch size')
parser.add_argument('--times', type=int, default=0, help='do repeat experiments')

args = parser.parse_args()

"""
1) traninig a roubust model for unlearning (adding SISA)
2) pre calculate the matrix, store and load
3) the unlearning request coming, do unlearning and measure the metrics.
4) post of unlearning 
"""

### pre work
recorder = Recorder(args=args)

#####
# Stage 1) traninig a roubust model for unlearning (adding SISA)
#####

dataer = Dataer(dataset_name=args.dataset)
neter = Neter(dataer=dataer, args=args)
neter.training(epochs=args.epochs, lr=args.lr, batch_size=args.batchsize, isAdv=False)


#####
# stage 2) pre calculate the matrix, store and load
#####

muter = MUterRemover(neter=neter, dataer=dataer, isDelta=True, remove_method='MUter', args=args)
newton_delta = NewtonRemover(neter=neter, dataer=dataer, isDelta=True, remove_method='Newton_delta', args=args)
newton = NewtonRemover(neter=neter, dataer=dataer, isDelta=False, remove_method='Newton', args=args)
influence_delta = InfluenceRemover(neter=neter, dataer=dataer, isDelta=True, remove_method='Influence_delta', args=args)
influence = InfluenceRemover(neter=neter, dataer=dataer, isDelta=False, remove_method='Influence', args=args)
fisher_delta = FisherRemover(neter=neter, dataer=dataer, isDelta=True, remove_method='Fisher', args=args)
fisher = FisherRemover(neter=neter, dataer=dataer, isDelta=False, remove_method='Fisher', args=args)

muter.Load_matrix()
newton_delta.Load_matrix()
newton.Load_matrix()
influence_delta.Load_matrix()
influence.Load_matrix()
fisher_delta.Load_matrix()
fisher.Load_matrix()


#####
# stage 3) the unlearning request coming, do unlearning and measure the metrics.
# stage 4) post of unlearning 
#####

for remain_head in range(args.remove_batch, args.remove_numbers + 1, args.remove_batch):

    remove_head = remain_head - args.remove_batch

    ## 1) for retrain
    retrain_neter = Neter(dataer=dataer)
    spending_time = retrain_neter.training(args.epochs, lr=args.lr, batch_size=args.batchsize, head=remain_head)

    recorder.metrics_time_record(method='Retrain', time=spending_time)

    ## 2) for SISA
    ## under construction

    ## 3) for MUter
    unlearning_time = muter.Unlearning(head=remove_head, rear=remove_head)

    recorder.metrics_time_record(method=muter.remove_method, time=unlearning_time)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=muter)

    ## 4) for Newton_delta, Newton
    newton_delta.Unlearning(head=remove_head, rear=remain_head)
    newton.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=newton)


    ## 5) for Influence_delta, Influence
    influence_delta.Unlearning(head=remove_head, rear=remain_head)
    influence.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=influence)


    ## 6) for Fisher_delta, Fisher
    fisher_delta.Unlearning(head=remove_head, rear=remain_head)
    fisher.Unlearning(head=remove_head, rear=remain_head)

    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher_delta)
    recorder.log_metrics(retrain_neter=retrain_neter, compared_remover=fisher)


# save information
recorder.save()