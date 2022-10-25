!\bin\bash

## processing main.py

# python main.py --times 0 --seed 998
# python main.py --times 1 --seed 999
# python main.py --times 2 --seed 1000

### for Cifar100-->Cifar10

#### for BatchRemove-0 PGD
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1


# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.01
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01


# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.1

### for Lacuna-100->Lacuna-10

# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3

:<<!
For BatchRemove-1 PGD
!
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 3


:<<!
For BatchRemove-0 FGSM
cg 10
!
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0


:<<!
For BatchRemove-1 FGSM
!
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layer 28 --tuning_lr 0.01 --gpu_id 0


## pre training retrain model

### for Cifar100-->Cifar10

#### for BatchRemove-0 PGD
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1

#### for BatchRemove-1 PGD
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1

#### for BatchRemove-0 FGSM
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.01 --tuning_epochs 20
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20

#### for BatchRemove-1 FGSM
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.01 --tuning_epochs 20
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20


### for Lacuna-100-->Lacuna-10

#### for BatchRemove-0 PGD
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
 
#### for BatchRemove-1 PGD
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3

#### for BatchRemove-0 FGSM
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
 
# #### for BatchRemove-1 FGSM
# python Pre_retrain_model.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0
# python Pre_retrain_model.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 0


## SISA

### for Cifar100-->Cifar10

#### for BatchRemove-0 PGD
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1

#### for BatchRemove-1 PGD
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34 --tuning_lr 0.1
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type PGD --dataset Cifar100 --layers 34  --tuning_lr 0.1

#### for BatchRemove-0 FGSM
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.01 --tuning_epochs 20
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20

#### for BatchRemove-1 FGSM
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34 --tuning_lr 0.01 --tuning_epochs 20
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Cifar100 --layers 34  --tuning_lr 0.01 --tuning_epochs 20


### for Lacuna-100-->Lacuna-10

#### for BatchRemove-0 PGD
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
 
#### for BatchRemove-1 PGD
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3

#### for BatchRemove-0 FGSM
# python main.py --times 0 --seed 998 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 0 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
 
# #### for BatchRemove-1 FGSM
# python main.py --times 0 --seed 998 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 1 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3


### time record

#### for Lacuna-100 BatchRemove 2 PGD remove
# python main.py --times 0 --seed 998 --isBatchRemove 2 --adv_type PGD --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 2 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 2 --adv_type PGD --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3

#### for ImageNet BatchRemove 2 PGD remove
# python main.py --times 0 --seed 998 --isBatchRemove 2 --adv_type PGD --dataset ImageNet --layers 28 --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 2 --adv_type PGD --dataset ImageNet --layers 28  --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 2 --adv_type PGD --dataset ImageNet --layers 28  --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3

#### for Lacuna-100 BatchRemove 2 FGSM remove
# python main.py --times 0 --seed 998 --isBatchRemove 2 --adv_type FGSM --dataset Lacuna-100 --layers 28 --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 2 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 2 --adv_type FGSM --dataset Lacuna-100 --layers 28  --tuning_lr 0.01 --tuning_epochs 20 --gpu_id 3

#### for ImageNet BatchRemove 2 FGSM remove
# python main.py --times 0 --seed 998 --isBatchRemove 2 --adv_type FGSM --dataset ImageNet --layers 28 --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3
# python main.py --times 1 --seed 999 --isBatchRemove 2 --adv_type FGSM --dataset ImageNet --layers 28  --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3
# python main.py --times 2 --seed 1000 --isBatchRemove 2 --adv_type FGSM --dataset ImageNet --layers 28  --tuning_lr 0.1 --tuning_epochs 10 --gpu_id 3
