python train_cifar.py --lr 0.1 --dataset cifar10 --epochs 200 --arch bn_wideresnet16 --switch_time 1 --inner_lr 0.01 --inner_steps 1 --augment --reg_type adv_full --save_dir log/ --data_dir ~/datasets