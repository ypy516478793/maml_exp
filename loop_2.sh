#!/bin/bash
# Basic range with steps for loop

echo start
#
#python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --active=True
python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=1 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot3way/ --num_classes=3 --active=False

#python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=4 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/ --train=True
#
#echo randomLengthTrain finished
#
#python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --active=False
python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=1 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot3way/ --num_classes=3 --active=True
#python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=4 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/ --train=True --randomLengthTrain=False
#
#echo normalTrain finished

#python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/ --train=False --active=False
#python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=2 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot3way/ --num_classes=3 --beta=0.0001 --randomLengthTrain=True --active=True

#python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/ --train=False --active=True
#python test_omni.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=2 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot3way/ --num_classes=3 --beta=0.0001 --randomLengthTrain=True --active=False

#echo active omniTest finished

echo All done