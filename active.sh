#!/bin/bash
# Basic range with steps for loop

python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --active=True

echo active experiment finished

python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --active=False

echo All done