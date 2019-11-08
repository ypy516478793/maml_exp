#!/bin/bash
# Basic range with steps for loop

beta_array=(0.01 0.1 0.001)

for beta in ${beta_array[@]};
do
#  echo $beta
   python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10 --beta=$beta
done

echo All done