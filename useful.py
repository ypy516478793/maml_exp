import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def read_bias(file):
    with open(file, "rb") as f:
        bias_array = pickle.load(f)
    return bias_array

def plot_line(bias, label):
    """
    plot line with uncertainty
    :param bias: shape (lines, total_shots)
    :param label: active or baseline
    """
    x = np.arange(bias.shape[-1])
    mean = np.mean(bias, axis=0)
    # mean = np.median(bias, axis=0)
    std = np.std(bias, axis=0)
    plt.plot(mean, label=label)
    plt.fill_between(x.reshape(-1), mean + std, mean - std, alpha=0.1)


# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90.beta0.001_allb_randomLengthTrain/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90.beta0.001_allb/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/omniglot5way/cls_5.mbs_32.ubs_1.numstep1.updatelr0.4batchnorm.mt60000kp0.90_allb/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/omniglot3way/cls_3.mbs_32.ubs_2.numstep1.updatelr0.4batchnorm.mt60000kp0.90.beta0.000_allb_noDropTest_labelmax4/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90.beta0.001_allb_noDropTest/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90.beta0.001_allb_newRandomLengthTrain_noDropTest/"
folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/omniglot3way/cls_3.mbs_32.ubs_1.numstep1.updatelr0.4batchnorm.mt60000kp1.00.beta0.000_allb_newRandomLengthTrain_noDropTest_labelmax4/"
# folderName = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/omniglot3way/cls_3.mbs_32.ubs_2.numstep1.updatelr0.4batchnorm.mt60000kp0.90.beta0.000_allb_randomLengthTrain_noDropTest_labelmax4/zero_initial/"
# active_bias = read_bias(os.path.join(folderName, "active/bias_array.pkl"))
active_bias = read_bias(os.path.join(folderName, "active/acc_array.pkl"))
# baseline_bias = read_bias(os.path.join(folderName, "active_baseline/bias_array.pkl"))
baseline_bias = read_bias(os.path.join(folderName, "active_baseline/acc_array.pkl"))

plt.figure()
plot_line(active_bias, "active")
plot_line(baseline_bias, "baseline")
plt.legend()
# plt.ylim([0, 2])
# plt.ylim([0.4, 1])
plt.savefig(folderName+"active/active_learning.png", bbox_inches="tight", dpi=300)
# plt.savefig(folderName+"active/active_learning_median.png", bbox_inches="tight", dpi=300)
plt.close()

print("")

