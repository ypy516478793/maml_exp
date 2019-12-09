"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

FLAGS = flags.FLAGS
extra_string = ""

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
# flags.DEFINE_string('baseline', "oracle", 'oracle, or None')
flags.DEFINE_string('baseline', None, 'oracle, or None')
flags.DEFINE_bool('active', False, 'if True, use active method to pick training sample, otherwise, random pick.')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
# flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_float('update_lr', 1e-2, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('allb', True, "if True, inputbs are all data")
flags.DEFINE_bool('randomLengthTrain', False, "if True, length for inputas are random")

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('keep_prob', 0.9, 'if not None, used as keep_prob for all layers')
# flags.DEFINE_float('beta', 0, 'coefficient for l2_regularization on weights')
flags.DEFINE_float('beta', 0.001, 'coefficient for l2_regularization on weights')
flags.DEFINE_bool('drop_connect', False, 'if True, use dropconnect, otherwise, use dropout')
# flags.DEFINE_float('keep_prob', None, 'if not None, used as keep_prob for all layers')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot
flags.DEFINE_bool('load_tensor', True, 'whether we prefetch the data') # equivalent to tf_load_data
flags.DEFINE_bool('no_drop_test', True, 'do not drop on testB') # equivalent to tf_load_data

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator) and FLAGS.datasource == 'sinusoid':
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            # if FLAGS.allb and FLAGS.datasource == 'sinusoid':
            if FLAGS.allb:
                if FLAGS.randomLengthTrain:
                    K_shots = np.random.choice(num_classes*FLAGS.update_batch_size) + 1
                else:
                    K_shots = num_classes*FLAGS.update_batch_size
                a_idx = np.zeros([batch_x.shape[0], K_shots, batch_x.shape[2]]).astype(np.int)
                inputa = np.zeros([batch_x.shape[0], K_shots, batch_x.shape[2]])
                labela = np.zeros([batch_x.shape[0], K_shots, batch_x.shape[2]])
                for i in range(batch_x.shape[0]):
                    a_idx[i] = np.random.choice(batch_x.shape[1], [K_shots, batch_x.shape[2]], replace=False)
                    inputa[i] = batch_x[i, a_idx[i, :, 0]]
                    labela[i] = batch_y[i, a_idx[i, :, 0]]
                inputb = batch_x
                labelb = batch_y

                # h, w, d = batch_x.shape[0], num_classes*FLAGS.update_batch_size, batch_x.shape[2]
                # task_idx = np.repeat(np.arange(h), w*d).reshape(h,w,d)
                # a_idx = np.zeros([h, w, d]).astype(np.int)
                # for i in range(batch_x.shape[0]):
                #     a_idx[i] = np.random.choice(batch_x.shape[1], [w, d], replace=False)
                # feature_idx = np.tile(np.arange(d), h*w).reshape(h,w,d)
                # inputa = batch_x[task_idx, a_idx, feature_idx]
                # labela = batch_y[task_idx, a_idx, feature_idx]
                # inputb = batch_x
                # labelb = batch_y
            else:
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        # inputa = sess.run(model.inputa)
        # labela = sess.run(model.labela)
        # inputb = sess.run(model.inputb)
        # labelb = sess.run(model.labelb)
        # for i in range(5):
        #     plt.figure()
        #     plt.imshow(inputa[0, i, :].reshape(28, 28))
        #     print("image_a {:d}'s label is {}".format(i, labela[0, i, :]))
        #     image_name = "image_a_{:d}_label_{:d}.png".format(i, np.argmax(labela[0, i, :]))
        #     plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/" + image_name , bbox_inches="tight", dpi=300)
        #     plt.close()
        # for i in range(5):
        #     plt.figure()
        #     plt.imshow(inputb[0, i, :].reshape(28, 28))
        #     print("image_b {:d}'s label is {}".format(i, labelb[0, i, :]))
        #     image_name = "image_b_{:d}_label_{:d}.png".format(i, np.argmax(labelb[0, i, :]))
        #     plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/" + image_name , bbox_inches="tight", dpi=300)
        #     plt.close()


        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            # if 'generate' not in dir(data_generator):
            if FLAGS.load_tensor:
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600

def generate_test():
    # batch_size = 2
    batch_size = 10
    num_points = 101
    # amp = np.array([5, 3])
    # phase = np.array([2.3, 0])
    amp = np.array([5, 3, 2.3, 2, 2, 0.9, 1.7, 3.5, 4, 4.5])
    phase = np.array([2.3, 0, 1.2, 2.5, 3.1, 0.5, 0.1, 2.6, 4.6, 1.7])
    outputs = np.zeros([batch_size, num_points, 1])
    init_inputs = np.zeros([batch_size, num_points, 1])
    for func in range(batch_size):
        init_inputs[func, :, 0] = np.linspace(-5, 5, num_points)
        outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])

    if FLAGS.baseline == 'oracle':  # NOTE - this flag is specific to sinusoid
        init_inputs = np.concatenate([init_inputs, np.zeros([init_inputs.shape[0], init_inputs.shape[1], 2])], 2)
        for i in range(batch_size):
            init_inputs[i, :, 1] = amp[i]
            init_inputs[i, :, 2] = phase[i]

    return init_inputs, outputs, amp, phase

def generate_statistic_test(random_seed=42):
    np.random.seed(random_seed)
    amp_range = [0.1, 5.0]
    phase_range = [0, np.pi]
    batch_size = 20
    num_points = 101
    amp = np.random.uniform(amp_range[0], amp_range[1], [batch_size])
    phase = np.random.uniform(phase_range[0], phase_range[1], [batch_size])
    outputs = np.zeros([batch_size, num_points, 1])
    init_inputs = np.zeros([batch_size, num_points, 1])
    for func in range(batch_size):
        init_inputs[func, :, 0] = np.linspace(-5, 5, num_points)
        outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])

    return init_inputs, outputs, amp, phase

def mutual_info(mean_prob, mc_prob):
    """
    computes the mutual information
    :param mean_prob: average MC probabilities of shape [batch_size, img_h, img_w, num_cls]
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, img_h, img_w, num_cls]
    :return: mutual information of shape [batch_size, img_h, img_w, num_cls]
    """
    eps = 1e-5
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=-1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=-1)
    return first_term + second_term


def query(model, sess, inputs_all, outputs_all, mc_simulation, inputs_a=None, outputs_a=None):
    mc_prediction = []
    if inputs_a is None:
        feed_dict_line_initial = {model.inputa: inputs_all, model.inputb: inputs_all, model.labela: outputs_all,
                                  model.labelb: outputs_all, model.meta_lr: 0.0}
        for mc_iter in range(mc_simulation):
            init_predictions_all = sess.run(model.outputas, feed_dict_line_initial)
            mc_prediction.append(np.array(init_predictions_all))
        prob_mean = np.mean(mc_prediction, axis=0)
        prob_variance = np.var(mc_prediction, axis=0)
        query_idx = np.argmax(prob_variance, axis=1)
        return query_idx, prob_mean.squeeze(), prob_variance.squeeze()
    else:
        feed_dict_line = {model.inputa: inputs_a, model.inputb: inputs_all, model.labela: outputs_a,
                          model.labelb: outputs_all, model.meta_lr: 0.0}
        for mc_iter in range(mc_simulation):
            predictions_all = sess.run(model.outputbs, feed_dict_line)
            mc_prediction.append(np.array(predictions_all))
        prob_mean = np.nanmean(mc_prediction, axis=0)  # predictions_all shape: [20, 10, 2, 101, 1]
        prob_variance = np.var(mc_prediction, axis=0)  # prob_mean shape: [10, 2, 101, 1]
        query_idx = np.argmax(prob_variance[-1], axis=1)
        return query_idx, prob_mean[-1].squeeze(), prob_variance[-1].squeeze()

def plot_pred(meta_info, mean, var, X_test, y_test, X_training=None, y_training=None, x_new=None, y_new=None):
    plt.figure()
    plt.plot(X_test, y_test, "--r", label="gt")
    plt.plot(X_test, mean, "-", label="pred")
    std = np.sqrt(var)
    plt.fill_between(X_test.reshape(-1), mean + std, mean - std, alpha=0.1)
    if X_training is not None:
        plt.plot(X_training, y_training, 'gx', label="training")
    if x_new is not None:
        plt.plot(x_new, y_new, 'x', label="new_query")
    plt.legend()
    avg_bias = np.mean(np.abs(mean - y_test.reshape(-1)))
    plt.title("avg_bias: {:.2f}".format(avg_bias))

    amp, phase, exp_string, step = meta_info
    axes = plt.gca()
    ymin = - amp - 3
    ymax = amp + 3
    axes.set_ylim([ymin, ymax])

    line_name = "amp{:.2f}_ph{:.2f}".format(amp, phase)
    save_folder = FLAGS.logdir + '/' + exp_string + "/" + line_name + "/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    out_figure = save_folder + 'test_ubs' + str(
        FLAGS.update_batch_size) + '_stepsize' + str(
        FLAGS.update_lr) + 'step_{:d}.png'.format(step)

    plt.savefig(out_figure, bbox_inches="tight", dpi=300)
    plt.close()

def add_train(index, query_idx):
    if index is None:
        index = query_idx
    else:
        index = np.hstack([index, query_idx])
    return index

def random_query(batch, Train_index, index=None):
    query_idx = np.zeros([batch, 1], dtype=np.int)
    for i in range(len(query_idx)):
        sample_idx = np.random.choice(Train_index)
        if index is not None:
            while sample_idx in index[i]:
                sample_idx = np.random.choice(Train_index)
        query_idx[i, 0] = sample_idx
    return query_idx

def test_line_active_Baye(model, sess, exp_string, mc_simulation=20, total_points_train=10, random=False):

    statistic_results = False
    if statistic_results:
        inputs_all, outputs_all, amp_test, phase_test = generate_statistic_test()
    else:
        inputs_all, outputs_all, amp_test, phase_test = generate_test()
    # np.random.seed(random_seed)
    # Train_index = np.arange(int(inputs_all.shape[1] *1/3), int(inputs_all.shape[1] *2/3))
    Train_index = np.arange(inputs_all.shape[1])

    exp_string = exp_string + "/active"
    if random: exp_string = exp_string + "_baseline"

    index = None
    if index is None:
        total_step = total_points_train
    else:
        total_step = total_points_train - index.shape[-1]

    bias_array = np.zeros([len(inputs_all), total_step+1])

    query_time = 0
    if index is None:
        query_idx, pred, var = query(model, sess, inputs_all, outputs_all, mc_simulation)
        if random:
            query_idx = random_query(inputs_all.shape[0], Train_index)
        for line in range(len(inputs_all)):
            if statistic_results:
                bias_array[line, query_time] = np.mean(np.abs( pred[line] - outputs_all[line].reshape(-1)))
            else:
                meta_info = amp_test[line], phase_test[line], exp_string, query_time
                plot_pred(meta_info, pred[line], var[line], inputs_all[line], outputs_all[line])
    else:
        inputs_a = np.zeros([inputs_all.shape[0], index.shape[-1], inputs_all.shape[2]])
        outputs_a = np.zeros([outputs_all.shape[0], index.shape[-1], outputs_all.shape[2]])
        for line in range(len(inputs_all)):
            inputs_a[line] = inputs_all[line, index[line], :]
            outputs_a[line] = outputs_all[line, index[line], :]
        query_idx, pred, var = query(model, sess, inputs_all, outputs_all, mc_simulation, inputs_a, outputs_a)
        if random:
            query_idx = random_query(inputs_all.shape[0], Train_index, index)
        for line in range(len(inputs_all)):
            if statistic_results:
                bias_array[line, query_time] = np.mean(np.abs( pred[line] - outputs_all[line].reshape(-1)))
            else:
                meta_info = amp_test[line], phase_test[line], exp_string, query_time
                plot_pred(meta_info, pred[line], var[line], inputs_all[line], outputs_all[line], inputs_a[line], outputs_a[line], x_new=None, y_new=None)
    index = add_train(index, query_idx)

    for query_time in tqdm(range(1, total_step+1)):
        inputs_a = np.zeros([inputs_all.shape[0], index.shape[-1], inputs_all.shape[2]])
        outputs_a = np.zeros([outputs_all.shape[0], index.shape[-1], outputs_all.shape[2]])
        for line in range(len(index)):
            inputs_a[line] = inputs_all[line, index[line], :]
            outputs_a[line] = outputs_all[line, index[line], :]
        query_idx, pred, var = query(model, sess, inputs_all, outputs_all, mc_simulation, inputs_a, outputs_a)
        if random:
            query_idx = random_query(inputs_all.shape[0], Train_index, index)
        for line in range(len(index)):
            if statistic_results:
                bias_array[line, query_time] = np.mean(np.abs( pred[line] - outputs_all[line].reshape(-1)))
            else:
                meta_info = amp_test[line], phase_test[line], exp_string, query_time
                plot_pred(meta_info, pred[line], var[line], inputs_all[line], outputs_all[line], inputs_a[line, :-1], outputs_a[line, :-1], inputs_a[line, -1], outputs_a[line, -1])
        index = add_train(index, query_idx)

    if statistic_results:
        with open(FLAGS.logdir + '/' + exp_string + "/bias_array.pkl", "wb") as f:
            pickle.dump(bias_array, f)

def test_line_limit_Baye(model, sess, exp_string, mc_simulation=20, points_train=10, random_seed=1999):

    inputs_all, outputs_all, amp_test, phase_test = generate_test()
    np.random.seed(random_seed)
    # Train_index = np.arange(int(inputs_all.shape[1] *1/3), int(inputs_all.shape[1] *2/3))
    Train_index = np.arange(inputs_all.shape[1])
    index = np.random.choice(Train_index, [inputs_all.shape[0], points_train], replace=False)
    inputs_a = np.zeros([inputs_all.shape[0], points_train, inputs_all.shape[2]])
    outputs_a = np.zeros([outputs_all.shape[0], points_train, outputs_all.shape[2]])
    for line in range(len(index)):
        inputs_a[line] = inputs_all[line, index[line], :]
        outputs_a[line] = outputs_all[line, index[line], :]
    feed_dict_line = {model.inputa: inputs_a, model.inputb: inputs_all,  model.labela: outputs_a, model.labelb: outputs_all, model.meta_lr: 0.0}
    feed_dict_line_initial = {model.inputa: inputs_all, model.inputb: inputs_all,  model.labela: outputs_all, model.labelb: outputs_all, model.meta_lr: 0.0}
    # initial = sess.run(model.outputas, feed_dict_line_initial)
    init_mc_prediction = []
    for mc_iter in range(mc_simulation):
        init_predictions_all = sess.run(model.outputas, feed_dict_line_initial)
        init_mc_prediction.append(np.array(init_predictions_all))
    initial = np.mean(init_mc_prediction, axis=0)
    initial_variance = np.var(init_mc_prediction, axis=0)

    mc_prediction = []
    for mc_iter in range(mc_simulation):
        predictions_all = sess.run(model.outputbs, feed_dict_line)
        mc_prediction.append(np.array(predictions_all))
    print("total mc simulation: ", mc_simulation)
    print("shape of predictions_all is: ", predictions_all[0].shape)

    prob_mean = np.nanmean(mc_prediction, axis=0)      # shape: [10, 2, 101, 1]
    prob_variance = np.var(mc_prediction, axis=0)      # shape: [20, 10, 2, 101, 1]
    # mutual_information = mutual_info(prob_mean, mc_prediction)

    for line in range(len(inputs_all)):

        for update_step in range(len(predictions_all)):
            if update_step % 20 == 0:

                plt.figure()
                X = inputs_all[line, ..., 0].squeeze()
                initial_mu = initial[line, ...].squeeze()
                initial_uncertainty = np.sqrt(initial_variance[line, ...].squeeze())
                plt.plot(X, initial_mu, "g:", label="initial_pred", linewidth=1)
                plt.plot(X, outputs_all[line, ..., 0].squeeze(), "r-", label="ground_truth")
                plt.fill_between(X, initial_mu + initial_uncertainty, initial_mu - initial_uncertainty, alpha=0.1)
                # for update_step in range(len(predictions_all)):

                mu = prob_mean[update_step][line, ...].squeeze()
                uncertainty = np.sqrt(prob_variance[update_step][line, ...].squeeze())
                # uncertainty = mutual_information[update_step][line, ...].squeeze()
                plt.plot(X, mu, "--", label="update_step_{:d}".format(update_step))
                plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)


                plt.legend()
                axes = plt.gca()
                ymin = - amp_test[line] - 3
                ymax = amp_test[line] + 3
                axes.set_ylim([ymin, ymax])

                line_name = "amp{:.2f}_ph{:.2f}_pts{:d}".format(amp_test[line], phase_test[line], points_train)
                save_folder = FLAGS.logdir + '/' + exp_string + '/' + line_name + "/"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                out_figure = save_folder + 'test_ubs' + str(
                    FLAGS.update_batch_size) + '_stepsize' + str(
                    FLAGS.update_lr) + 'line_{0:d}_numtrain_{1:d}_seed_{2:d}_step_{3:d}.png'.format(line, points_train,
                                                                                         random_seed, update_step)
                plt.plot(inputs_a[line, :, 0], outputs_a[line, :, 0], "b*", label="training points")

                plt.savefig(out_figure, bbox_inches="tight", dpi=300)
                plt.close()

        # plt.figure()
        # X = inputs_all[line, ..., 0].squeeze()
        # plt.plot(X, initial[line, ..., 0].squeeze(), "g:", label="initial_pred",  linewidth=1)
        # plt.plot(X, outputs_all[line, ..., 0].squeeze(), "r-", label="ground_truth")
        # # for update_step in range(len(predictions_all)):
        # for update_step in [0, len(predictions_all)-1]:
        #     mu = prob_mean[update_step][line, ...].squeeze()
        #     uncertainty = np.sqrt(prob_variance[update_step][line, ...].squeeze())
        #     # uncertainty = mutual_information[update_step][line, ...].squeeze()
        #     plt.plot(X, mu, "--", label="update_step_{:d}".format(update_step))
        #     plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
        # plt.legend()
        # axes = plt.gca()
        # ymin = - amp_test[line] - 3
        # ymax = amp_test[line] + 3
        # axes.set_ylim([ymin, ymax])
        #
        # line_name = "amp{:.2f}_ph{:.2f}_pts{:d}".format(amp_test[line], phase_test[line], points_train)
        # save_folder = FLAGS.logdir + '/' + exp_string + '/' + line_name + "/"
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # out_figure = save_folder + 'test_ubs' + str(
        #     FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + 'line_{0:d}_numtrain_{1:d}_seed_{2:d}.png'.format(line, points_train, random_seed)
        # plt.plot(inputs_a[line, :, 0], outputs_a[line, :, 0], "b*", label="training points")
        #
        # plt.savefig(out_figure, bbox_inches="tight", dpi=300)
        # plt.close()

def test_line_limit(model, sess, exp_string, num_train=10, random_seed=1999):

    inputs_all, outputs_all, amp_test, phase_test = generate_test()
    np.random.seed(random_seed)
    index = np.random.choice(inputs_all.shape[1], [inputs_all.shape[0], num_train], replace=False)
    inputs_a = np.zeros([inputs_all.shape[0], num_train, inputs_all.shape[2]])
    outputs_a = np.zeros([outputs_all.shape[0], num_train, outputs_all.shape[2]])
    for line in range(len(index)):
        inputs_a[line] = inputs_all[line, index[line], :]
        outputs_a[line] = outputs_all[line, index[line], :]
    feed_dict_line = {model.inputa: inputs_a, model.inputb: inputs_all,  model.labela: outputs_a, model.labelb: outputs_all, model.meta_lr: 0.0}
    predictions_all = sess.run([model.outputas, model.outputbs], feed_dict_line)
    print("shape of predictions_all is: ", predictions_all[0].shape)

    for line in range(len(inputs_all)):
        plt.figure()
        plt.plot(inputs_all[line, ..., 0].squeeze(), outputs_all[line, ..., 0].squeeze(), "r-", label="ground_truth")
        for update_step in range(len(predictions_all[1])):
            plt.plot(inputs_all[line, ..., 0].squeeze(), predictions_all[1][update_step][line, ...].squeeze(), "--", label="update_step_{:d}".format(update_step))
        plt.legend()

        out_figure = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
            FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + 'line_{0:d}_numtrain_{1:d}_seed_{2:d}.png'.format(line, num_train, random_seed)
        plt.plot(inputs_a[line, :, 0], outputs_a[line, :, 0], "b*", label="training points")

        plt.savefig(out_figure, bbox_inches="tight", dpi=300)
        plt.close()

def test_line(model, sess, exp_string):

    inputs_all, outputs_all, amp_test, phase_test = generate_test()


    feed_dict_line = {model.inputa: inputs_all, model.inputb: inputs_all,  model.labela: outputs_all, model.labelb: outputs_all, model.meta_lr: 0.0}
    predictions_all = sess.run([model.outputas, model.outputbs], feed_dict_line)
    print("shape of predictions_all is: ", predictions_all[0].shape)

    for line in range(len(inputs_all)):
        plt.figure()
        plt.plot(inputs_all[line, ..., 0].squeeze(), outputs_all[line, ..., 0].squeeze(), "r-", label="ground_truth")
        for update_step in range(len(predictions_all[1])):
            plt.plot(inputs_all[line, ..., 0].squeeze(), predictions_all[1][update_step][line, ...].squeeze(), "--", label="update_step_{:d}".format(update_step))
        plt.legend()

        out_figure = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
            FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + 'line_{0:d}.png'.format(line)

        plt.savefig(out_figure, bbox_inches="tight", dpi=300)
        plt.close()

    # for line in range(len(inputs_all)):
    #     plt.figure()
    #     plt.plot(inputs_all[line, ..., 0].squeeze(), outputs_all[line, ..., 0].squeeze(), "r-", label="ground_truth")
    #
    #     plt.plot(inputs_all[line, ..., 0].squeeze(), predictions_all[0][line, ...].squeeze(), "--",
    #              label="initial")
    #     plt.legend()
    #
    #     out_figure = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + str(
    #         FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + 'init_line_{0:d}.png'.format(line)
    #
    #     plt.savefig(out_figure, bbox_inches="tight", dpi=300)
    #     plt.close()


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main(random_seed=1999):
    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 1000
            # test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.  ## why?????
        # FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory


    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.load_tensor and (FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot'):
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    if FLAGS.num_filters != 64:
        exp_string += 'hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += 'maxpool'
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')
    if FLAGS.pretrain_iterations != 0:
        exp_string += '.pt' + str(FLAGS.pretrain_iterations)
    if FLAGS.metatrain_iterations != 0:
        exp_string += '.mt' + str(FLAGS.metatrain_iterations)
    if FLAGS.keep_prob is not None:
        exp_string += "kp{:.2f}".format(FLAGS.keep_prob)
    if FLAGS.drop_connect is True:
        exp_string += ".dropconn"
    if FLAGS.beta != 0:
        exp_string += ".beta{:.3f}".format(FLAGS.beta)
    if FLAGS.allb:
        exp_string += "_allb"
    if FLAGS.randomLengthTrain:
        exp_string += "_randomLengthTrain"
    if FLAGS.no_drop_test:
        exp_string += "_noDropTest"

    exp_string += extra_string

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        if exp_string == 'cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000':
            model_file = 'logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000/model69999'
        # elif exp_string == "cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000kp0.50":
        #     model_file = 'logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000kp0.50/model69999'
        else:
            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        # model_file = 'logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000/model69999'
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        # test_line(model, sess, exp_string)
        # test_line_limit(model, sess, exp_string, num_train=2, random_seed=1999)
        random_pick = not FLAGS.active
        test_line_active_Baye(model, sess, exp_string, mc_simulation=20, total_points_train=FLAGS.update_batch_size, random=random_pick)
        # test_line_limit_Baye(model, sess, exp_string, mc_simulation=20, points_train=10, random_seed=1999)
        # test(model, saver, sess, exp_string, data_generator, test_num_updates)

        # repeat_exp = 2
        # np.random.seed(random_seed)
        # sample_seed = np.random.randint(0, 10000, size=repeat_exp)
        # for i in tqdm(range(repeat_exp)):
        #     test_line_limit_Baye(model, sess, exp_string, mc_simulation=20, points_train=2, random_seed=sample_seed[i])

if __name__ == "__main__":
    main()


# import matplotlib.pyplot as plt
# plt.plot(inputa.squeeze(), labela.squeeze(), "*")
# re = sess.run(model.result, feed_dict)
# plt.plot(inputa.squeeze(), re[0].squeeze(), "*")
# plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/maml/preda.png", bbox_inches="tight", dpi=300)
# for i in range(len(re[1])):
#     plt.figure()
#     plt.plot(inputb.squeeze(), labelb.squeeze(), "*")
#     plt.plot(inputb.squeeze(), re[1][i].squeeze(), "*")
#     plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/maml/predb_{:d}.png".format(i), bbox_inches="tight", dpi=300)
#     plt.close()

# plt.figure()
# plt.imshow(metaval_accuracies)
# plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/maml/losses.png", bbox_inches="tight", dpi=300)


## Generate all sine
# def generate_test():
#     amp_range = [0.1, 5.0]
#     phase_range = [0, np.pi]
#     batch_size = 100
#     num_points = 101
#     # amp = np.array([3, 5])
#     # phase = np.array([0, 2.3])
#     amp = np.random.uniform(amp_range[0], amp_range[1], [batch_size])
#     phase = np.random.uniform(phase_range[0], phase_range[1], [batch_size])
#     outputs = np.zeros([batch_size, num_points, 1])
#     init_inputs = np.zeros([batch_size, num_points, 1])
#     for func in range(batch_size):
#         init_inputs[func, :, 0] = np.linspace(-5, 5, num_points)
#         outputs[func] = amp[func] * np.sin(init_inputs[func] - phase[func])
#     return init_inputs, outputs, amp, phase
# init_inputs, outputs, amp, phase = generate_test()
# plt.figure()
# for i in range(len(init_inputs)):
#     plt.plot(init_inputs[i].squeeze(), outputs[i].squeeze())

## Create a video out of images
# import cv2
# import numpy as np
# import glob
# from natsort import natsorted
#
# img_array = []
# for filename in natsorted(glob.glob('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp5.00_ph2.30_pts2/*.png')):
#     img = cv2.imread(filename)
#     height, width, layers = img.shape
#     size = (width, height)
#     img_array.append(img)
#
# out = cv2.VideoWriter('/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/logs/sine/cls_5.mbs_25.ubs_10.numstep1.updatelr0.01nonorm.mt70000kp0.90/amp5.00_ph2.30_pts2/amp5.00_ph2.30_pts2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
#
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()

