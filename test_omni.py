import numpy as np
import pickle
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def query(model, sess, inputa, labela, inputb, labelb, mc_simulation, train_index=None, val_index=None):
    mc_prediction = []
    batch_size, len_val = val_index.shape
    if train_index is None:
        input_val_test = np.concatenate([inputa, inputb], axis=1)
        label_val_test = np.concatenate([labela, labelb], axis=1)
        feed_dict_line_initial = {model.inputa: input_val_test, model.inputb: inputb, model.labela: label_val_test,
                                  model.labelb: labelb, model.meta_lr: 0.0}
        for mc_iter in range(mc_simulation):
            init_predictions_all = sess.run(model.outputas, feed_dict_line_initial)
            mc_prediction.append(np.array(init_predictions_all))
        prob_mean = np.mean(mc_prediction, axis=0)
        prob_variance = np.var(mc_prediction, axis=0)
        remove_idx = np.argmax(prob_variance[:, :len_val], axis=1)
        query_idx = val_index[np.arange(batch_size), remove_idx][..., np.newaxis]
        val_index = np.array([np.delete(val_index[i], remove_idx[i]) for i in range(len(val_index))])
        return query_idx, val_index, prob_mean[:, len_val:].squeeze(), prob_variance[:, len_val:].squeeze()
    else:
        input_train = np.array([inputa[i, train_index[i], :] for i in range(batch_size)])
        label_train = np.array([labela[i, train_index[i], :] for i in range(batch_size)])
        input_val = np.array([inputa[i, val_index[i], :] for i in range(batch_size)])
        label_val = np.array([labela[i, val_index[i], :] for i in range(batch_size)])
        input_val_test = np.concatenate([input_val, inputb], axis=1)
        label_val_test = np.concatenate([label_val, labelb], axis=1)
        feed_dict_line = {model.inputa: input_train, model.inputb: input_val_test, model.labela: label_train,
                          model.labelb: label_val_test, model.meta_lr: 0.0}
        for mc_iter in range(mc_simulation):
            predictions_all = sess.run(model.outputbs, feed_dict_line)
            mc_prediction.append(np.array(predictions_all))
        prob_mean = np.nanmean(mc_prediction, axis=0)  # predictions_all shape: [20, 10, 2, 101, 1]
        prob_variance = np.var(mc_prediction, axis=0)  # prob_mean shape: [10, 2, 101, 1]
        # pred = np.argmax(prob_mean, axis=-1)
        var_one = np.nanmean(prob_variance[-1], axis=-1)
        # var_one = var_calculate_2d(pred, prob_variance)
        # var_one = predictive_entropy(prob_mean)
        # var_one = mutual_info(prob_mean, mc_prediction)
        remove_idx = np.argmax(var_one[:, :len_val], axis=1)
        # query_idx = np.array([val_index[i, remove_idx[i]] for i in range(batch_size)])
        query_idx = val_index[np.arange(batch_size), remove_idx][..., np.newaxis]
        val_index = np.array([np.delete(val_index[i], remove_idx[i]) for i in range(batch_size)])
        return query_idx, val_index, prob_mean[-1][:, len_val:], prob_variance[-1][:, len_val:]

def add_train(train_index, query_idx):
    if train_index is None:
        train_index = query_idx
    else:
        train_index = np.hstack([train_index, query_idx])
    return train_index

def random_query(batch, All_index, train_index=None):
    query_idx = np.zeros([batch, 1], dtype=np.int)
    for i in range(len(query_idx)):
        sample_idx = np.random.choice(All_index)
        if train_index is not None:
            while sample_idx in train_index[i]:
                sample_idx = np.random.choice(All_index)
        query_idx[i, 0] = sample_idx
    return query_idx

def plot_query(input_query, label_query, input_b, pred_b, label_b, exp_string, step, task):
    exp_string = exp_string + "/task_{:d}".format(task)
    plt.figure()
    plt.imshow(input_query.reshape(28, 28), cmap="Greys")
    plt.title("charactor: {:d}".format(np.argmax(label_query)))
    save_folder = FLAGS.logdir + '/' + exp_string + "/query_list/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    out_figure = save_folder + "query_charactor-{0:d}_step-{1:d}.png".format(np.argmax(label_query), step)
    plt.savefig(out_figure, bbox_inches="tight", dpi=300)
    plt.close()

    num_correct = 0
    num_wrong = 0
    save_correct_folder = FLAGS.logdir + '/' + exp_string + "/step-{:d}_correct/".format(step)
    os.makedirs(save_correct_folder)
    save_wrong_folder = FLAGS.logdir + '/' + exp_string + "/step-{:d}_wrong/".format(step)
    os.makedirs(save_wrong_folder)

    for i in tqdm(range(len(pred_b))):
        plt.figure()
        plt.imshow(input_b[i].reshape(28, 28), cmap="Greys")
        pred_label = np.argmax(pred_b[i])
        gt_label = np.argmax(label_b[i])
        plt.title("pred-{0:d}_gt-{1:d}".format(pred_label, gt_label))
        if pred_label == gt_label:
            save_folder = save_correct_folder
            num_correct += 1
        else:
            save_folder = save_wrong_folder
            num_wrong += 1
        out_figure = save_folder + "test-{0:d}_gt-{1:d}_pred-{2:d}.png".format(i, gt_label, pred_label)
        plt.savefig(out_figure, bbox_inches="tight", dpi=300)
        plt.close()

    os.renames(save_correct_folder, save_correct_folder[:-1] + "_" + str(num_correct))
    os.renames(save_wrong_folder, save_wrong_folder[:-1] + "_" + str(num_wrong))


def test_omni_active_Baye(model, sess, exp_string, data_generator, mc_simulation=20, total_points_train=10, random_pick=False):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    exp_string = exp_string + "/active"
    if random_pick: exp_string = exp_string + "_baseline"

    statistic_results = False
    if statistic_results:
        inputs_all, outputs_all, amp_test, phase_test = data_generator.generate(train=False)
    else:
        inputs_all, outputs_all, amp_test, phase_test = data_generator.generate(train=False)
    # np.random.seed(random_seed)
    # All_index = np.arange(int(inputs_all.shape[1] *1/3), int(inputs_all.shape[1] *2/3))
    num_train_val = int(num_classes * data_generator.total_examples_per_class * 0.6)
    inputa = inputs_all[:, :num_train_val, :]
    inputb = inputs_all[:, num_train_val:, :]
    labela = outputs_all[:, :num_train_val, :]
    labelb = outputs_all[:, num_train_val:, :]

    All_index = np.arange(num_train_val)
    # All_index = np.arange(num_classes * FLAGS.update_batch_size)

    # train_index = None  # shape: (batch_size, num_classes*labeled_samples_per_class)
    seed_size = FLAGS.update_batch_size
    start_idx = np.random.choice(int(num_train_val / num_classes) - (seed_size-1), size=(FLAGS.meta_batch_size, 1))
    train_index = np.array([np.arange(start_idx[i]*num_classes, (start_idx[i]+seed_size)*num_classes) for i in range(FLAGS.meta_batch_size)])
    # train_index = np.array([np.random.choice(All_index, size=1, replace=False) for _ in range(FLAGS.meta_batch_size)])
    if train_index is None:
        total_step = total_points_train
        val_index = All_index
    else:
        total_step = total_points_train - train_index.shape[-1]
        val_index = np.array([np.setdiff1d(All_index, i) for i in train_index])

    acc_array = np.zeros([len(inputb), total_step+1])
    query_time = 0

    query_idx, val_index, pred, var = query(model, sess, inputa, labela, inputb, labelb, mc_simulation, train_index, val_index)
    if random_pick:
        query_idx = random_query(len(inputs_all), All_index, train_index)
    for task in range(len(inputs_all)):
        if statistic_results:
            acc_array[task, query_time] = accuracy_score(np.argmax(pred[task], axis=-1), np.argmax(labelb[task], axis=-1))
        else:
            acc_array[task, query_time] = accuracy_score(np.argmax(pred[task], axis=-1),
                                                         np.argmax(labelb[task], axis=-1))
            plot_query(inputa[task, query_idx[task]], labela[task, query_idx[task]], inputb[task], pred[task], labelb[task], exp_string, query_time, task)
    train_index = add_train(train_index, query_idx)


    for query_time in tqdm(range(1, total_step+1)):
        query_idx, val_index, pred, var = query(model, sess, inputa, labela, inputb, labelb, mc_simulation, train_index,
                                                val_index)
        if random_pick:
            query_idx = random_query(len(inputs_all), All_index, train_index)
        for task in range(len(inputs_all)):
            if statistic_results:
                acc_array[task, query_time] = accuracy_score(np.argmax(pred[task], axis=-1),
                                                             np.argmax(labelb[task], axis=-1))
            else:
                acc_array[task, query_time] = accuracy_score(np.argmax(pred[task], axis=-1),
                                                             np.argmax(labelb[task], axis=-1))
                plot_query(inputa[task, query_idx[task]], labela[task, query_idx[task]], inputb[task], pred[task],
                           labelb[task], exp_string, query_time, task)
        train_index = add_train(train_index, query_idx)

    if statistic_results:
        with open(FLAGS.logdir + '/' + exp_string + "/acc_array.pkl", "wb") as f:
            pickle.dump(acc_array, f)
    else:
        with open(FLAGS.logdir + '/' + exp_string + "/acc_array.pkl", "wb") as f:
            pickle.dump(acc_array, f)
if __name__ == '__main__':

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
    flags.DEFINE_integer('metatrain_iterations', 15000,
                         'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
    flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
    flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
    flags.DEFINE_integer('update_batch_size', 5,
                         'number of examples used for inner gradient update (K for K-shot learning).')
    # flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
    flags.DEFINE_float('update_lr', 1e-2, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
    flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
    flags.DEFINE_bool('allb', True, "if True, inputbs are all data")
    flags.DEFINE_bool('randomLengthTrain', True, "if True, length for inputas are random")

    ## Model options
    flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
    flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
    flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
    flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
    flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
    flags.DEFINE_float('keep_prob', 1, 'if not None, used as keep_prob for all layers')
    flags.DEFINE_float('beta', 0.0001, 'coefficient for l2_regularization on weights')
    flags.DEFINE_bool('drop_connect', False, 'if True, use dropconnect, otherwise, use dropout')
    # flags.DEFINE_float('keep_prob', None, 'if not None, used as keep_prob for all layers')

    ## Logging, saving, and testing options
    flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
    flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
    flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
    flags.DEFINE_bool('train', False, 'True to train, False to test.')
    flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
    flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
    flags.DEFINE_integer('train_update_batch_size', -1,
                         'number of examples used for gradient update during training (use if you want to test with a different number).')
    flags.DEFINE_float('train_update_lr', -1,
                       'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot
    flags.DEFINE_bool('load_tensor', False, 'whether we prefetch the data')  # equivalent to tf_load_data
    flags.DEFINE_bool('no_drop_test', True, 'do not drop on testB')  # equivalent to tf_load_data
    flags.DEFINE_integer('label_max', 4, 'maximal labels we can require per classes')

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
            # test_num_updates = 10
            test_num_updates = 100

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1                                    ########################################### not sure if we can use batch size > 1 in test (We cannot. Very important!!)

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
    if FLAGS.label_max is not None:
        exp_string += "_labelmax{:d}".format(FLAGS.label_max)

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

            # exp_string = "cls_3.mbs_32.ubs_1.numstep1.updatelr0.4batchnorm.mt60000kp1.00.beta0.000_allb_randomLengthTrain_noDropTest_labelmax4"
            # exp_string = "cls_3.mbs_32.ubs_1.numstep1.updatelr0.4batchnorm.mt60000kp0.90.beta0.000_allb_randomLengthTrain_noDropTest_labelmax4"


            model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        # model_file = 'logs/sine//cls_5.mbs_25.ubs_10.numstep1.updatelr0.001nonorm.mt70000/model69999'
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    random_pick = not FLAGS.active
    total_points_train = FLAGS.label_max * FLAGS.num_classes
    test_omni_active_Baye(model, sess, exp_string, data_generator, mc_simulation=20, total_points_train=total_points_train, random_pick=random_pick)



# import time
# start = time.time()
# for i in range(100):
#     in_train = np.array([inputa[i, train_index[i], :] for i in range(batch_size)])
#     la_train = np.array([labela[i, train_index[i], :] for i in range(batch_size)])
# end = time.time()
# print("spent {}".format(end-start))
# # >>> spent 0.017204761505126953
#
#
# start = time.time()
# len_train = train_index.shape[-1]
# for i in range(100):
#     X1 = np.repeat(np.arange(batch_size), len_train * data_generator.dim_input).reshape(batch_size, len_train, data_generator.dim_input)
#     Y1 = np.repeat(train_index, data_generator.dim_input).reshape(batch_size, len_train, data_generator.dim_input)
#     Z1 = np.tile(np.arange(data_generator.dim_input), len_train * batch_size).reshape(batch_size, len_train, data_generator.dim_input)
#     X2 = np.repeat(np.arange(batch_size), len_train * data_generator.num_classes).reshape(batch_size, len_train, data_generator.num_classes)
#     Y2 = np.repeat(train_index, data_generator.num_classes).reshape(batch_size, len_train, data_generator.num_classes)
#     Z2 = np.tile(np.arange(data_generator.num_classes), len_train * batch_size).reshape(batch_size, len_train, data_generator.num_classes)
#     in_train_2 = inputa[X1, Y1, Z1]
#     la_train_2 = labela[X2, Y2, Z2]
# end = time.time()
# print("spent {}".format(end-start))
# # >>> spent 0.06331181526184082


# for exp in range(10):
#     mc_iteration = 3
#     train_acc_list = np.zeros([mc_iteration, 101])
#     train_val_list = np.zeros([mc_iteration, 101])
#     test_list = np.zeros([mc_iteration, 101])
#     train_1_val_list = np.zeros([mc_iteration, 101])
#     train_2_val_list = np.zeros([mc_iteration, 101])
#     train_3_val_list = np.zeros([mc_iteration, 101])
#
#     for i in range(mc_iteration):
#         feed_init = {model.inputa: inputb, model.inputb: inputb, model.labela: labelb,
#                      model.labelb: labelb, model.meta_lr: 0.0}
#         acc0, _ = sess.run([model.total_accuracy1, model.total_accuracies2], feed_init)
#         train_acc_list[i][0] = acc0
#         train_val_list[i][0] = acc0
#         test_list[i][0] = acc0
#         train_1_val_list[i][0] = acc0
#         train_2_val_list[i][0] = acc0
#         train_3_val_list[i][0] = acc0
#
#     for i in range(mc_iteration):
#         feed_dict_line = {model.inputa: input_train, model.inputb: inputb, model.labela: label_train,
#                           model.labelb: labelb, model.meta_lr: 0.0}
#         feed_dict_line_1 = {model.inputa: inputa, model.inputb: inputb, model.labela: labela,
#                             model.labelb: labelb, model.meta_lr: 0.0}
#         feed_dict_line_2 = {model.inputa: inputb, model.inputb: inputb, model.labela: labelb,
#                             model.labelb: labelb, model.meta_lr: 0.0}
#         idx = np.random.choice(range(val_index.shape[-1]), size=3, replace=False)
#         input_train_val1 = np.concatenate([input_train, input_val[:, idx[:1], :]], axis=1)
#         label_train_val1 = np.concatenate([label_train, label_val[:, idx[:1], :]], axis=1)
#         input_train_val2 = np.concatenate([input_train, input_val[:, idx[:2], :]], axis=1)
#         label_train_val2 = np.concatenate([label_train, label_val[:, idx[:2], :]], axis=1)
#         input_train_val3 = np.concatenate([input_train, input_val[:, idx[:3], :]], axis=1)
#         label_train_val3 = np.concatenate([label_train, label_val[:, idx[:3], :]], axis=1)
#
#         feed_dict_line_3 = {model.inputa: input_train_val1, model.inputb: inputb, model.labela: label_train_val1,
#                             model.labelb: labelb, model.meta_lr: 0.0}
#         feed_dict_line_4 = {model.inputa: input_train_val2, model.inputb: inputb, model.labela: label_train_val2,
#                             model.labelb: labelb, model.meta_lr: 0.0}
#         feed_dict_line_5 = {model.inputa: input_train_val3, model.inputb: inputb, model.labela: label_train_val3,
#                             model.labelb: labelb, model.meta_lr: 0.0}
#
#         acc1, accs2 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line)
#         acc1_1, accs2_1 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_1)
#         acc1_2, accs2_2 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_2)
#         acc1_3, accs2_3 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_3)
#         acc1_4, accs2_4 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_4)
#         acc1_5, accs2_5 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_5)
#
#         train_acc_list[i][1:] = accs2
#         train_val_list[i][1:] = accs2_1
#         test_list[i][1:] = accs2_2
#         train_1_val_list[i][1:] = accs2_3
#         train_2_val_list[i][1:] = accs2_4
#         train_3_val_list[i][1:] = accs2_5
#
#     plt.figure(figsize=(6.4 * 2, 4.8 * 2))
#     plt.plot(np.mean(train_acc_list, axis=0), "-*", label="train")
#     plt.fill_between(np.arange(101), np.mean(train_acc_list, axis=0) + np.std(train_acc_list, axis=0),
#                      np.mean(train_acc_list, axis=0) - np.std(train_acc_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_val_list, axis=0), "-*", label="train+val")
#     plt.fill_between(np.arange(101), np.mean(train_val_list, axis=0) + np.std(train_val_list, axis=0),
#                      np.mean(train_val_list, axis=0) - np.std(train_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(test_list, axis=0), "-*", label="test")
#     plt.fill_between(np.arange(101), np.mean(test_list, axis=0) + np.std(test_list, axis=0),
#                      np.mean(test_list, axis=0) - np.std(test_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_1_val_list, axis=0), "-*", label="train+1_val")
#     plt.fill_between(np.arange(101), np.mean(train_1_val_list, axis=0) + np.std(train_1_val_list, axis=0),
#                      np.mean(train_1_val_list, axis=0) - np.std(train_1_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_2_val_list, axis=0), "-*", label="train+2_val")
#     plt.fill_between(np.arange(101), np.mean(train_2_val_list, axis=0) + np.std(train_2_val_list, axis=0),
#                      np.mean(train_2_val_list, axis=0) - np.std(train_2_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_3_val_list, axis=0), "-*", label="train+3_val")
#     plt.fill_between(np.arange(101), np.mean(train_3_val_list, axis=0) + np.std(train_3_val_list, axis=0),
#                      np.mean(train_3_val_list, axis=0) - np.std(train_3_val_list, axis=0), alpha=0.1)
#     plt.legend()
#     plt.title("train 100 steps with")
#     # out_figure = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/" + "3way1shot_oriTask_mc{}.png".format(mc_iteration)
#     out_figure = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/3way1shot1kp_normGrad/" + "3way1shot_oriTask_mc{}_{}_exp{}.png".format(
#         mc_iteration, np.argmax(label_train_val3[0, 3:], axis=-1).tolist(), exp)
#     plt.savefig(out_figure, bbox_inches="tight", dpi=200)
#     plt.close()

## similarity (gradient check)
# with tf.variable_scope("model", reuse=True) as scope:
#     task_outputa = model.forward(model.inputa, model.weights, reuse=True,
#                                  keep_prob=model.keep_prob)  # only reuse on the first iter
#     task_lossa = model.loss_func(task_outputa, model.labela, model.weights, beta=FLAGS.beta)
#     grads = tf.gradients(task_lossa, list(model.weights.values()))
#
# grads_train = sess.run(grads, {model.inputa: input_train, model.labela: label_train, model.meta_lr: 0.0})
# grads_a = sess.run(grads, {model.inputa: inputa, model.labela: labela, model.meta_lr: 0.0})
# grads_train_1 = sess.run(grads, {model.inputa: input_train_val1, model.labela: label_train_val1, model.meta_lr: 0.0})
# grads_train_2 = sess.run(grads, {model.inputa: input_train_val2, model.labela: label_train_val2, model.meta_lr: 0.0})
# grads_train_3 = sess.run(grads, {model.inputa: input_train_val3, model.labela: label_train_val3, model.meta_lr: 0.0})
# # >>> idx
# # >>> Out[43]: array([15, 26, 13])
# idx = np.array([0, 1, 2])
# input_train_val3_2 = np.concatenate([input_train, input_val[:, idx[:3], :]], axis=1)
# label_train_val3_2 = np.concatenate([label_train, label_val[:, idx[:3], :]], axis=1)
# grads_train_3_2 = sess.run(grads, {model.inputa: input_train_val3_2, model.labela: label_train_val3_2, model.meta_lr: 0.0})
#
# grad1 = grads_train[-1]
# grad2 = grads_a[-1] / 12
# grad3 = grads_train_1[-1] / 4 * 3
# grad4 = grads_train_2[-1] / 5 * 3
# grad5 = grads_train_3[-1] / 6 * 3
# grad6 = grads_train_3_2[-1] / 6 * 3
# all_grads = np.vstack([grad1, grad2, grad3, grad4, grad5, grad6])
#
# from numpy import linalg as LA
# similarity = np.zeros([6,6])
# for i in range(6):
#     for j in range(6):
#         similarity[i, j] = all_grads[i].dot(all_grads[j])/ ((LA.norm(all_grads[i], 2)) * (LA.norm(all_grads[j], 2)))
# # plt.imshow(similarity, cmap='Blues')
# # plt.colorbar()
# label = [3, 36, 4, 5, 6, "3*2"]
# import pandas as pd
# data = pd.DataFrame(data=similarity, index=label, columns=label)
# import seaborn as sns
# sns.heatmap(data, cmap="Blues", annot=True)
# plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/similarity_sns.png", bbox_inches="tight", dpi=300)



# ## Connect_exp_similarity
# for exp in range(1, 10):
#     mc_iteration = 5
#     train_acc_list = np.zeros([mc_iteration, 101])
#     train_val_list = np.zeros([mc_iteration, 101])
#     test_list = np.zeros([mc_iteration, 101])
#     train_1_val_list = np.zeros([mc_iteration, 101])
#     train_2_val_list = np.zeros([mc_iteration, 101])
#     train_3_val_list = np.zeros([mc_iteration, 101])
#
#     for i in range(mc_iteration):
#         feed_init = {model.inputa: inputb, model.inputb: inputb, model.labela: labelb,
#                      model.labelb: labelb, model.meta_lr: 0.0}
#         acc0, _ = sess.run([model.total_accuracy1, model.total_accuracies2], feed_init)
#         train_acc_list[i][0] = acc0
#         train_val_list[i][0] = acc0
#         test_list[i][0] = acc0
#         train_1_val_list[i][0] = acc0
#         train_2_val_list[i][0] = acc0
#         train_3_val_list[i][0] = acc0
#
#     feed_dict_line = {model.inputa: input_train, model.inputb: inputb, model.labela: label_train,
#                       model.labelb: labelb, model.meta_lr: 0.0}
#     feed_dict_line_1 = {model.inputa: inputa, model.inputb: inputb, model.labela: labela,
#                         model.labelb: labelb, model.meta_lr: 0.0}
#     feed_dict_line_2 = {model.inputa: inputb, model.inputb: inputb, model.labela: labelb,
#                         model.labelb: labelb, model.meta_lr: 0.0}
#     idx = np.random.choice(range(val_index.shape[-1]), size=3, replace=False)
#     input_train_val1 = np.concatenate([input_train, input_val[:, idx[:1], :]], axis=1)
#     label_train_val1 = np.concatenate([label_train, label_val[:, idx[:1], :]], axis=1)
#     input_train_val2 = np.concatenate([input_train, input_val[:, idx[:2], :]], axis=1)
#     label_train_val2 = np.concatenate([label_train, label_val[:, idx[:2], :]], axis=1)
#     input_train_val3 = np.concatenate([input_train, input_val[:, idx[:3], :]], axis=1)
#     label_train_val3 = np.concatenate([label_train, label_val[:, idx[:3], :]], axis=1)
#
#     feed_dict_line_3 = {model.inputa: input_train_val1, model.inputb: inputb, model.labela: label_train_val1,
#                         model.labelb: labelb, model.meta_lr: 0.0}
#     feed_dict_line_4 = {model.inputa: input_train_val2, model.inputb: inputb, model.labela: label_train_val2,
#                         model.labelb: labelb, model.meta_lr: 0.0}
#     feed_dict_line_5 = {model.inputa: input_train_val3, model.inputb: inputb, model.labela: label_train_val3,
#                         model.labelb: labelb, model.meta_lr: 0.0}
#
#     for i in range(mc_iteration):
#         acc1, accs2 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line)
#         acc1_1, accs2_1 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_1)
#         acc1_2, accs2_2 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_2)
#         acc1_3, accs2_3 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_3)
#         acc1_4, accs2_4 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_4)
#         acc1_5, accs2_5 = sess.run([model.total_accuracy1, model.total_accuracies2], feed_dict_line_5)
#
#         train_acc_list[i][1:] = accs2
#         train_val_list[i][1:] = accs2_1
#         test_list[i][1:] = accs2_2
#         train_1_val_list[i][1:] = accs2_3
#         train_2_val_list[i][1:] = accs2_4
#         train_3_val_list[i][1:] = accs2_5
#
#     plt.figure(figsize=(6.4 * 2, 4.8 * 2))
#     plt.plot(np.mean(train_acc_list, axis=0), "-*", label="train")
#     plt.fill_between(np.arange(101), np.mean(train_acc_list, axis=0) + np.std(train_acc_list, axis=0),
#                      np.mean(train_acc_list, axis=0) - np.std(train_acc_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_val_list, axis=0), "-*", label="train+val")
#     plt.fill_between(np.arange(101), np.mean(train_val_list, axis=0) + np.std(train_val_list, axis=0),
#                      np.mean(train_val_list, axis=0) - np.std(train_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(test_list, axis=0), "-*", label="test")
#     plt.fill_between(np.arange(101), np.mean(test_list, axis=0) + np.std(test_list, axis=0),
#                      np.mean(test_list, axis=0) - np.std(test_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_1_val_list, axis=0), "-*", label="train+1_val")
#     plt.fill_between(np.arange(101), np.mean(train_1_val_list, axis=0) + np.std(train_1_val_list, axis=0),
#                      np.mean(train_1_val_list, axis=0) - np.std(train_1_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_2_val_list, axis=0), "-*", label="train+2_val")
#     plt.fill_between(np.arange(101), np.mean(train_2_val_list, axis=0) + np.std(train_2_val_list, axis=0),
#                      np.mean(train_2_val_list, axis=0) - np.std(train_2_val_list, axis=0), alpha=0.1)
#     plt.plot(np.mean(train_3_val_list, axis=0), "-*", label="train+3_val")
#     plt.fill_between(np.arange(101), np.mean(train_3_val_list, axis=0) + np.std(train_3_val_list, axis=0),
#                      np.mean(train_3_val_list, axis=0) - np.std(train_3_val_list, axis=0), alpha=0.1)
#     plt.legend()
#     plt.title("train 100 steps with")
#     # out_figure = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/" + "3way1shot_oriTask_mc{}.png".format(mc_iteration)
#     out_figure = "/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/3way1shot1kp_pairwise/" + "exp{}_3way1shot_oriTask_mc{}_{}.png".format(exp,
#         mc_iteration, np.argmax(label_train_val3[0, 3:], axis=-1).tolist())
#     plt.savefig(out_figure, bbox_inches="tight", dpi=200)
#     plt.close()
#
#     grads_train = sess.run(grads, {model.inputa: input_train, model.labela: label_train, model.meta_lr: 0.0})
#     grads_a = sess.run(grads, {model.inputa: inputa, model.labela: labela, model.meta_lr: 0.0})
#     grads_train_1 = sess.run(grads, {model.inputa: input_train_val1, model.labela: label_train_val1, model.meta_lr: 0.0})
#     grads_train_2 = sess.run(grads, {model.inputa: input_train_val2, model.labela: label_train_val2, model.meta_lr: 0.0})
#     grads_train_3 = sess.run(grads, {model.inputa: input_train_val3, model.labela: label_train_val3, model.meta_lr: 0.0})
#     # >>> idx
#     # >>> Out[43]: array([15, 26, 13])
#     idx = np.array([0, 1, 2])
#     input_train_val3_2 = np.concatenate([input_train, input_val[:, idx[:3], :]], axis=1)
#     label_train_val3_2 = np.concatenate([label_train, label_val[:, idx[:3], :]], axis=1)
#     grads_train_3_2 = sess.run(grads, {model.inputa: input_train_val3_2, model.labela: label_train_val3_2, model.meta_lr: 0.0})
#
#     grad1 = grads_train[-1]
#     grad2 = grads_a[-1] / 12
#     grad3 = grads_train_1[-1] / 4 * 3
#     grad4 = grads_train_2[-1] / 5 * 3
#     grad5 = grads_train_3[-1] / 6 * 3
#     grad6 = grads_train_3_2[-1] / 6 * 3
#     all_grads = np.vstack([grad1, grad2, grad3, grad4, grad5, grad6])
#
#     from numpy import linalg as LA
#     similarity = np.zeros([6,6])
#     for i in range(6):
#         for j in range(6):
#             similarity[i, j] = all_grads[i].dot(all_grads[j])/ ((LA.norm(all_grads[i], 2)) * (LA.norm(all_grads[j], 2)))
#     # plt.imshow(similarity, cmap='Blues')
#     # plt.colorbar()
#     label = [3, 36, 4, 5, 6, "3*2"]
#     import pandas as pd
#     data = pd.DataFrame(data=similarity, index=label, columns=label)
#     import seaborn as sns
#     sns.heatmap(data, cmap="Blues", annot=True)
#     plt.savefig("/home/cougarnet.uh.edu/pyuan2/Projects2019/maml/Figures/omniglot/3way1shot1kp_pairwise/exp{}_similarity_sns_{}.png".format(exp, np.argmax(label_train_val3[0, 3:], axis=-1).tolist()), bbox_inches="tight", dpi=300)
#     plt.close()
