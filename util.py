from dataproducer import *
from ws_vhred import *
import os
import pickle
import sys
import cPickle
# global for epoch
epoch = 1


# restore trainable variables from a checkpoint file, excluede some specific variables
def restore_trainable(sess, chkpt):
    #trainable = {v.op.name: v for v in tf.trainable_variables()}
    trainable = {v.op.name: v for v in tf.get_collection(tf.GraphKeys.VARIABLES)}
    print('trainable:', trainable)
    # exclude = set()
    exclude = {'hier/Init_W', 'hier/Init_b', 'decode/GRUCell/Candidate/Linear/Matrix',
               'decode/GRUCell/Candidate/Linear/Bias', 'decode/GRUCell/Gates/Linear/Matrix',
               'decode/GRUCell/Gates/Linear/Bias'}  # excluded variables
    trainable = {key: value for key, value in trainable.items()}
    reader = tf.train.NewCheckpointReader(chkpt)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # only restore variables existed in the checkpoint file
    variables_to_restore = {key: value for key, value in trainable.items() if key in var_to_shape_map}
    print('to_restore:', variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, chkpt)


# add variables to summary
def variable_summaries(var, name):
    """Attach the mean summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)


def build_graph(options, path):
    # get input file list and word vectors
    fileList = os.listdir(path)
    if fileList == []:
        print('\nNo input file found!')
        sys.exit()
    else:
        try:
            print('Loading saved embeddings for tokens...')
            with open(options.wvec_mat, 'rb') as f:
                word_vecs = pickle.load(f)
        except IOError:
            raise Exception('[ERROR]Word Vector File not found!')
    # get input data
    vocab_size, e_size = word_vecs.shape
    fileList = [os.path.join(path, item) for item in fileList]
    dataproducer = data_producer(fileList, int(options.num_epochs))
    labels, length = dataproducer.batch_data(int(options.batch_size))
    #labels, obj, length = dataproducer.batch_data(int(options.batch_size))
    # build model and graph
    model = vhred(labels, length, int(options.h_size), int(options.c_size), int(options.z_size), vocab_size, word_vecs,
                 int(options.batch_size), float(options.lr), int(options.mode),epoch=epoch)
    #model = hred(labels, length, int(options.h_size), int(options.c_size), vocab_size, word_vecs,
    #             int(options.batch_size), float(options.lr), int(options.mode))
    return model


def train(options, start=True):
    global epoch
    options.mode = 0
    model = build_graph(options, options.input_path)
    variable_summaries(model.cost, 'loss')
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sum_writer = tf.summary.FileWriter(options.tboard_dir, graph=sess.graph)
    # restore from a check point
    if options.load_chkpt:
        print('Loading saved variables from checkpoint file to graph')
        sess.run(init_op)
        if start:
            restore_trainable(sess, options.load_chkpt)
        else:
            saver.restore(sess, options.load_chkpt)
        print('Resume Training...')
    else:
        sess.run(init_op)
        print('Start Training...')
    try:
        #N_EXAMPLES = 1280
        N_EXAMPLES = 21542
        steps_per_epoch = N_EXAMPLES // int(options.batch_size)
        while not coord.should_stop():
          #  labels, obj = sess.run([model.labels, model.obj])
            #print labels, obj
            batch_loss, training,length,num_eos, summary = sess.run([model.cost, model.optimise,model.length, model.num_eos, merged])
            #print length
            #length = 1
            length = np.sum(length)
            train_step = training[0]
            print('[Epoch:%d][size:%d]Mini-Batches run :%d  Loss :%f  KL:%f  Scale:%f  avgloss:%f  avgkl:%f  avgrloss:%f' % (
                epoch, int(options.batch_size), train_step, batch_loss[0], batch_loss[1], training[2],batch_loss[0]/length, batch_loss[1]/num_eos, batch_loss[2]/num_eos))
            if train_step % steps_per_epoch == 0:
                break
    except tf.errors.OutOfRangeError:
        print('Training Complete...')
    finally:
        print('[Epoch %d] training finished!' % (epoch))
        print('Saving checkpoint...Model saved at :', options.save_path)
        saver.save(sess, os.path.join(options.save_path, '-epoch' + str(epoch)))
        coord.request_stop()
        coord.join(threads)
        sess.close()
        tf.reset_default_graph()


def train_with_validate(options):
    global epoch
    extra_num = 0
    best_epoch = 1
    min_validate_loss = 1000
    train(options)
    while epoch<70:
        # validate
        options.load_chkpt = os.path.join(options.save_path, '-epoch' + str(epoch))
        loss, kl,nll,rloss = test_loss(options)
        with open('output.txt', 'a') as f:
            f.write('[Epoch:%d]' % (epoch))
            f.write(' Loss : %f' % (loss))
            f.write(' Kl : %f\n' % (kl))
            f.write(' Nll : %f' % (nll))
            f.write(' Rloss : %f\n' % (rloss))
            
        # loss decreases
#        if current_validate_loss < min_validate_loss:
#            best_epoch = epoch
#            extra_num = 0
#            min_validate_loss = current_validate_loss
#        else:
#            # loss increases but less than 3 times
#            if extra_num < 3:
#                extra_num += 1
            # loss increases 3 times, stop
#            else:
#                print("Validation loss no longer decrease! Stop training!")
#                print("Best training epoch : %d" % (best_epoch))
#                break
        epoch += 1
        train(options)

def test_loss(options):
    options.mode = 1
    model = build_graph(options, options.validation_dir)
    model.mode=1
    model.rate=1
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    if options.load_chkpt:
        print('Loading saved variables from checkpoint file to graph')
        saver.restore(sess, options.load_chkpt)
        print('mode: ',model.mode)
        print('Starting test loss...')
        loss,kl,nll, rloss = evaluate(sess, model, int(options.batch_size))
        print('loss: %f\tkl: %f\tnll: %f\trloss:%f' % (loss,kl,nll, rloss))
    else:
        print('Forget checkpoint file.')
    return loss,kl,nll, rloss 


"""
evaluate a model with filedir and return the mean batch_loss
filedir: directory for evaluated tfrecords
"""


def evaluate(sess, model, batch_size):
    step_evaluate = 1951 // batch_size
    #step_evaluate = 1280/batch_size
    coord = tf.train.Coordinator()
    step = 0
    total_loss, total_loss_all = 0, 0
    avg1, avg2, avg3, avg4, avg5 = 0,0,0,0,0
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    while not coord.should_stop():
        batch_loss = sess.run(model.cost)
        batch_loss,length,num_eos = sess.run([model.cost,tf.reduce_sum(model.length), model.num_eos])
        step += 1
        avg1 += batch_loss[0]/length
        avg2 += batch_loss[1]/num_eos
        avg3 += batch_loss[2]/num_eos
        avg4+=(batch_loss[0]/128+batch_loss[1]/128)
        if step % 10 == 0:
            print('[Test][size:%d]Mini-Batches run : %d\tmeam loss: %f\tmean kl: %f\tnll: %f\tmean rloss:%f' % (
                batch_size, step, avg1 / step, avg2 / step, avg4/step, avg3/step))
        if step == step_evaluate:
            break

    coord.request_stop()
    coord.join(threads)
    tf.reset_default_graph()
    return avg1 / step, avg2 / step, avg4/step, avg3/step


def loss_calculation(output, y_flat, length):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y_flat) 
    mask = tf.sign(tf.to_float(y_flat))
    masked_losses = mask * loss
    mean_loss_by_example = tf.reduce_sum(masked_losses, axis=0) / tf.to_float(tf.shape(y_flat)[0])
    return tf.reduce_mean(mean_loss_by_example)

def generatez(options):
    options.batch_size = 1
    fileList = os.listdir(options.chat_test_path)
    fileList = [os.path.join(options.chat_test_path, item) for item in fileList]
    with open(options.wvec_dict, 'rb') as f:
        dics = pickle.load(f)
    # i+1, 0 stand for padding elements
    word_index_dic = {w: int(i + 1) for w, i, _, _ in dics}
    index_word_dic = {int(i + 1): w for w, i, _, _ in dics}
    z1,z2,z3,z4,z5,z6= [],[],[],[],[],[]
    act4 = 0
    # build model and graph
    labels = tf.placeholder(tf.int64, [None, 1])
    length = tf.placeholder(tf.int64, [1])
    model = vhred(labels, length, int(options.h_size), int(options.c_size),int(options.z_size), 20003, tf.zeros([20003, 300]),
                   int(options.batch_size), float(options.lr), int(options.mode))
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, options.load_chkpt)
    try:
       with open(options.chat_test_path + "/dialogues_text.txt", 'r') as f, open(options.chat_test_path + "/dialogues_act.txt", 'r') as f1:
                lines = f.readlines()[:10000]
                acts = f1.readlines()[:10000]
                acts = [a.split()[-1] for a in acts]
                # one test
                for line,act in zip(lines, acts):
                   # if int(act) == 0:
                   #     continue
                   # if int(act) == 4:
                   #     if act4 > 300:
                   #         continue
                   #     act4 += 1
                    labels_data = line.split()
                    length_data = [len(labels_data)]
                    labels_data = [[word_index_dic.get(i, 1)] for i in labels_data]
                    dec = sess.run(model.prediction, feed_dict={labels: labels_data, length: length_data})
                    #seq = ' '.join([index_word_dic[i] for i in dec[0]]) + '\n'
                    print(dec[0])
                    if int(act ) == 1:
                        z1.append(dec[0])
                    if int(act ) == 2:
                        z2.append(dec[0])
                    if int(act ) == 3:
                        z3.append(dec[0])
                    if int(act ) == 4:
                        z4.append(dec[0])
                    #if int(act ) == 5:
                    #    z5.append(dec[0])
                    #if int(act ) == 6:
                    #    z6.append(dec[0])

    finally:
        with open('z1.pkl', 'w') as f:
            cPickle.dump(z1, f, protocol=cPickle.HIGHEST_PROTOCOL)
        with open('z2.pkl', 'w') as f:
            cPickle.dump(z2, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open('z3.pkl', 'w') as f:
            cPickle.dump(z3, f, protocol=cPickle.HIGHEST_PROTOCOL)

        with open('z4.pkl', 'w') as f:
            cPickle.dump(z4, f, protocol=cPickle.HIGHEST_PROTOCOL)
      #  with open('ez5.pkl', 'w') as f:
      #      cPickle.dump(z5, f, protocol=cPickle.HIGHEST_PROTOCOL)
      #  with open('ez6.pkl', 'w') as f:
      #      cPickle.dump(z6, f, protocol=cPickle.HIGHEST_PROTOCOL)





def chat(options):
    options.batch_size = 1
    fileList = os.listdir(options.chat_test_path)
    fileList = [os.path.join(options.chat_test_path, item) for item in fileList]
    with open(options.wvec_dict, 'rb') as f:
        dics = pickle.load(f)
    # i+1, 0 stand for padding elements
    word_index_dic = {w: int(i + 1) for w, i, _, _ in dics}
    index_word_dic = {int(i + 1): w for w, i, _, _ in dics}
    r = []
    # build model and graph
    labels = tf.placeholder(tf.int64, [None, 1])
    length = tf.placeholder(tf.int64, [1])
    model = vhred(labels, length, int(options.h_size), int(options.c_size),int(options.z_size), 20003, tf.zeros([20003, 300]),
                   int(options.batch_size), float(options.lr), int(options.mode))
    config = tf.ConfigProto(allow_soft_placement=False)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, options.load_chkpt)
    try:
       with open(options.chat_test_path + "/dialogues_test_contexts.txt", 'r') as f:
                lines = f.readlines()
                # one test
                for line in lines:
                    labels_data = line.split()
                    length_data = [len(labels_data)]
                    labels_data = [[word_index_dic.get(i, 1)] for i in labels_data]
                    dec = sess.run(model.prediction, feed_dict={labels: labels_data, length: length_data})
                    seq = ' '.join([index_word_dic[i] for i in dec[0]]) + '\n'
                    print(seq)
                    r.append(seq)
    finally:
        with open('vhred.txt', 'w') as f:
            f.writelines(r)

