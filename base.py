import functools
#from tensorflow.python.ops import rnn_cell
import tensorflow as tf
from dense import *

# end token index
EOU = 2
EOT = 2

# force every function to execute only once
def exe_once(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# ensure optimise is executed at last so that the graph can be initialised
def init_final(function):
    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        function(self, *args, **kwargs)
        # In test mode, only build graph by prediction, no optimise part
        if int(self.mode) >= 1:
            self.prediction
        elif int(self.mode) ==0: # train mode, optimise
            self.optimise

    return decorator


class base_enc_dec:
    """
    labels: vocab index labels, max_length*batch_size, padding labels are 0s, 2 is eot sign, 18576 is eou sign
    length: Number of token of each dialogue batch_size
    embedding: vocab_size*embed_size
    """
    @init_final
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,beam_size=5):
        self.__dict__.update(locals())
        self.labels = tf.concat(axis=0,
                                values=[EOU * tf.ones([1, batch_size], dtype=tf.int64), labels])  # pad EOU at the first place
        self.rolled_label = tf.concat(axis=0, values=[EOU * tf.ones([1, batch_size], dtype=tf.int64), self.labels[:-1]])
        self.log_beam_probs, self.beam_path, self.output_beam_symbols, self.beam_symbols = [], [], [], []
        with tf.variable_scope('encode'):
            self.encodernet = tf.contrib.rnn.GRUCell(h_size)
            # embedding matrix
            self.embedding_W = tf.get_variable('Embedding_W', initializer=embedding)
            self.embedding_b = tf.get_variable('Embedding_b', initializer=tf.zeros([300]))
        self.output1 = Dense('decode', 300, h_size, name='output1')
        self.output2 = Dense('decode', vocab_size, 300, name='output2')
        with tf.variable_scope('decode'):
            self.decodernet = tf.contrib.rnn.GRUCell(h_size)

    """
    word-level rnn step
    takes the previous state and new input, output the new hidden state
    prev_h: batch_size*h_size
    input: batch_size*embed_size
    """

    def word_level_rnn(self, prev_h, input_embedding):
        with tf.variable_scope('encode'):
            _, h_new = self.encodernet(input_embedding, prev_h)
            return h_new

    """
    decode-level rnn step
    takes the previous state and new input, output the new hidden state
    prev_h: batch_size*c_size
    input: batch_size*(c_size+embed_size)
    """

    def decode_level_rnn(self, prev_h, input_h):
        with tf.variable_scope('decode'):
            _, h_new = self.decodernet(input_h, prev_h)
            return h_new

    """
    prev_h[0]: word-level last state
    prev_h[1]: decoder last state
    basic encoder-decoder model
    """

    def run(self, prev_h, input_labels):
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h[0], embedding)
        d = self.decode_level_rnn(prev_h[1], h)
        return [h, d]

    # turn labels into corresponding embeddings
    def embed_labels(self, input_labels):
        return tf.gather(self.embedding_W, input_labels) + self.embedding_b  #embedded inputs, batch_size*embed_size

    # generate mask for label, batch_size*1
    # value is masked as 0
    def gen_mask(self, input_labels, value):
        # mask all 0 and 2 as 0
        mask = tf.cast(tf.not_equal(input_labels, value), tf.float32)
        return tf.reshape(mask, [self.batch_size, 1])

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, h_d = tf.scan(self.run, self.labels, initializer=[init_encode, init_decoder])
        return [_, h_d]

    # return output layer
    @exe_once
    def prediction(self):
        h_d = self.scan_step()
        if self.mode == 2:
            sequences = self.decode_bs(h_d)
            return sequences
        predicted = tf.reshape(h_d[1][:-1], [-1, self.h_size])  # exclude the last prediction
        output = self.output1(predicted)  # (max_len*batch_size)*vocab_size
        output = self.output2(output)
        return output

    def decode_bs(self, h_d):
        h_e = h_d[0]
        h_d = h_d[1]
        
        k = 0
        prev = tf.reshape(h_d[-1], [1, self.h_size])
        prev_e = tf.tile(h_e[-1], [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev
            inp = self.beam_search(prev_d, k)
            k += 1
            with tf.variable_scope('encode') as enc:
                enc.reuse_variables()
                prev_e = tf.reshape(tf.gather(prev_e, self.beam_path[-1]), [self.beam_size, self.h_size])
                _, e_new = self.encodernet(inp, prev_e)
            if k == 1:
                prev_d = tf.tile(h_d[-1], [self.beam_size, 1])
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
                _, d_new = self.decodernet(e_new, prev_d)
            prev_e = e_new
            prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded 
 
    def beam_search(self, prev, k):
        output = self.output1(prev)
        output = self.output2(output)
        probs = tf.log(tf.nn.softmax(output))
        minus_probs = [0 for i in range(self.vocab_size)]
        minus_probs[1] = -1e20
#        minus_probs[2] = -1e20
        probs = probs +  minus_probs
        if k > 0:
            probs = tf.reshape(probs + self.log_beam_probs[-1],
                               [-1, self.beam_size * self.vocab_size])

        best_probs, indices = tf.nn.top_k(probs, self.beam_size)
        indices = tf.reshape(indices, [-1, 1])
        best_probs = tf.reshape(best_probs, [-1, 1])

        symbols = indices % self.vocab_size  # Which word in vocabulary.
        beam_parent = indices // self.vocab_size  # Which hypothesis it came from.

        self.beam_path.append(beam_parent)
        symbols_live = symbols
        if k > 0:
            symbols_history = tf.gather(self.output_beam_symbols[-1], beam_parent)
            symbols_live = tf.concat(axis=1,values=[tf.reshape(symbols_history,[-1,k]), tf.reshape(symbols, [-1, 1])])
        self.output_beam_symbols.append(symbols_live)
        self.beam_symbols.append(symbols)
        self.log_beam_probs.append(best_probs)
        embedded = self.embed_labels(symbols)
        emb_prev = tf.reshape(embedded, [self.beam_size, 300])
        return emb_prev
 
    @exe_once
    def cost(self):
        y_flat = tf.reshape(self.labels[1:], [-1])  # exclude the first padded label
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,labels=y_flat)
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = tf.reshape(mask * loss, tf.shape(self.labels[1:]))
        # normalized loss per example
        mean_loss_by_example = tf.reduce_sum(masked_losses, axis=0) / tf.to_float(self.length)
        return tf.reduce_mean(mean_loss_by_example)  # average loss of the batch

    @exe_once
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost, global_step=global_step)
        return global_step, train_op
