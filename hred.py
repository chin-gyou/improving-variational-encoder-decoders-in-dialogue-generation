from base import *
from initialize import *


class hred(base_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode,beam_size = 5):
        self.c_size = c_size
        with tf.variable_scope('hier'):
            self.hiernet =  tf.contrib.rnn.GRUCell(c_size)
            print(self.context_len)
            self.init_W = tf.get_variable('Init_W',
                                          initializer=tf.random_normal([self.context_len, h_size], stddev=0.01))
            self.init_b = tf.get_variable('Init_b', initializer=tf.zeros([h_size]))
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    @property
    def context_len(self):
        return self.c_size

    """
    word-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, output initializing state
    prev_h: batch_size*h_size
    input: batch_size*embed_size
    """

    def word_level_rnn(self, prev_h, input_embedding, mask):
        with tf.variable_scope('encode', initializer=orthogonal_initializer()):
            prev_h = prev_h * mask  # mask the fist state as zero
            _, h_new = self.encodernet(input_embedding, prev_h)
            return h_new

    """
    hier-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, update state, else stay unchange
    prev_h: batch_size*c_size
    input: batch_size*h_size
    """

    def hier_level_rnn(self, prev_h, input_vec, mask):
        with tf.variable_scope('hier', initializer=orthogonal_initializer()):
            _, h_new = self.hiernet(input_vec, prev_h)
            h_masked = h_new * (1 - mask) + prev_h * mask  # update when meeting EOU
            return h_masked

    """
    decode-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, use initializing state learned from context
    prev_h: batch_size*c_size
    input: batch_size*(c_size+embed_size)
    """
    def decode_level_rnn(self, prev_h, input_h, mask):
        with tf.variable_scope('decode', initializer=orthogonal_initializer()):
            prev_h = prev_h * mask + tf.tanh(tf.matmul(input_h[:, :self.context_len], self.init_W) + self.init_b) * (
            1 - mask)  # learn initial state from context
            _, h_new = self.decodernet(input_h, prev_h)
            return h_new


    """
    prev_h[0]: word-level last state
    prev_h[1]: hier last state
    prev_h[2]: decoder last state
    hier encoder-decoder model
    """

    def run(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels[0], EOU)
        rolled_mask = self.gen_mask(input_labels[1], EOU)
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        h_s = self.hier_level_rnn(prev_h[1], h, mask)
        embedding*=mask#mark first embedding as 0
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(prev_h[2], tf.concat(axis=1, values=[h_s, embedding]), mask)
        return [h, h_s, d]

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = tf.zeros([self.batch_size, self.c_size])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, h_s, h_d = tf.scan(self.run, [self.labels, self.rolled_label],
                            initializer=[init_encode, init_hier, init_decoder])
        return [h_s,h_d]

    def decode_bs(self, h_d):
        last_h = h_d[0][-1]
        last_d = h_d[1][-1]
        k = 0
        prev = tf.reshape(last_d, [1, self.h_size])
        prev_h = tf.tile(last_h, [self.beam_size, 1])
        prev_d = tf.tile(last_d, [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev    
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                _, d_new = self.decodernet(tf.concat(axis=1, values=[prev_h, inp]), prev_d)
                prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        #decoded =  tf.reshape(self.beam_symbols, [self.beam_size, -1])
        return decoded 
