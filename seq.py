from base import *


# baseline seq2seq model
class seq_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode, beam_size=5):
        with tf.variable_scope('decode'):
            self.init_W = tf.get_variable('Init_W', initializer=tf.random_normal([h_size, h_size]))
            self.init_b = tf.get_variable('Init_b', initializer=tf.zeros([h_size]))
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    """
    prev_h[0]: word-level last state
    prev_h[1]: decoder last state
    baseline seq2seq model
    """

    def run(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels, EOU)
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h[0], embedding)
        prev_h[1] = prev_h[1] * mask + tf.tanh(tf.matmul(h, self.init_W) + self.init_b) * (
        1 - mask)  # learn initial state from context
        d = self.decode_level_rnn(prev_h[1], embedding)
        return [h, d]

    def decode_bs(self, h_d):
        last_d = h_d[1][-1]
        k = 0
        prev = tf.reshape(last_d, [1, self.h_size])
        prev_d = tf.tile(prev, [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev    
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                _, d_new = self.decodernet(inp, prev_d)
                prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        #decoded =  tf.reshape(self.beam_symbols, [self.beam_size, -1])
        return decoded 
 
    
