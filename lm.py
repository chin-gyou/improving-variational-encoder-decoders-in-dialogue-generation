from base import *


class lm(base_enc_dec):
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode, beam_size=5):
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    """
    prev_h: word-level last state
    language model
    """

    def run(self, prev_h, input_labels):
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h, embedding)
        return h

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        h_d = tf.scan(self.run, self.labels, initializer=init_encode)
        return [1, h_d]

    def decode_bs(self, h_d):
        h = h_d[1]
        k = 0
        prev = tf.reshape(h[-1], [1, self.h_size])
        prev_d = tf.tile(prev, [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev
            inp = self.beam_search(prev_d, k)
            k += 1
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            with tf.variable_scope('encode') as enc:
                enc.reuse_variables()
                _, d_new = self.encodernet(inp, prev_d)
            prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded     
