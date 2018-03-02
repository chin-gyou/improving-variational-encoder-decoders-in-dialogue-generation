import tensorflow as tf

# Fully-connected layer
class Dense():
    def __init__(self, scope="dense_layer", size=512, xsize=512, dropout=1.,
                 nonlinearity=tf.identity, name='dense'):
        self.__dict__.update(locals())
        with tf.name_scope(scope):
            self.w, self.b = self.origin_init(xsize, size, name)
            self.w = tf.nn.dropout(self.w, dropout)

    # x: N1...Nn*x_size
    # return: N1...Nn*self.size
    # if reshape, return shape the same as x
    def __call__(self, x, reshape=False):
        with tf.name_scope(self.scope):
            reshaped = tf.reshape(x, [-1, self.xsize])
            result = self.nonlinearity(tf.matmul(reshaped, self.w) + self.b)
            if reshape:
                dims = tf.slice(tf.shape(x), [0], [tf.rank(x) - 1])
                new_shape = tf.concat(axis=0, values=[dims, [self.size]])
                return tf.reshape(result, new_shape)
            return tf.reshape(result, [-1, self.size])

    @staticmethod
    def origin_init(fan_in, fan_out, name):
        initial_w = tf.random_normal([fan_in, fan_out], stddev=0.01)
        initial_b = tf.zeros([fan_out])
        return (tf.Variable(initial_w, trainable=True, name=name + "weights"),
                tf.Variable(initial_b, trainable=True, name=name + "biases"))


    # Helper to initialize weights and biases, via He's adaptation
    # of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    @staticmethod
    def wbVars(fan_in, fan_out, name):

        stddev = tf.cast((2.0 / fan_in) ** 0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name=name + "weights"),
                tf.Variable(initial_b, trainable=True, name=name + "biases"))
