import tensorflow as tf

init_kernel = tf.glorot_normal_initializer()

def relu(x):
    return tf.nn.relu(x)

def lrelu(x, th=0.1):
    return tf.maximum(th * x, x)

def batch_norm(x, is_training_pl, center):
    return tf.layers.batch_normalization(x, training=is_training_pl, center=center)

def dense(x, num_hidden_units, bias=True):
    return tf.layers.dense(x, num_hidden_units, kernel_initializer=init_kernel, use_bias = bias)

def trans_conv2d(x, num_filters, filter_size, st, pad='same', bias=True):
    return tf.layers.conv2d_transpose(x, num_filters, filter_size, strides = st, padding = pad, kernel_initializer=init_kernel, use_bias = bias)

def conv2d(x, num_filters, filter_size, st, pad='same', bias=True):
    return tf.layers.conv2d(x, num_filters, filter_size, strides = st, padding = pad, kernel_initializer=init_kernel, use_bias = bias)

def drop_out(x, rate, is_training_pl):
    return tf.layers.dropout(x, rate=rate, training=is_training_pl)

def max_pool(x, k, s, pad = 'SAME'):
    return tf.nn.max_pool(x, ksize= [1,k,k,1], strides = [1,s,s,1], padding= pad)

class MnistModel():
    def __init__(self, latent_dim):
        """
            latent_dim : embedding dimension
        """
        self.latent_dim = latent_dim
        self.input_shape = [None, 28, 28, 1]

    def encoder(self, inp, is_training, reuse=False):
        with tf.variable_scope('encoder', reuse=reuse):
            h1 = lrelu(batch_norm(conv2d(inp, 16, 5, 2, bias=False), is_training, center=False))
            h2 = lrelu(batch_norm(conv2d(h1, 32, 5, 2, bias=False), is_training, center=False))
            embed = dense(tf.layers.flatten(h2), self.latent_dim, bias=False)

        return embed

    def decoder(self, embed, is_training, reuse=False):
        with tf.variable_scope('decoder', reuse=reuse):
            h1 = lrelu(batch_norm(dense(embed, 7*7*32, bias=False), is_training, center=False))
            h1 = tf.reshape(h1, [-1,7,7,32])
            h2 = lrelu(batch_norm(trans_conv2d(h1, 16, 5, 2, bias=False), is_training, center=False))
            out = tf.nn.tanh(trans_conv2d(h2, 1, 5, 2, bias=False))

        return out





        









