import numpy as np
import tensorflow as tf
tf.set_random_seed(2018)


def batches(l, n):
    """Yield successive n-sized batches from l, the last batch is the left indexes."""
    for i in range(0, l, n):
        yield range(i, min(l, i + n))


class DAE(object):
    def __init__(self, sess, input_dim_list=[784, 400], learning_rate=0.15):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        ## Encoders parameters
        for i in range(len(input_dim_list) - 1):
            init_max_value = np.sqrt(6. / (self.dim_list[i] + self.dim_list[i + 1]))
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i], self.dim_list[i + 1]],
                                                             np.negative(init_max_value), init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i + 1]], -0.1, 0.1)))
        ## Decoders parameters
        for i in range(len(input_dim_list) - 2, -1, -1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]], -0.1, 0.1)))
        ## Placeholder for input
        self.input_x = tf.placeholder(tf.float32, [None, self.dim_list[0]])
        ## coding graph :
        last_layer = self.input_x
        for weight, bias in zip(self.W_list, self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            last_layer = hidden
        self.hidden = hidden
        ## decode graph:
        for weight, bias in zip(reversed(self.W_list), self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer, tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer

        self.cost = tf.reduce_mean(tf.square(self.input_x - self.recon))
        # self.cost = tf.losses.log_loss(self.recon, self.input_x)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        sess.run(tf.global_variables_initializer())

    def fit(self, X, sess, iteration=200, batch_size=50, init=False, verbose=False):
        assert X.shape[1] == self.dim_list[0]
        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]
        for i in range(iteration):

            for one_batch in batches(sample_size, batch_size):
                sess.run([self.train_step], feed_dict={self.input_x: X[one_batch]})

            if verbose and (i + 1) % 5 == 0:
                e = self.cost.eval(session=sess, feed_dict={self.input_x: X[one_batch]})
                print("    iteration : ", i + 1, ", cost : ", e)

    def transform(self, X, sess):
        return self.hidden.eval(session=sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session=sess, feed_dict={self.input_x: X})

def l21shrink(epsilon, x):
    """
    auther : Chong Zhou
    date : 10/20/2016
    Args:
        epsilon: the shrinkage parameter
        x: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
        elif norm[i] < -epsilon:
            for j in range(x.shape[0]):
                output[j,i] = x[j,i] + epsilon * x[j,i] / norm[i]
        else:
            output[:,i] = 0.
    return output


class RobustL21Autoencoder(object):
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016

    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    Improve:
        1. fix the 0-cost bugs

    """

    def __init__(self, sess, layers_sizes, lambda_=1.0, error=1.0e-8, learning_rate=0.15):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors = []
        self.AE = DAE(sess=sess, input_dim_list=self.layers_sizes, learning_rate=learning_rate)

    def fit(self, X, sess, inner_iteration=50,
            iteration=20, batch_size=133, re_init=False, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        ##LS0 = self.L + self.S
        ## To estimate the size of input X
        if verbose:
            print("X shape: ", X.shape)
            print("L shape: ", self.L.shape)
            print("S shape: ", self.S.shape)

        for it in range(iteration):
            if verbose:
                print("Out iteration: ", it + 1)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder
            self.AE.fit(self.L, sess=sess,
                        iteration=inner_iteration,
                        batch_size=batch_size,
                        init=re_init,
                        verbose=verbose)
            ## get optmized L
            self.L = self.AE.getRecon(X=self.L, sess=sess)
            ## alternating project, now project to S and shrink S
            self.S = l21shrink(self.lambda_, (X - self.L).T).T
        return self.L, self.S

    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X=L, sess=sess)

    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess=sess)

    def predict(self, X, sess):
        L = self.AE.getRecon(X=X, sess=sess)

        S = l21shrink(self.lambda_, (X - L).T).T

        return L, S
