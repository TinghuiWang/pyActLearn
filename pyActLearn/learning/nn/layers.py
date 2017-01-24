import math
import tensorflow as tf
from . import variable_summary


class HiddenLayer:
    """ Typical hidden layer for Multi-layer perceptron
    User is allowed to specify the non-linearity activation function.

    Args:
        n_in (:obj:`int`): Number of input cells.
        n_out (:obj:`int`): Number of output cells.
        name (:obj:`str`): Name of the hidden layer.
        x (:class:`tensorflow.placeholder`): Input tensor.
        W (:class:`tensorflow.Variable`): Weight matrix.
        b (:class:`tensorflow.Variable`): Bias matrix.
        activation_fn: Activation function used in this hidden layer.
           Common values   :method:`tensorflow.sigmoid` for ``sigmoid`` function, :method:`tensorflow.tanh` for ``tanh``
           function, :method:`tensorflow.relu` for RELU.

    Attributes:
        n_in (:obj:`int`): Number of inputs into this layer.
        n_out (:obj:`int`): Number of outputs out of this layer.
        name (:obj:`str`): Name of the hidden layer.
        x (:class:`tensorflow.placeholder`): Tensorflow placeholder or tensor that represents the input of this layer.
        W (:class:`tensorflow.Variable`): Weight matrix of current layer.
        b (:class:`tensorflow.Variable`): Bias matrix of current layer.
        variables (:obj:`list` of :class:`tensorflow.Variable`): variables of current layer.
        logits (:obj:`tensorflow.Tensor`): Tensorflow tensor of linear logits computed in current layer.
        y (:class:`tensorflow.Tensor`): Tensorflow tensor represents the output function of this layer.
        summaries (:obj:`list`): List of Tensorflow summary buffer.
    """
    def __init__(self, n_in, n_out, name, x=None, W=None, b=None, activation_fn=tf.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        with tf.name_scope(name):
            if x is None:
                self.x = tf.placeholder(tf.float32, shape=[None, n_in])
            else:
                self.x = x
            if W is None:
                self.W = tf.Variable(
                    tf.truncated_normal(shape=[n_in, n_out],stddev=1.0/math.sqrt(float(n_in))),
                    name='weights'
                )
            else:
                self.W = W
            if b is None:
                self.b = tf.Variable(tf.zeros(shape=[n_out]), name='biases')
            else:
                self.b = b
            self.variables = [self.W, self.b]
            self.logits = tf.matmul(self.x, self.W) + self.b
            self.y = activation_fn(self.logits, name='activations')
            self.summaries = []
            self.summaries += variable_summary(self.W, tag=name + '/weights')
            self.summaries += variable_summary(self.b, tag=name + '/bias')
            self.summaries.append(tf.summary.histogram(name + '/pre_act', self.logits))
            self.summaries.append(tf.summary.histogram(name + '/act', self.y))


class SoftmaxLayer:
    """ Softmax Layer as multi-class binary classification output layer

    Parameters:
        n_in (:obj:`int`): Number of input cells.
        n_out (:obj:`int`): Number of output cells.
        name (:obj:`str`): Name of the layer.
        x (:class:`tensorflow.placeholder`): Input tensor.
        W (:class:`tensorflow.Variable`): Weight matrix.
        b (:class:`tensorflow.Variable`): Bias matrix.

    Attributes:
        n_in (:obj:`int`): Number of inputs into this layer.
        n_out (:obj:`int`): Number of outputs out of this layer.
        name (:obj:`str`): Name of the hidden layer.
        x (:class:`tensorflow.placeholder`): Tensorflow placeholder or tensor that represents the input of this layer.
        W (:class:`tensorflow.Variable`): Weight matrix of current layer.
        b (:class:`tensorflow.Variable`): Bias matrix of current layer.
        variables (:obj:`list` of :class:`tensorflow.Variable`): variables of current layer.
        logits (:obj:`tensorflow.Tensor`): Tensorflow tensor of linear logits computed in current layer.
        y (:class:`tensorflow.Tensor`): Tensorflow tensor represents the output function of this layer.
    """
    def __init__(self, n_in, n_out, name, x=None, W=None, b=None):
        self.n_in = n_in
        self.n_out = n_out
        with tf.name_scope(name):
            if x is None:
                self.x = tf.placeholder(tf.float32, shape=[None, n_in], name='input-x')
            else:
                self.x = x
            if W is None:
                self.W = tf.Variable(
                    tf.truncated_normal(shape=[n_in, n_out], stddev=1.0/math.sqrt(float(n_in))),
                    name='weights'
                )
            else:
                self.W = W
            if b is None:
                self.b = tf.Variable(tf.zeros(shape=[n_out]), name='biases')
            else:
                self.b = b
            self.variables = [self.W, self.b]
            self.logits = tf.matmul(self.x, self.W) + self.b
            self.name = name
            self.y = tf.nn.softmax(self.logits, name='softmax')
            self.summaries = []
            self.summaries += variable_summary(self.W, tag=name + '/weights')
            self.summaries += variable_summary(self.b, tag=name + '/bias')
            self.summaries.append(tf.summary.histogram(name + '/pre_act', self.logits))
            self.summaries.append(tf.summary.histogram(name + '/act', self.y))


class AutoencoderLayer(HiddenLayer):
    """Autoencoder Layer

    Auto-encoder inherits hidden layer for feed-forward calculation, and adds self encoding
    tensor for unsupervised pre-training.

    Args:
        n_in (:obj:`int`): Number of input cells.
        n_out (:obj:`int`): Number of output cells.
        name (:obj:`str`): Name of the hidden layer.
        x (:class:`tensorflow.placeholder`): Input tensor.
        W (:class:`tensorflow.Variable`): Weight matrix.
        b (:class:`tensorflow.Variable`): Bias matrix.
        shared_weights (:obj:`bool`): If weights is shared between encoding and decoding.

    Attributes:
        n_in (:obj:`int`): Number of inputs into this layer.
        n_out (:obj:`int`): Number of outputs out of this layer.
        name (:obj:`str`): Name of the hidden layer.
        x (:class:`tensorflow.placeholder`): Tensorflow placeholder or tensor that represents the input of this layer.
        W (:class:`tensorflow.Variable`): Weight matrix used in encoding.
        b (:class:`tensorflow.Variable`): Bias matrix in encoding.
        W_prime (:obj:`tensorflow.Tensor`): Weight matrix used in self-decoding process. If weights are shared, it
            equals to transpose of encoding weight matrix.
        b_prime (:obj:`tensorflow.Tensor`): Bias matrix used in self-decoding process.
        variables (:obj:`list` of :class:`tensorflow.Variable`): variables of current layer.
        logits (:obj:`tensorflow.Tensor`): Tensorflow tensor of linear logits computed after encoding.

        y (:class:`tensorflow.Tensor`): Tensorflow tensor represents the output function of this layer.
        summaries (:obj:`list`): List of Tensorflow summary buffer.
    """
    def __init__(self, n_in, n_out, name, x=None, W=None, b=None, shared_weights=True):
        super().__init__(n_in, n_out, name, x, W, b, tf.sigmoid)
        self.b_prime = tf.Variable(tf.zeros(shape=[n_in]), name='biases_prime')
        self.variables.append(self.b_prime)
        if shared_weights:
            self.W_prime = tf.transpose(self.W)
        else:
            self.W_prime = tf.Variable(
                tf.truncated_normal(shape=[n_out, n_in], stddev=1.0 / math.sqrt(float(n_in))),
                name='weights_prime'
            )
            self.variables.append(self.W_prime)
        self.encode_logit = tf.matmul(self.y, self.W_prime) + self.b_prime
        self.encode = tf.sigmoid(self.encode_logit)
        self.encode_loss = tf.reduce_mean(tf.pow(self.x - self.encode, 2))
        self.summaries.append(tf.summary.scalar(name+'/ae_rmse', self.encode_loss))
        self.merged = tf.summary.merge(self.summaries)
