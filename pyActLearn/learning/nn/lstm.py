import os
import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfcontrib

from pyActLearn.learning.nn import variable_summary
from .layers import HiddenLayer, SoftmaxLayer
from .injectors import BatchSequenceInjector
from .criterion import MonitorBased, ConstIterations

logger = logging.getLogger(__name__)


class LSTM_Legacy:
    """Basic Single Layer Long-Short-Term Memory
    """
    def __init__(self, num_features, num_classes, num_units, num_steps, optimizer=None):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_units = num_units
        self.summaries = []
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, num_steps, num_features], name='input_x')
            self.init_state = tf.placeholder(tf.float32, shape=[None, 2 * num_units], name='init_state')
            self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        # Input Hidden Layer - Need to unroll num_steps and apply W/b
        hidden_x = tf.reshape(tf.transpose(self.x, [1, 0, 2]), [-1, num_features])
        self.hidden_layer = HiddenLayer(num_features, num_units, 'Hidden', x=hidden_x)
        # Output of the hidden layer needs to be split to be used with RNN
        hidden_y = tf.split(axis=0, num_or_size_splits=int(num_steps), value=self.hidden_layer.y)
        # Apply RNN
        self.cell = tfcontrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=False)
        outputs, states = tfcontrib.rnn.static_rnn(self.cell, hidden_y, initial_state=self.init_state)
        self.last_state = states[-1]
        # Output Softmax Layer
        self.output_layer = SoftmaxLayer(num_units, num_classes, 'SoftmaxLayer', x=outputs[-1])
        # Predicted Probability
        self.y = self.output_layer.y
        self.y_class = tf.argmax(self.y, 1)
        # Softmax Cross-Entropy Loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer.logits, labels=self.y_,
                                                    name='SoftmaxCrossEntropy')
        )
        # Setup Optimizer
        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        else:
            self.optimizer = optimizer
        # Evaluation
        self.correct_prediction = tf.equal(self.y_class, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        # Fit Step
        with tf.name_scope('train'):
            self.fit_step = self.optimizer.minimize(self.loss)
        # Setup Summaries
        self.summaries += self.hidden_layer.summaries
        self.summaries += self.output_layer.summaries
        self.summaries.append(tf.summary.scalar('cross_entropy', self.loss))
        self.summaries.append(tf.summary.scalar('accuracy', self.accuracy))
        self.merged = tf.summary.merge(self.summaries)
        self.sess = None

    def fit(self, x, y, batch_size=100, iter_num=100, summaries_dir=None, summary_interval=10,
            test_x=None, test_y=None, session=None, criterion='const_iteration'):
        """Fit the model to the dataset

        Args:
            x (:obj:`numpy.ndarray`): Input features of shape (num_samples, num_features).
            y (:obj:`numpy.ndarray`): Corresponding Labels of shape (num_samples) for binary classification,
                or (num_samples, num_classes) for multi-class classification.
            batch_size (:obj:`int`): Batch size used in gradient descent.
            iter_num (:obj:`int`): Number of training iterations for const iterations, step depth for monitor based
                stopping criterion.
            summaries_dir (:obj:`str`): Path of the directory to store summaries and saved values.
            summary_interval (:obj:`int`): The step interval to export variable summaries.
            test_x (:obj:`numpy.ndarray`): Test feature array used for monitoring training progress.
            test_y (:obj:`numpy.ndarray): Test label array used for monitoring training progress.
            session (:obj:`tensorflow.Session`): Session to run training functions.
            criterion (:obj:`str`): Stopping criteria. 'const_iterations' or 'monitor_based'
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        if summaries_dir is not None:
            train_writer = tf.summary.FileWriter(summaries_dir + '/train')
            test_writer = tf.summary.FileWriter(summaries_dir + '/test')
        session.run(tf.global_variables_initializer())
        # Get Stopping Criterion
        if criterion == 'const_iteration':
            _criterion = ConstIterations(num_iters=iter_num)
        elif criterion == 'monitor_based':
            num_samples = x.shape[0]
            valid_set_len = int(1/5 * num_samples)
            valid_x = x[num_samples-valid_set_len:num_samples, :]
            valid_y = y[num_samples-valid_set_len:num_samples, :]
            x = x[0:num_samples-valid_set_len, :]
            y = y[0:num_samples-valid_set_len, :]
            _criterion = MonitorBased(n_steps=iter_num,
                                      monitor_fn=self.predict_accuracy,
                                      monitor_fn_args=(valid_x, valid_y[self.num_steps:, :]),
                                      save_fn=tf.train.Saver().save,
                                      save_fn_args=(session, summaries_dir + '/best.ckpt'))
        else:
            logger.error('Wrong criterion %s specified.' % criterion)
            return
        # Setup batch injector
        injector = BatchSequenceInjector(data_x=x, data_y=y, batch_size=batch_size, seq_len=self.num_steps)
        # Train/Test sequence for brief reporting of accuracy and loss
        train_seq_x, train_seq_y = BatchSequenceInjector.to_sequence(
            self.num_steps, x, y, start=0, end=2000
        )
        if (test_x is not None) and (test_y is not None):
            test_seq_x, test_seq_y = BatchSequenceInjector.to_sequence(
                self.num_steps, test_x, test_y, start=0, end=2000
            )
        # Iteration Starts
        i = 0
        while _criterion.continue_learning():
            batch_x, batch_y = injector.next_batch()
            if summaries_dir is not None and (i % summary_interval == 0):
                summary, loss, accuracy = session.run(
                    [self.merged, self.loss, self.accuracy],
                    feed_dict={self.x: train_seq_x, self.y_: train_seq_y,
                               self.init_state: np.zeros((train_seq_x.shape[0], 2 * self.num_units))}
                )
                train_writer.add_summary(summary, i)
                logger.info('Step %d, train_set accuracy %g, loss %g' % (i, accuracy, loss))
                if (test_x is not None) and (test_y is not None):
                    merged, accuracy = session.run(
                        [self.merged, self.accuracy],
                        feed_dict={self.x: test_seq_x, self.y_: test_seq_y,
                                   self.init_state: np.zeros((test_seq_x.shape[0], 2*self.num_units))})
                    test_writer.add_summary(merged, i)
                    logger.info('test_set accuracy %g' % accuracy)
            loss, accuracy, _ = session.run(
                [self.loss, self.accuracy, self.fit_step],
                feed_dict={self.x: batch_x, self.y_: batch_y,
                           self.init_state: np.zeros((batch_x.shape[0], 2 * self.num_units))})
            i += 1
        # Finish Iteration
        if criterion == 'monitor_based':
            tf.train.Saver().restore(session, os.path.join(summaries_dir, 'best.ckpt'))
        logger.debug('Total Epoch: %d, current batch %d', injector.num_epochs, injector.cur_batch)

    def predict_proba(self, x, session=None, batch_size=500):
        """Predict probability (Softmax)
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        injector = BatchSequenceInjector(batch_size=batch_size, data_x=x, seq_len=self.num_steps)
        injector.reset()
        result = None
        while injector.num_epochs == 0:
            batch_x = injector.next_batch()
            batch_y = session.run(self.y,
                                  feed_dict={self.x: batch_x,
                                             self.init_state: np.zeros((batch_x.shape[0], 2 * self.num_units))})
            if result is None:
                result = batch_y
            else:
                result = np.concatenate((result, batch_y), axis=0)
        return result

    def predict(self, x, session=None, batch_size=500):
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        injector = BatchSequenceInjector(batch_size=batch_size, data_x=x, seq_len=self.num_steps)
        injector.reset()
        result = None
        while injector.num_epochs == 0:
            batch_x = injector.next_batch()
            batch_y = session.run(self.y_class,
                                  feed_dict={self.x: batch_x,
                                             self.init_state: np.zeros((batch_x.shape[0], 2 * self.num_units))})
            if result is None:
                result = batch_y
            else:
                result = np.concatenate((result, batch_y), axis=0)
        return result

    def predict_accuracy(self, x, y, session=None):
        """Get Accuracy given feature array and corresponding labels
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        predict = self.predict(x, session=session)
        accuracy = np.sum(predict == y.argmax(y.ndim - 1)) / float(y.shape[0])
        return accuracy


class LSTM:
    """Single Layer LSTM Implementation

    In this new implementation, state_is_tuple is disabled to suppress the "deprecated" warning and
    performance improvement. The static unrolling of the RNN is replaced with dynamic unrolling.
    As a result, no batch injector is needed for prediction.
    
    Args:
        num_features (:obj:`int`): Number of input features.
        num_classes (:obj:`int`): Number of target classes.
        num_hidden (:obj:`int`): Number of units in the input hidden layer.
        num_units (:obj:`int`): Number of units in the RNN layer.
        
    Attributes:
        num_features (:obj:`int`): Number of input features.
        num_classes (:obj:`int`): Number of target classes.
        num_hidden (:obj:`int`): Number of units in the input hidden layer.
        num_units (:obj:`int`): Number of units in the RNN layer.
        summaries (:obj:`list`): List of tensorflow summaries to be displayed on tensorboard.
        x (:obj:`tf.Tensor`): Input tensor of size [num_batches, length, num_features]
        length (:obj:`tf.Tensor`): 1D length array (int) of size [num_batches, 1] for the length of each batch data.
        init_state (:obj:`tf.Tensor`): Initial states. 2D tensor (float) of size [num_batches, 2*num_units].
        y_ (:obj:`tf.Tensor`): Ground Truth of size [num_batches, length, num_classes].
    """
    def __init__(self, num_features, num_classes, num_hidden, num_units, num_skip=0, graph=None, optimizer=None):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_units = num_units
        self.num_skip = num_skip
        self.summaries = []
        if graph is None:
            self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs
            with tf.name_scope('input'):
                # Input tensor X, shape: [batch, length, features]
                self.x = tf.placeholder(tf.float32, shape=[None, None, num_features], name='input_x')
                # Length, shape: [batch, length]
                self.length = tf.placeholder(tf.float32, shape=[None, ], name='input_x_length')
                # Initial states (as tupples), shape: [batch, units]
                self.initial_state_c = tf.placeholder(tf.float32, shape=[None, num_units], name='initial_state_c')
                self.initial_state_h = tf.placeholder(tf.float32, shape=[None, num_units], name='initial_state_h')
                # Targets, shape: [batch, length, classes]
                self.y_ = tf.placeholder(tf.float32, shape=[None, None, num_classes], name='targets')
            # Input hidden layer with num_hidden units
            with tf.name_scope('input_layer'):
                self.input_W = tf.Variable(
                    tf.truncated_normal(
                        shape=[num_features, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden))),
                        name='weights')
                self.input_b = tf.Variable(tf.zeros(shape=[num_hidden]), name='bias')

                def hidden_fn(slice):
                    return tf.nn.sigmoid(tf.matmul(slice, self.input_W) + self.input_b)
                # Activation of hidden layer, shape: [batch, length, num_hidden]
                self.hidden_y = tf.map_fn(hidden_fn, self.x)
            # Recursive Layer (RNN)
            with tf.name_scope('rnn'):
                # Apply RNN
                self.cell = tfcontrib.rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
                # rnn outputs, shape: [batch, length, num_units]
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                    self.cell, self.hidden_y, sequence_length=self.length,
                    initial_state=tfcontrib.rnn.LSTMStateTuple(self.initial_state_c, self.initial_state_h))
            # Apply Softmax Layer to all outputs in all batches
            with tf.name_scope('output_layer'):
                self.output_W = tf.Variable(
                    tf.truncated_normal(shape=[num_units, num_classes], stddev=1.0/math.sqrt(float(num_units))),
                    name='weights'
                )
                self.output_b = tf.Variable(tf.zeros(shape=[num_classes]), name='biases')

                def out_mult_fn(slice):
                    return tf.matmul(slice, self.output_W) + self.output_b

                def out_softmax_fn(slice):
                    return tf.nn.softmax(slice)

                def out_class_fn(slice):
                    return tf.argmax(slice, axis=1)

                def out_softmax_entropy(params):
                    logits, labels = params
                    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

                # self.logit_outputs is a tensor of shape [batch, length, num_classes]
                self.logit_outputs = tf.map_fn(out_mult_fn, rnn_outputs)
                # self.softmax_outputs applies softmax to logit_outputs as a tensor of shape
                # [batch, length, num_classes]
                self.softmax_outputs = tf.map_fn(out_softmax_fn, self.logit_outputs)
            # Probability output y, shape: [batch, length-num_skip, num_classes]
            self.y = self.softmax_outputs[:, num_skip:, :]
            self.y_class = tf.map_fn(out_class_fn, self.y, dtype=tf.int64)

            # Acciracy
            def accuracy_fn(params):
                prediction, truth = params
                return tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(truth, 1)), tf.float32))

            self.accuracy_outputs = tf.map_fn(accuracy_fn, (self.y_class, self.y_[:, num_skip:, :]), dtype=tf.float32)
            self.accuracy = tf.reduce_mean(self.accuracy_outputs)
            # self.class_outputs gets the class label for each item in sequence as a tensor of shape
            # [batch_size, max_time, 1]
            self.entropy_outputs = tf.map_fn(out_softmax_entropy,
                                             (self.logit_outputs[:, num_skip:, :], self.y_[:, num_skip:, :]),
                                             dtype=tf.float32)
            # Softmax Cross-Entropy Loss
            self.loss = tf.reduce_mean(self.entropy_outputs)
            # Setup Optimizer
            if optimizer is None:
                self.optimizer = tf.train.AdamOptimizer()
            else:
                self.optimizer = optimizer
            # Fit Step
            with tf.name_scope('train'):
                self.fit_step = self.optimizer.minimize(self.loss)
            # Setup Summaries
            self.summaries.append(variable_summary(self.input_W, tag='input_layer/weights'))
            self.summaries.append(variable_summary(self.input_b, tag='input_layer/biases'))
            self.summaries.append(variable_summary(self.output_W, tag='output_layer/weights'))
            self.summaries.append(variable_summary(self.output_b, tag='output_layer/biases'))
            self.summaries.append(tf.summary.scalar('cross_entropy', self.loss))
            self.summaries.append(tf.summary.scalar('accuracy', self.accuracy))
            self.merged = tf.summary.merge(self.summaries)
            self.init_op = tf.global_variables_initializer()
            self.sess = None

    def fit(self, x, y, length, batch_size=100, iter_num=100, summaries_dir=None, summary_interval=100,
            test_x=None, test_y=None, session=None, criterion='const_iteration', reintialize=True):
        """Fit the model to the dataset

        Args:
            x (:obj:`numpy.ndarray`): Input features x, shape: [num_samples, num_features].
            y (:obj:`numpy.ndarray`): Corresponding Labels of shape (num_samples) for binary classification,
                or (num_samples, num_classes) for multi-class classification.
            length (:obj:`int`): Length of each batch (needs to be greater than self.num_skip.
            batch_size (:obj:`int`): Batch size used in gradient descent.
            iter_num (:obj:`int`): Number of training iterations for const iterations, step depth for monitor based
                stopping criterion.
            summaries_dir (:obj:`str`): Path of the directory to store summaries and saved values.
            summary_interval (:obj:`int`): The step interval to export variable summaries.
            test_x (:obj:`numpy.ndarray`): Test feature array used for monitoring training progress.
            test_y (:obj:`numpy.ndarray): Test label array used for monitoring training progress.
            session (:obj:`tensorflow.Session`): Session to run training functions.
            criterion (:obj:`str`): Stopping criteria. 'const_iterations' or 'monitor_based'
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        if summaries_dir is not None:
            train_writer = tf.summary.FileWriter(summaries_dir + '/train')
            test_writer = tf.summary.FileWriter(summaries_dir + '/test')
            valid_writer = tf.summary.FileWriter(summaries_dir + '/valid')
        else:
            train_writer = None
            test_writer = None
            valid_writer = None
        if reintialize:
            session.run(self.init_op)
        with self.graph.as_default():
            saver = tf.train.Saver()
        num_samples = x.shape[0]
        # Get Stopping Criterion
        if criterion == 'const_iteration':
            _criterion = ConstIterations(num_iters=iter_num)
        elif criterion == 'monitor_based':
            valid_set_start = int(4/5 * (num_samples - self.num_skip))
            valid_x = x[valid_set_start:num_samples, :]
            valid_y = y[valid_set_start:num_samples, :]
            x = x[0:valid_set_start + self.num_skip, :]
            y = y[0:valid_set_start + self.num_skip, :]
            _criterion = MonitorBased(n_steps=iter_num,
                                      monitor_fn=self.predict_accuracy,
                                      monitor_fn_args=(valid_x, valid_y),
                                      save_fn=saver.save,
                                      save_fn_args=(session, summaries_dir + '/best.ckpt'))
        else:
            logger.error('Wrong criterion %s specified.' % criterion)
            return
        # Setup batch injector
        injector = BatchSequenceInjector(data_x=x, data_y=y, batch_size=batch_size, length=self.num_skip + length,
                                         with_seq=True)
        # Iteration Starts
        i = 0
        while _criterion.continue_learning():
            # Learning
            batch_x, batch_y, batch_length = injector.next_batch(skip=50)
            loss, accuracy, _ = session.run(
                [self.loss, self.accuracy, self.fit_step],
                feed_dict={self.x: batch_x, self.y_: batch_y, self.length: batch_length,
                           self.initial_state_c: np.zeros((batch_x.shape[0], self.num_units)),
                           self.initial_state_h: np.zeros((batch_x.shape[0], self.num_units))})
            # Take summaries
            if summaries_dir is not None and (i % summary_interval == 0):
                accuracy, loss = self.predict_accuracy(x, y, writer=train_writer, writer_id=i, with_loss=True)
                logger.info('Step %d, train_set accuracy %g, loss %g' % (i, accuracy, loss))
                accuracy, loss = self.predict_accuracy(test_x, test_y, writer=test_writer, writer_id=i, with_loss=True)
                logger.info('Step %d, test_set accuracy %g, loss %g' % (i, accuracy, loss))
                if criterion == 'monitor_based':
                    accuracy, loss = self.predict_accuracy(valid_x, valid_y, writer=valid_writer, writer_id=i, with_loss=True)
                    logger.info('Step %d, valid_set accuracy %g, loss %g' % (i, accuracy, loss))
            # Get Summary
            i += 1
        # Finish Iteration
        if criterion == 'monitor_based':
            saver.restore(session, os.path.join(summaries_dir, 'best.ckpt'))
        logger.debug('Total Epoch: %d, current batch %d', injector.num_epochs, injector.cur_batch)

    def predict_proba(self, x, session=None, writer=None, writer_id=None):
        """Predict probability (Softmax)
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        targets = [self.y]
        if writer is not None:
            targets += [self.merged]
        results = session.run(targets,
                              feed_dict={self.x: x.reshape(tuple([1]) + x.shape),
                                         self.length: np.array([x.shape[0]], dtype=np.int),
                                         self.initial_state_c: np.zeros((1, self.num_units)),
                                         self.initial_state_h: np.zeros((1, self.num_units))})
        if writer is not None:
            writer.add_summary(results[1], writer_id)
        batch_y = results[0]
        # Get result
        return batch_y[0, :, :]

    def predict(self, x, session=None, writer=None, writer_id=None):
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        targets = [self.y_class]
        if writer is not None:
            targets += [self.merged]
        results = session.run(targets,
                              feed_dict={self.x: x.reshape(tuple([1]) + x.shape),
                                         self.length: np.array([x.shape[0]], dtype=np.int),
                                         self.initial_state_c: np.zeros((1, self.num_units)),
                                         self.initial_state_h: np.zeros((1, self.num_units))})
        if writer is not None:
            writer.add_summary(results[1], writer_id)
        batch_y = results[0]
        # Get result
        return batch_y[0, :]

    def predict_accuracy(self, x, y, session=None, writer=None, writer_id=None, with_loss=False):
        """Get Accuracy given feature array and corresponding labels
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        targets = [self.accuracy]
        if with_loss:
            targets += [self.loss]
        if writer is not None:
            targets += [self.merged]
        results = session.run(targets,
                              feed_dict={self.x: x.reshape(tuple([1]) + x.shape),
                                         self.y_: y.reshape(tuple([1]) + y.shape),
                                         self.length: np.array([x.shape[0]], dtype=np.int),
                                         self.initial_state_c: np.zeros((1, self.num_units)),
                                         self.initial_state_h: np.zeros((1, self.num_units))})
        if with_loss:
            return_values = results[0], results[1]
        else:
            return_values = results[0]
        if writer is not None:
            writer.add_summary(results[-1], writer_id)
        # Get result
        return return_values



class SimpleLSTM:
    """Single Layer LSTM Implementation

    In this new implementation, state_is_tuple is disabled to suppress the "deprecated" warning and
    performance improvement. The static unrolling of the RNN is replaced with dynamic unrolling.
    As a result, no batch injector is needed for prediction.

    Args:
        num_features
        num_classes
        num_units
    """

    def __init__(self, num_features, num_classes, num_hidden, num_units, num_skip, graph=None, optimizer=None):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_units = num_units
        self.num_skip = num_skip
        self.summaries = []
        if graph is None:
            graph = tf.Graph()
        with graph.as_default():
            # Inputs
            with tf.name_scope('input'):
                # X in the shape of (seq_length + num_skip, features)
                self.x = tf.placeholder(tf.float32, shape=[None, num_features], name='input_x')
                # length is the actual length of the sequence for each batch
                self.length = tf.placeholder(tf.int64, shape=[1], name='input_x_length')
                self.init_state = tf.placeholder(tf.float32, shape=[2 * num_units], name='init_state')
                self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
                self.y_skiped = self.y_[num_skip:, :]
            # Input Hidden layers
            self.hidden_layer = HiddenLayer(num_features, num_units, 'Hidden', x=self.x)
            # Recursive Layer
            with tf.name_scope('rnn'):
                # Apply RNN
                self.cell = rnn.BasicLSTMCell(num_units=num_units, state_is_tuple=False)
                # Outputs is a tensor with shape [seq_length + num_skip, num_units]
                outputs, states = tf.nn.dynamic_rnn(
                    self.cell, tf.reshape(self.hidden_layer.y, [1, -1, num_units]),
                    sequence_length=(self.length + num_skip),
                    initial_state=self.init_state, time_major=False)
            # Apply Softmax Layer to all outputs in the valid items in the sequence.
            self.output_layer = SoftmaxLayer(num_units, num_classes, 'SoftmaxLayer',
                                             x=outputs[1, num_skip:, :])
            # Softmax Cross-Entropy Loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.output_layer.logits, self.y_skiped,
                                                        name='SoftmaxCrossEntropy')
            )
            # Setup Optimizer
            if optimizer is None:
                self.optimizer = tf.train.AdamOptimizer()
            else:
                self.optimizer = optimizer
            # Predicted Probability
            self.y = self.output_layer.y
            self.y_class = tf.argmax(self.y, 1)
            # Evaluation
            self.correct_prediction = tf.equal(self.y_class, tf.argmax(self.y_skiped, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            # Fit Step
            with tf.name_scope('train'):
                self.fit_step = self.optimizer.minimize(self.loss)
            # Setup Summaries
            self.summaries += self.hidden_layer.summaries
            self.summaries += self.output_layer.summaries
            self.summaries.append(tf.summary.scalar('cross_entropy', self.loss))
            self.summaries.append(tf.summary.scalar('accuracy', self.accuracy))
            self.merged = tf.summary.merge(self.summaries)
            self.sess = None

    def fit(self, x, y, num_skip=100, batch_size=100, iter_num=100, summaries_dir=None, summary_interval=10,
            test_x=None, test_y=None, session=None, criterion='const_iteration'):
        """Fit the model to the dataset

        Args:
            x (:obj:`numpy.ndarray`): Input features of shape (num_samples, num_features).
            y (:obj:`numpy.ndarray`): Corresponding Labels of shape (num_samples) for binary classification,
                or (num_samples, num_classes) for multi-class classification.
            batch_size (:obj:`int`): Batch size used in gradient descent.
            iter_num (:obj:`int`): Number of training iterations for const iterations, step depth for monitor based
                stopping criterion.
            summaries_dir (:obj:`str`): Path of the directory to store summaries and saved values.
            summary_interval (:obj:`int`): The step interval to export variable summaries.
            test_x (:obj:`numpy.ndarray`): Test feature array used for monitoring training progress.
            test_y (:obj:`numpy.ndarray): Test label array used for monitoring training progress.
            session (:obj:`tensorflow.Session`): Session to run training functions.
            criterion (:obj:`str`): Stopping criteria. 'const_iterations' or 'monitor_based'
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        if summaries_dir is not None:
            train_writer = tf.summary.FileWriter(summaries_dir + '/train')
            test_writer = tf.summary.FileWriter(summaries_dir + '/test')
        session.run(tf.global_variables_initializer())
        # Get Stopping Criterion
        if criterion == 'const_iteration':
            _criterion = ConstIterations(num_iters=iter_num)
        elif criterion == 'monitor_based':
            num_samples = x.shape[0]
            valid_set_len = int(1 / 5 * (num_samples - num_skip))
            valid_x = x[num_samples - valid_set_len - num_skip:num_samples, :]
            valid_y = y[num_samples - valid_set_len - num_skip:num_samples, :]
            x = x[0:num_samples - valid_set_len, :]
            y = y[0:num_samples - valid_set_len, :]
            _criterion = MonitorBased(n_steps=iter_num,
                                      monitor_fn=self.predict_accuracy,
                                      monitor_fn_args=(valid_x, valid_y),
                                      save_fn=tf.train.Saver().save,
                                      save_fn_args=(session, summaries_dir + '/best.ckpt'))
        else:
            logger.error('Wrong criterion %s specified.' % criterion)
            return
        # Iteration Starts
        i = 0
        while _criterion.continue_learning():
            # Learning
            batch_x = x[i:num_skip + batch_size, :]
            batch_y = y[i:num_skip + batch_size, :]
            loss, accuracy, _ = session.run(
                [self.loss, self.accuracy, self.fit_step],
                feed_dict={self.x: batch_x, self.y_: batch_y, self.length: batch_size,
                           self.init_state: np.zeros(2 * self.num_units)})
            # Summary
            if summaries_dir is not None and (i % summary_interval == 0):
                summary, loss, accuracy = session.run(
                    [self.merged, self.loss, self.accuracy],
                    feed_dict={self.x: x, self.y_: y, self.length: num_samples - valid_set_len - num_skip,
                               self.init_state: np.zeros(2 * self.num_units)}
                )
                train_writer.add_summary(summary, i)
                logger.info('Step %d, train_set accuracy %g, loss %g' % (i, accuracy, loss))
                if (test_x is not None) and (test_y is not None):
                    merged, accuracy = session.run(
                        [self.merged, self.accuracy],
                        feed_dict={self.x: test_x, self.y_: test_y, self.length: test_x.shape[0] - num_skip,
                                   self.init_state: np.zeros(2*self.num_units)})
                    test_writer.add_summary(merged, i)
                    logger.info('test_set accuracy %g' % accuracy)
            # Get Summary
            if i == x.shape[0] - num_skip:
                i = 0
            else:
                i += 1
        # Finish Iteration
        if criterion == 'monitor_based':
            tf.train.Saver().restore(session, os.path.join(summaries_dir, 'best.ckpt'))

    def predict_proba(self, x, session=None, batch_size=500):
        """Predict probability (Softmax)
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.y,
                           feed_dict={self.x: x, self.length: x.shape[0] - self.num_skip,
                                      self.init_state: np.zeros(2*self.num_units)})

    def predict(self, x, session=None):
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.y_class,
                           feed_dict={self.x: x, self.length: x.shape[0] - self.num_skip,
                                      self.init_state: np.zeros(2*self.num_units)})

    def predict_accuracy(self, x, y, session=None):
        """Get Accuracy given feature array and corresponding labels
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.accuracy,
                           feed_dict={self.x: x, self.y_: y, self.length: x.shape[0] - self.num_skip,
                                      self.init_state: np.zeros(2*self.num_units)})