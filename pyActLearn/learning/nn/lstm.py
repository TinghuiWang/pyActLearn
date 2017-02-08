import os
import logging
import numpy as np
import tensorflow as tf
from .layers import HiddenLayer, SoftmaxLayer
from .injectors import BatchSequenceInjector
from .criterion import MonitorBased, ConstIterations

logger = logging.getLogger(__name__)


class LSTM:
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
        hidden_y = tf.split(0, num_steps, self.hidden_layer.y)
        # Apply RNN
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=False)
        outputs, states = tf.nn.rnn(self.cell, hidden_y, initial_state=self.init_state)
        self.last_state = states[-1]
        # Output Softmax Layer
        self.output_layer = SoftmaxLayer(num_units, num_classes, 'SoftmaxLayer', x=outputs[-1])
        # Predicted Probability
        self.y = self.output_layer.y
        self.y_class = tf.argmax(self.y, 1)
        # Softmax Cross-Entropy Loss
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.output_layer.logits, self.y_,
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
        # Setup batch injector
        injector = BatchSequenceInjector(data_x=x, data_y=y, batch_size=batch_size, seq_len=self.num_steps)
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
