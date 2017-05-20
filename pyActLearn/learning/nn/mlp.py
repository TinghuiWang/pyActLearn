import os
import logging
import numpy as np
import tensorflow as tf
from .layers import HiddenLayer, SoftmaxLayer
from .injectors import BatchInjector
from .criterion import MonitorBased, ConstIterations

logger = logging.getLogger(__name__)


class MLP:
    """Multi-Layer Perceptron

    Args:
        num_features (:obj:`int`): Number of features.
        num_classes (:obj:`int`): Number of classes.
        layers (:obj:`list` of :obj:`int`): Series of hidden auto-encoder layers.
        activation_fn: activation function used in hidden layer.
        optimizer: Optimizer used for updating weights.

    Attributes:
        num_features (:obj:`int`): Number of features.
        num_classes (:obj:`int`): Number of classes.
        x (:obj:`tensorflow.placeholder`): Input placeholder.
        y_ (:obj:`tensorflow.placeholder`): Output placeholder.
        inner_layers (:obj:`list`): List of inner hidden layers.
        summaries (:obj:`list`): List of tensorflow summaries.
        output_layer: Output softmax layer for multi-class classification, sigmoid for binary classification
        y (:obj:`tensorflow.Tensor`): Softmax/Sigmoid output layer output tensor.
        y_class (:obj:`tensorflow.Tensor`): Tensor to get class label from output layer.
        loss (:obj:`tensorflow.Tensor`): Tensor that represents the cross-entropy loss.
        correct_prediction (:obj:`tensorflow.Tensor`): Tensor that represents the correctness of classification result.
        accuracy (:obj:`tensorflow.Tensor`): Tensor that represents the accuracy of the classifier (exact matching
            ratio in multi-class classification)
        optimizer: Optimizer used for updating weights.
        fit_step (:obj:`tensorflow.Tensor`): Tensor to update weights based on the optimizer algorithm provided.
        sess: Tensorflow session.
        merged: Merged summaries.
    """
    def __init__(self, num_features, num_classes, layers, activation_fn=tf.sigmoid, optimizer=None):
        self.num_features = num_features
        self.num_classes = num_classes
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, num_features], name='input_x')
            self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.inner_layers = []
        self.summaries = []
        # Create Layers
        for i in range(len(layers)):
            if i == 0:
                # First Layer
                self.inner_layers.append(
                    HiddenLayer(num_features, layers[i], x=self.x, name=('Hidden%d' % i), activation_fn=activation_fn)
                )
            else:
                # inner Layer
                self.inner_layers.append(
                    HiddenLayer(layers[i-1], layers[i], x=self.inner_layers[i-1].y,
                                name=('Hidden%d' % i), activation_fn=activation_fn)
                )
            self.summaries += self.inner_layers[i].summaries
        if num_classes == 1:
            # Output Layers
            self.output_layer = HiddenLayer(layers[len(layers) - 1], num_classes, x=self.inner_layers[len(layers)-1].y,
                                            name='Output', activation_fn=tf.sigmoid)
            # Predicted Probability
            self.y = self.output_layer.y
            self.y_class = tf.cast(tf.greater_equal(self.y, 0.5), tf.float32)
            # Loss
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_layer.logits, labels=self.y_,
                                                        name='SigmoidCrossEntropyLoss')
            )
            self.correct_prediction = tf.equal(self.y_class, self.y_)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        else:
            # Output Layers
            self.output_layer = SoftmaxLayer(layers[len(layers) - 1], num_classes, x=self.inner_layers[len(layers)-1].y,
                                             name='OutputLayer')
            # Predicted Probability
            self.y = self.output_layer.y
            self.y_class = tf.argmax(self.y, 1)
            # Loss
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer.logits, labels=self.y_,
                                                        name='SoftmaxCrossEntropyLoss')
            )
            self.correct_prediction = tf.equal(self.y_class, tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.summaries.append(tf.summary.scalar('cross_entropy', self.loss))
        self.summaries.append(tf.summary.scalar('accuracy', self.accuracy))
        self.summaries += self.output_layer.summaries
        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        else:
            self.optimizer = optimizer
        with tf.name_scope('train'):
            self.fit_step = self.optimizer.minimize(self.loss)
        self.merged = tf.summary.merge(self.summaries)
        self.sess = None

    def fit(self, x, y, batch_size=100, iter_num=100,
            summaries_dir=None, summary_interval=100,
            test_x=None, test_y=None,
            session=None, criterion='const_iteration'):
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
            train_writer = tf.summary.FileWriter(summaries_dir + '/train', session.graph)
            test_writer = tf.summary.FileWriter(summaries_dir + '/test')
            valid_writer = tf.summary.FileWriter(summaries_dir + '/valid')
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
                                      monitor_fn=self.predict_accuracy, monitor_fn_args=(valid_x, valid_y),
                                      save_fn=tf.train.Saver().save,
                                      save_fn_args=(session, summaries_dir + '/best.ckpt'))
        else:
            logger.error('Wrong criterion %s specified.' % criterion)
            return
        # Setup batch injector
        injector = BatchInjector(data_x=x, data_y=y, batch_size=batch_size)
        i = 0
        train_accuracy = 0
        while _criterion.continue_learning():
            batch_x, batch_y = injector.next_batch()
            if summaries_dir is not None and (i % summary_interval == 0):
                summary, loss, accuracy = session.run([self.merged, self.loss, self.accuracy],
                                                      feed_dict={self.x: x, self.y_: y})
                train_writer.add_summary(summary, i)
                train_accuracy = accuracy
                logger.info('Step %d, train_set accuracy %g, loss %g' % (i, accuracy, loss))
                if (test_x is not None) and (test_y is not None):
                    merged, accuracy = session.run([self.merged, self.accuracy],
                                                   feed_dict={self.x: test_x, self.y_: test_y})
                    test_writer.add_summary(merged, i)
                    logger.info('test_set accuracy %g' % accuracy)
                if criterion == 'monitor_based':
                    merged, accuracy = session.run([self.merged, self.accuracy],
                                                   feed_dict={self.x: valid_x, self.y_: valid_y})
                    valid_writer.add_summary(merged, i)
                    logger.info('valid_set accuracy %g' % accuracy)
            loss, accuracy, _ = session.run([self.loss, self.accuracy, self.fit_step],
                                            feed_dict={self.x: batch_x, self.y_: batch_y})
            #logger.info('Step %d, training accuracy %g, loss %g' % (i, accuracy, loss))
            #_ = session.run(self.fit_step, feed_dict={self.x: batch_x, self.y_: batch_y})
            #logger.info('Step %d, training accuracy %g, loss %g' % (i, accuracy, loss))
            i += 1
        if criterion == 'monitor_based':
            tf.train.Saver().restore(session, os.path.join(summaries_dir, 'best.ckpt'))
        logger.debug('Total Epoch: %d, current batch %d', injector.num_epochs, injector.cur_batch)

    def predict_accuracy(self, x, y, session=None):
        """Get Accuracy given feature array and corresponding labels
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.accuracy, feed_dict={self.x: x, self.y_: y})

    def predict_proba(self, x, session=None):
        """Predict probability (Softmax)
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.y, feed_dict={self.x: x})

    def predict(self, x, session=None):
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(self.y_class, feed_dict={self.x: x})
