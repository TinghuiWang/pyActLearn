import logging
import numpy as np
import tensorflow as tf
from .layers import AutoencoderLayer, HiddenLayer, SoftmaxLayer
from .injectors import BatchInjector
from .criterion import MonitorBased, ConstIterations

logger = logging.getLogger(__name__)


class SDA:
    """Stacked Auto-encoder

    Args:
        num_features (:obj:`int`): Number of features.
        num_classes (:obj:`int`): Number of classes.
        layers (:obj:`list` of :obj:`int`): Series of hidden auto-encoder layers.
        encode_optimizer: Optimizer used for auto-encoding process.
        tuning_optimizer: Optimizer used for fine tuning.

    Attributes:
        num_features (:obj:`int`): Number of features.
        num_classes (:obj:`int`): Number of classes.
        x (:obj:`tensorflow.placeholder`): Input placeholder.
        y_ (:obj:`tensorflow.placeholder`): Output placeholder.
        inner_layers (:obj:`list`): List of auto-encoder hidden layers.

    """
    def __init__(self, num_features, num_classes, layers, encode_optimizer=None, tuning_optimizer=None):
        self.num_features = num_features
        self.num_classes = num_classes
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape=[None, num_features], name='input_x')
            self.y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.inner_layers = []
        self.summaries = []
        self.encode_opts = []
        if encode_optimizer is None:
            self.encode_optimizer = tf.train.AdamOptimizer()
        else:
            self.encode_optimizer = encode_optimizer
        if tuning_optimizer is None:
            self.tuning_optimizer = tf.train.AdamOptimizer()
        else:
            self.tuning_optimizer = tuning_optimizer
        # Create Layers
        for i in range(len(layers)):
            if i == 0:
                # First Layer
                self.inner_layers.append(
                    AutoencoderLayer(num_features, layers[i], x=self.x, name=('Hidden%d' % i))
                )
            else:
                # inner Layer
                self.inner_layers.append(
                    AutoencoderLayer(layers[i-1], layers[i], x=self.inner_layers[i-1].y, name=('Hidden%d' % i))
                )
            self.summaries += self.inner_layers[i].summaries
            self.encode_opts.append(
                self.encode_optimizer.minimize(self.inner_layers[i].encode_loss,
                                               var_list=self.inner_layers[i].variables)
            )
        if num_classes == 1:
            # Output Layers
            self.output_layer = HiddenLayer(layers[len(layers) - 1], num_classes, x=self.inner_layers[len(layers)-1].y,
                                            name='Output', activation_fn=tf.sigmoid)
            # Predicted Probability
            self.y = self.output_layer.y
            self.y_class = tf.cast(tf.greater_equal(self.y, 0.5), tf.float32)
            # Loss
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.output_layer.logits, self.y_,
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
        with tf.name_scope('train'):
            self.fine_tuning = self.tuning_optimizer.minimize(self.loss)
        self.merged = tf.summary.merge(self.summaries)
        self.sess = None

    def fit(self, x, y, batch_size=100,
            pretrain_iter_num=100, pretrain_criterion='const_iterations',
            tuning_iter_num=100, tuning_criterion='const_iterations',
            summaries_dir=None, test_x=None, test_y=None, summary_interval=10,
            session=None):
        """Fit the model to the dataset

        Args:
            x (:obj:`numpy.ndarray`): Input features of shape (num_samples, num_features).
            y (:obj:`numpy.ndarray`): Corresponding Labels of shape (num_samples) for binary classification,
                or (num_samples, num_classes) for multi-class classification.
            batch_size (:obj:`int`): Batch size used in gradient descent.
            pretrain_iter_num (:obj:`int`): Number of const iterations or search depth for monitor based stopping
                criterion in pre-training stage
            pretrain_criterion (:obj:`str`): Stopping criteria in pre-training stage ('const_iterations' or
                'monitor_based')
            tuning_iter_num (:obj:`int`): Number of const iterations or search depth for monitor based stopping
                criterion in fine-tuning stage
            tuning_criterion (:obj:`str`): Stopping criteria in fine-tuning stage ('const_iterations' or
                'monitor_based')
            summaries_dir (:obj:`str`): Path of the directory to store summaries and saved values.
            summary_interval (:obj:`int`): The step interval to export variable summaries.
            test_x (:obj:`numpy.ndarray`): Test feature array used for monitoring training progress.
            test_y (:obj:`numpy.ndarray): Test label array used for monitoring training progress.
            session (:obj:`tensorflow.Session`): Session to run training functions.
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        session.run(tf.global_variables_initializer())
        # Pre-training stage: layer by layer
        for j in range(len(self.inner_layers)):
            current_layer = self.inner_layers[j]
            if summaries_dir is not None:
                layer_summaries_dir = '%s/pretrain_layer%d' % (summaries_dir, j)
                train_writer = tf.summary.FileWriter(layer_summaries_dir + '/train')
                test_writer = tf.summary.FileWriter(layer_summaries_dir + '/test')
                valid_writer = tf.summary.FileWriter(layer_summaries_dir + '/valid')
            # Get Stopping Criterion
            if pretrain_criterion == 'const_iterations':
                _pretrain_criterion = ConstIterations(num_iters=pretrain_iter_num)
                train_x = x
                train_y = y
            elif pretrain_criterion == 'monitor_based':
                num_samples = x.shape[0]
                valid_set_len = int(1 / 5 * num_samples)
                valid_x = x[num_samples - valid_set_len:num_samples, :]
                valid_y = y[num_samples - valid_set_len:num_samples, :]
                train_x = x[0:num_samples - valid_set_len, :]
                train_y = y[0:num_samples - valid_set_len, :]
                _pretrain_criterion = MonitorBased(n_steps=pretrain_iter_num,
                                                   monitor_fn=self.get_encode_loss,
                                                   monitor_fn_args=(current_layer, valid_x, valid_y),
                                                   save_fn=tf.train.Saver().save,
                                                   save_fn_args=(session, layer_summaries_dir + '/best.ckpt'))
            else:
                logger.error('Wrong criterion %s specified.' % pretrain_criterion)
                return
            injector = BatchInjector(data_x=train_x, data_y=train_y, batch_size=batch_size)
            i = 0
            while _pretrain_criterion.continue_learning():
                batch_x, batch_y = injector.next_batch()
                if summaries_dir is not None and (i % summary_interval == 0):
                    summary, loss = session.run(
                        [current_layer.merged, current_layer.encode_loss],
                        feed_dict={self.x: x, self.y_: y}
                    )
                    train_writer.add_summary(summary, i)
                    logger.info('Pre-training Layer %d, Step %d, training loss %g' % (j, i, loss))
                    if test_x is not None and test_y is not None:
                        summary, loss = session.run(
                            [current_layer.merged, current_layer.encode_loss],
                            feed_dict={self.x: test_x, self.y_: test_y}
                        )
                        test_writer.add_summary(summary, i)
                        logger.info('Pre-training Layer %d, Step %d, test loss %g' % (j, i, loss))
                    if pretrain_criterion == 'monitor_based':
                        summary, loss = session.run(
                            [current_layer.merged, current_layer.encode_loss],
                            feed_dict={self.x: valid_x, self.y_: valid_y}
                        )
                        valid_writer.add_summary(summary, i)
                        logger.info('Pre-training Layer %d, Step %d, valid loss %g' % (j, i, loss))
                _ = session.run(self.encode_opts[j], feed_dict={self.x: batch_x, self.y_: batch_y})
                i += 1
            if pretrain_criterion == 'monitor_based':
                tf.train.Saver().restore(session, layer_summaries_dir + '/best.ckpt')
            if summaries_dir is not None:
                train_writer.close()
                test_writer.close()
                valid_writer.close()
        # Finish all internal layer-by-layer pre-training
        # Start fine tuning
        if summaries_dir is not None:
            tuning_summaries_dir = '%s/fine_tuning' % summaries_dir
            train_writer = tf.summary.FileWriter(tuning_summaries_dir + '/train')
            test_writer = tf.summary.FileWriter(tuning_summaries_dir + '/test')
            valid_writer = tf.summary.FileWriter(tuning_summaries_dir + '/valid')
        # Setup Stopping Criterion
        if tuning_criterion == 'const_iterations':
            _tuning_criterion = ConstIterations(num_iters=pretrain_iter_num)
            train_x = x
            train_y = y
        elif tuning_criterion == 'monitor_based':
            num_samples = x.shape[0]
            valid_set_len = int(1 / 5 * num_samples)
            valid_x = x[num_samples - valid_set_len:num_samples, :]
            valid_y = y[num_samples - valid_set_len:num_samples, :]
            train_x = x[0:num_samples - valid_set_len, :]
            train_y = y[0:num_samples - valid_set_len, :]
            _tuning_criterion = MonitorBased(n_steps=pretrain_iter_num,
                                             monitor_fn=self.predict_accuracy,
                                             monitor_fn_args=(valid_x, valid_y),
                                             save_fn=tf.train.Saver().save,
                                             save_fn_args=(session, tuning_summaries_dir + '/best.ckpt'))
        else:
            logger.error('Wrong criterion %s specified.' % pretrain_criterion)
            return
        injector = BatchInjector(data_x=train_x, data_y=train_y, batch_size=batch_size)
        i = 0
        while _tuning_criterion.continue_learning():
            batch_x, batch_y = injector.next_batch()
            if summaries_dir is not None and (i % summary_interval == 0):
                summary, loss, accuracy = session.run([self.merged, self.loss, self.accuracy],
                                                      feed_dict={self.x: train_x, self.y_: train_y})
                train_writer.add_summary(summary, i)
                logger.info('Fine-Tuning: Step %d, training accuracy %g, loss %g' % (i, accuracy, loss))
                if (test_x is not None) and (test_y is not None):
                    merged, accuracy = session.run([self.merged, self.accuracy],
                                                   feed_dict={self.x: test_x, self.y_: test_y})
                    test_writer.add_summary(merged, i)
                    logger.info('Fine-Tuning: Step %d, test accuracy %g' % (i, accuracy))
                if tuning_criterion == 'monitor_based':
                    merged, accuracy = session.run([self.merged, self.accuracy],
                                                   feed_dict={self.x: valid_x, self.y_: valid_y})
                    valid_writer.add_summary(merged, i)
                    logger.info('Fine-Tuning: Step %d, valid accuracy %g' % (i, accuracy))
            _ = session.run(self.fine_tuning, feed_dict={self.x: batch_x, self.y_: batch_y})
            i += 1
        if tuning_criterion == 'monitor_based':
            tf.train.Saver().restore(session, tuning_summaries_dir + '/best.ckpt')
        if summaries_dir is not None:
            train_writer.close()
            test_writer.close()
            valid_writer.close()

    def get_encode_loss(self, layer, x, y, session=None):
        """Get encoder loss of layer specified
        """
        if session is None:
            if self.sess is None:
                session = tf.Session()
                self.sess = session
            else:
                session = self.sess
        return session.run(layer.encode_loss, feed_dict={self.x: x, self.y_: y})

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
