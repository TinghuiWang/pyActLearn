import math
import numpy as np
import tensorflow as tf
from ..learning.nn.injectors import SkipGramInjector


def sensor2vec(num_sensors, sensor_event_list, embedding_size=20,
               batch_size=128, num_skips=8, skip_window=5,
               num_neg_samples=64, learning_rate=1.0):
    """Sensor to Vector
    """
    if num_neg_samples > num_sensors:
        num_neg_samples = num_sensors
    # Initialize a SkipGram Injector
    injector = SkipGramInjector(sensor_event_list, batch_size, num_skips, skip_window)
    # Build Training Model
    graph = tf.Graph()
    with graph.as_default():
        # Input Place Holder
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # As we normally do not have too many sensors - it is OK to use all of them
        valid_dataset = tf.constant([i for i in range(num_sensors)], dtype=tf.int32)
        # Only CPU supports NCE loss
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([num_sensors, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([num_sensors, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([num_sensors]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_neg_samples,
                           num_classes=num_sensors))

        # Construct the Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()

        # Begin training.
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = injector.next_batch()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

            final_embeddings = normalized_embeddings.eval()
            final_similarity = 1 - similarity.eval()
            distance_matrix = final_similarity / np.max(final_similarity, axis=1)[:, None]
    return final_embeddings, distance_matrix



def sensor2vec_data(sensor_list, event_list, embedding_size=20,
               batch_size=128, num_skips=8, skip_window=5,
               num_neg_samples=64, learning_rate=1.0, ignore_off=True):
    """Transform sensor to high dimensional space

    Similar to word embedding used in natural language processing system, we want
    to represent sensors using in a synthesized vector space as well, instead of
    using an arbitrary labels for each sensors without any useful information.

    The methods used to find word embeddings can be classified into two categories:
    count-based methods (Latent Semantic Analysis) and predictive models.
    In this implementation for mapping sensor into high dimension vector space, we
    use skip-gram negative sampling models.

    Args:
        sensor_list (:obj:`list` of :obj:`dict`): List of dictionary containing
            sensor information.
        event_list (:obj:`list` of :obj:`dict`): List of events.
        embedding_size (:obj:`int`): The size of embedding vector.
        batch_size (:obj:`int`): The number of batch used in training
        num_skips (:obj:`int`): How many times to re-use an input to generate a label
            in skip-gram model.
        skip_window (:obj:`int`): How many items to consider left or right in skip-gram
            model.
        num_neg_samples (:obj:`int`): Number of negative samples to draw from the vocabulary.
        ignore_off (:obj:`bool`): Ignore motion-sensor with ``Off`` state in event.rst list.

    Please refer to :func:`sensor_distance` for an example of ``sensor_list``.
    Please refer to :func:`sensor_mi_distance` for an example of ``event_list``.
    """
    # Put sensor in hash table for fast fetch of index
    num_sensors = len(sensor_list)
    # Negative samples cannot exceed sensor numbers
    if num_neg_samples > num_sensors:
        num_neg_samples = num_sensors
    # Store sensor ID in hash table for faster access
    sensor_dict = {}
    for i in range(num_sensors):
        sensor_dict[sensor_list[i]['name']] = i
    # Generate event.rst sensor list
    event_sensor_list = []
    for event_entry in event_list:
        if ignore_off and event_entry['sensor_status'].upper() == "OFF":
            continue
        event_sensor_list.append(sensor_dict[event_entry['sensor_id']])
    # Initialize a SkipGram Injector
    injector = SkipGramInjector(event_sensor_list, batch_size, num_skips, skip_window)
    # Build Training Model
    graph = tf.Graph()
    with graph.as_default():
        # Input Place Holder
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # As we normally do not have too many sensors - it is OK to use all of them
        valid_dataset = tf.constant([i for i in range(num_sensors)], dtype=tf.int32)
        # Only CPU supports NCE loss
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([num_sensors, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([num_sensors, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([num_sensors]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_neg_samples,
                           num_classes=num_sensors))

        # Construct the Optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.initialize_all_variables()

        # Begin training.
        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")

            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = injector.next_batch()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(num_sensors):
                        valid_sensor = sensor_list[i]['name']
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_sensor
                        for k in range(top_k):
                            close_sensor = sensor_list[nearest[k]]['name']
                            log_str = "%s %s," % (log_str, close_sensor)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()
            final_similarity = 1 - similarity.eval()
            distance_matrix = final_similarity / np.max(final_similarity, axis=1)[:,None]

    # try:
    #     from sklearn.manifold import TSNE
    #     import matplotlib.pyplot as plt
    #
    #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #     low_dim_embs = tsne.fit_transform(final_embeddings)
    #     labels = [sensor_list[i]['name'] for i in range(num_sensors)]
    #
    #     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    #     plt.figure(figsize=(18, 18))  # in inches
    #     for i, label in enumerate(labels):
    #         x, y = low_dim_embs[i, :]
    #         plt.scatter(x, y)
    #         plt.annotate(label,
    #                      xy=(x, y),
    #                      xytext=(5, 2),
    #                      textcoords='offset points',
    #                      ha='right',
    #                      va='bottom')
    #     plt.show()
    # except ImportError:
    #     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

    return final_embeddings, distance_matrix
