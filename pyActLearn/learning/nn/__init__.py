import tensorflow as tf


def variable_summary(var, tag):
    """Attach a lot of summaries to a Tensor for visualization

    Args:
        var (:obj:`tensorflow.Tensor`): Tensor variable.
        tag (:obj:`str`): Tag string for the tensor variable.
    """
    summary_array = []
    with tf.name_scope(tag + '/summaries'):
        mean = tf.reduce_mean(var)
        summary_array.append(tf.summary.scalar('mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summary_array.append(tf.summary.scalar('stddev', stddev))
        summary_array.append(tf.summary.scalar('max', tf.reduce_max(var)))
        summary_array.append(tf.summary.scalar('min', tf.reduce_min(var)))
        summary_array.append(tf.summary.histogram('histogram', var))
    return summary_array
