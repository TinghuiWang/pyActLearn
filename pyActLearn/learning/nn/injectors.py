import math
import random
import logging
import collections
import numpy as np

logger = logging.getLogger(__name__)


class BatchInjector:
    """Retrieving dataset values in batches

    Args:
        data_x (:obj:`numpy.ndarray`): Input feature array.
        data_y (:obj:`numpy.ndarray`): Input label array.
        batch_size (:obj:`int`): Batch size.
        num_batches (:obj:`int`): The number of batches in the input data.

    Attributes:
        size (:obj:`int`): Number of input vectors.
        batch_size (:obj:`int`): Batch size.
        num_batches (:obj:`int`): Number of batches in the input data.
        num_epochs (:obj:`int`): Number of epoch of current iteration.
        cur_batch (:obj:`int`): Current batch index.
        data_x (:obj:`numpy.ndarray`): Reference to input feature array.
        data_y (:obj:`numpy.ndarray`): Reference to input label array.s
    """
    def __init__(self, data_x, data_y=None, batch_size=-1, num_batches=-1):
        self.size = data_x.shape[0]
        if 0 < batch_size <= self.size:
            self.batch_size = batch_size
            self.num_batches = math.floor(self.size / self.batch_size)
        elif num_batches > 0:
            self.batch_size = math.floor(self.size / num_batches)
            self.num_batches = num_batches
        else:
            raise ValueError('Invalid batch_size or num_batches.')
        self.num_epochs = 0
        self.cur_batch = 0
        self.data_x = data_x
        self.data_y = data_y
        if data_y is not None:
            if self.data_x.shape[0] != self.data_y.shape[0]:
                raise ValueError('data_x, data_y provided have different number of rows.')

    def next_batch(self):
        """Get Next Batch
        """
        if self.cur_batch == self.num_batches - 1:
            start = self.batch_size * self.cur_batch
            end = self.size
            self.cur_batch = 0
            self.num_epochs += 1
        else:
            start = self.batch_size * self.cur_batch
            end = start + self.batch_size
            self.cur_batch += 1
        if self.data_y is None:
            return self.data_x[start:end, :]
        else:
            return self.data_x[start:end, :], self.data_y[start:end, :]

    def reset(self):
        """Reset all counters
        """
        self.cur_batch = 0
        self.num_epochs = 0


class BatchSequenceInjector:
    """Retrieving dataset values in batches and form a sequence of events

    Args:
        data_x (:obj:`numpy.ndarray`): Input feature array.
        data_y (:obj:`numpy.ndarray`): Input label array.
        seq_len (:obj:`int`): Length of sequence.
        batch_size (:obj:`int`): Batch size.
        num_batches (:obj:`int`): The number of batches in the input data.

    Attributes:
        length (:obj:`int`): Length of sequence.
        size (:obj:`int`): Number of input vectors.
        batch_size (:obj:`int`): Batch size.
        num_batches (:obj:`int`): Number of batches in the input data.
        num_epochs (:obj:`int`): Number of epoch of current iteration.
        cur_batch (:obj:`int`): Current batch index.
        data_x (:obj:`numpy.ndarray`): Reference to input feature array.
        data_y (:obj:`numpy.ndarray`): Reference to input label array.s
    """
    def __init__(self, data_x, data_y=None, length=100, batch_size=-1, num_batches=-1, with_seq=False):
        self.with_seq = with_seq
        self.length = length
        self.size = data_x.shape[0] - length
        if 0 < batch_size <= self.size:
            self.batch_size = batch_size
            self.num_batches = math.floor(self.size / self.batch_size)
        elif num_batches > 0:
            self.batch_size = math.floor(self.size / num_batches)
            self.num_batches = num_batches
        else:
            raise ValueError('Invalid batch_size or num_batches.')
        self.num_epochs = 0
        self.cur_batch = 0
        self.data_x = data_x
        self.data_y = data_y
        if data_y is not None:
            if self.data_x.shape[0] != self.data_y.shape[0]:
                raise ValueError('data_x, data_y provided have different number of rows.')

    def next_batch(self, skip=1):
        """Get Next Batch
        """
        self.cur_batch += skip-1
        if self.cur_batch > self.num_batches - 1:
            self.cur_batch = 0
            self.num_epochs += 1
        if self.cur_batch == self.num_batches - 1:
            start = self.batch_size * self.cur_batch
            end = self.size
            self.cur_batch = 0
            self.num_epochs += 1
        else:
            start = self.batch_size * self.cur_batch
            end = start + self.batch_size
            self.cur_batch += 1
        return self.to_sequence(self.length, self.data_x, self.data_y, start, end, with_seq=self.with_seq)

    def reset(self):
        """Reset all counters
        """
        self.cur_batch = 0
        self.num_epochs = 0

    @staticmethod
    def to_sequence(length, x, y=None, start=None, end=None, with_seq=False):
        """Turn feature array as a sequence array where each new feature contains seq_len number of original features.

        Args:
            length (:obj:`int`): Length of the sequence.
            x (:obj:`numpy.ndarray`): Feature array, with shape (num_samples, num_features).
            y (:obj:`numpy.ndarray`): Label array, with shape (num_samples. num_classes).
            start (:obj:`int`): Start index.
            end (:obj:`int`): End index

        Returns:
            (seq_x, seq_y) if y is provided, or seq_x if y is not provided.
            seq_x is a numpy array of shape (num_samples, seq_len, num_features), and seq_y is a numpy array
            of shape (num_samples, num_classes).
            num_samples is bounded by the value of start and end.
            If start or end are not specified, the code will use the full data provided, so that the
            array returned has (num_samples - seq_len) of samples.
        """
        if start is None or end is None:
            start = 0
            end = x.shape[0] - length
        if (start+length) > x.shape[0] or (end+length) > x.shape[0]:
            logger.error('start/end out of bound.')
            return None
        batch_x = np.zeros((end - start, length, x.shape[1]), np.float32)
        for i in range(start, end):
            batch_x[i-start, :, :] = x[i:i + length, :]
        return_tuple = tuple([batch_x])
        if y is not None:
            batch_y = np.zeros((end - start, length, y.shape[1]), np.float32)
            for i in range(start, end):
                batch_y[i-start, :, :] = y[i:i + length, :]
            return_tuple += tuple([batch_y])
        if with_seq:
            seq_ar = np.zeros((end - start,), np.float32)
            seq_ar[:] = length
            return_tuple += tuple([seq_ar])
        return return_tuple


class SkipGramInjector:
    """Skip-Gram Batch Injector

    It generates a k-skip-2-gram sets based on input sequence

    Args:
        data_x (:obj:`np.ndarray`): 1D array of integer index.
        batch_size (:obj:`int`): Size of each batch to be generated.
        num_skips (:obj:`int`): How many times to re-use an input to generate a label.
        skip_window (:obj:`int`): How many items to consider left or right.

    Attributes:
        data_x (:obj:`np.ndarray`): 1D array of integer index.
        batch_size (:obj:`int`): Size of each batch to be generated.
        num_skips (:obj:`int`): How many times to re-use an input to generate a label.
        skip_window (:obj:`int`): How many items to consider left or right.
        data_index (:obj:`int`): Current index used to generate next batch.
    """
    def __init__(self, data_x, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        self.data_x = data_x
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.data_index = 0

    def next_batch(self):
        """Get Next Batch
        """
        # Initialize batch and label array
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        # span is the size of window we are sampling from
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        # Add data in the buffer to a queue
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data_x[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data_x)
        # Now, populate the k-skip-2-gram data-label pair with random sampling
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data_x[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data_x)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data_x) - span) % len(self.data_x)
        return batch, labels
