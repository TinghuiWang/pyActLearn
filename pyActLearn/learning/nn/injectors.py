import math
import logging
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
        seq_len (:obj:`int`): Length of sequence.
        size (:obj:`int`): Number of input vectors.
        batch_size (:obj:`int`): Batch size.
        num_batches (:obj:`int`): Number of batches in the input data.
        num_epochs (:obj:`int`): Number of epoch of current iteration.
        cur_batch (:obj:`int`): Current batch index.
        data_x (:obj:`numpy.ndarray`): Reference to input feature array.
        data_y (:obj:`numpy.ndarray`): Reference to input label array.s
    """
    def __init__(self, data_x, data_y=None, seq_len=100, batch_size=-1, num_batches=-1):
        self.seq_len = seq_len
        self.size = data_x.shape[0] - seq_len
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
        return self.to_sequence(self.seq_len, self.data_x, self.data_y, start, end)

    def reset(self):
        """Reset all counters
        """
        self.cur_batch = 0
        self.num_epochs = 0

    @staticmethod
    def to_sequence(seq_len, x, y=None, start=None, end=None):
        """Turn feature array as a sequence array where each new feature contains seq_len number of original features.

        Args:
            seq_len (:obj:`int`): Length of the sequence.
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
            end = x.shape[0] - seq_len
        if (start+seq_len) > x.shape[0] or (end+seq_len) > x.shape[0]:
            logger.error('start/end out of bound.')
            return None
        batch_x = np.zeros((end - start, seq_len, x.shape[1]), np.float32)
        for i in range(start, end):
            batch_x[i-start, :, :] = x[i:i+seq_len, :]
        if y is None:
            return batch_x
        else:
            return batch_x, y[start+seq_len:end+seq_len, :]