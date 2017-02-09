import os
import pickle
import logging
import numpy as np
from fuel.datasets import H5PYDataset

logger = logging.getLogger(__name__)


class CASASFuel(object):
    """CASASFuel Class to retrieve CASAS smart home data as a fuel dataset object

    Args:
        dir_name (:obj:`string`):
            Directory path that contains HDF5 dataset file and complementary dataset information pkl file

    Attributes:
        data_filename (:obj:`str`): Path to `data.hdf5` dataset file
        info (:obj:`dict`): complementary dataset information stored in dict format
            keys of info includes:
    """
    def __init__(self, dir_name):
        logger.debug('Load Casas H5PYDataset from ' + dir_name)
        self.data_filename = dir_name + '/data.hdf5'
        if os.path.isfile(dir_name + '/info.pkl'):
            f = open(dir_name + '/info.pkl', 'rb')
            self.info = pickle.load(f)
            f.close()
        else:
            logger.error('Cannot find info.pkl from current H5PYDataset directory %s' % dir_name)

    def get_dataset(self, which_sets, load_in_memory=False, **kwargs):
        """Return fuel dataset object specified by which_sets tuple and load it in memory

        Args:
            which_sets (:obj:`tuple` of :obj:`str`):  containing the name of splits to load.
                Valid value are determined by the ``info.pkl`` loaded.
                You can get the list of split set names by :meth:`get_set_list()`.
                Usually, if the dataset is split by weeks, the split name is in the form of ``week <num>``.
                If the dataset is split by days, the split name is in the form of ``day <num>``.
            load_in_memory (:obj:`bool`, Optional): Default to False.
                Whether to load the data in main memory.

        Returns:
            :class:`fuel.datasets.base.Dataset`: A Fuel dataset object created by
                :class:`fuel.datasets.h5py.H5PYDataset`
        """
        # Check if sets exist as split name in metadata
        for set_name in which_sets:
            if set_name not in self.info['split_sets']:
                logger.error('set %s not found in splits' % set_name)
        # Load specified splits and return
        return H5PYDataset(file_or_path=self.data_filename,
                           which_sets=which_sets,
                           load_in_memory=load_in_memory, **kwargs)

    def get_set_list(self):
        """Get the split set list

        Returns:
            :obj:`tuple` of :obj:`str`: A list of split set names
        """
        return self.info['split_sets']

    def get_input_dims(self):
        """Get the dimension of features

        Returns:
            :obj:`int` : the input feature length
        """
        dims = len(self.info['index_to_feature'])
        return dims

    def get_output_dims(self):
        """Get the dimension of target indices

        Returns:
            :obj:`int` : the target indices
        """
        dims = len(self.info['index_to_activity'])
        return dims

    def get_activity_by_index(self, index):
        """Get activity name by index

        Args:
            index (:obj:`int`): Activity index

        Returns:
            :obj:`str`: Activity label
        """
        activity_len = len(self.info['index_to_activity'])
        if index < activity_len:
            return self.info['index_to_activity'][index]
        else:
            logger.error('Activity index %d out of bound. Dataset has %d activities' % (index, activity_len))
            return ''

    def get_feature_by_index(self, index):
        """Get feature string by index

        Args:
            index (:obj:`int`): Feature index

        Returns:
            :obj:`str`: Feature string
        """
        feature_len = len(self.info['index_to_feature'])
        if index < feature_len:
            return self.info['index_to_feature'][index]
        else:
            logger.error('Feature index %d out of bound. Dataset has %d features' % (index, feature_len))
            return ''

    def back_annotate(self, fp, prediction, split_id=-1, split_name=None):
        """Back annotated predictions of a split set into file pointer

        Args:
            fp (:obj:`file`): File object to the back annotation file.
            prediction (:obj:`numpy.ndarray`): Numpy array containing prediction labels.
            split_id (:obj:`int`): The index of split set to be annotated (required if split_name not specified).
            split_name (:obj:`str`): The name of the split set to be annotated (required if split_id is not specified).
        """
        time_array = self._get_time_array(split_id=split_id, split_name=split_name)
        # Check length of prediction and time array
        if prediction.shape[0] != len(time_array):
            logger.error('Prediction size miss-match. There are %d time points with only %d labels given.' %
                         (len(time_array), prediction.shape[0]))
            return
        # Perform back annotation
        for i in range(len(time_array)):
            if prediction[i] != -1:
                fp.write('%s %s\n' % (time_array[i].strftime('%Y-%m-%d %H:%M:%S'),
                                      self.get_activity_by_index(prediction[i])))

    def back_annotate_with_proba(self, fp, prediction_proba, split_id=-1, split_name=None, top_n=-1):
        """Back annotated prediction probabilities of a split set into file pointer

        Args:
            fp (:obj:`file`): File object to the back annotation file.
            prediction_proba (:obj:`numpy.ndarray`): Numpy array containing probability for each class in shape
                of (num_samples, num_class).
            split_id (:obj:`int`): The index of split set to be annotated (required if split_name not specified).
            split_name (:obj:`str`): The name of the split set to be annotated (required if split_id is not specified).
            top_n (:obj:`int`): Back annotate top n probabilities.
        """
        time_array = self._get_time_array(split_id=split_id, split_name=split_name)
        # Check length of prediction and time array
        if prediction_proba.shape[0] != len(time_array):
            logger.error('Prediction size miss-match. There are %d time points with only %d labels given.' %
                         (len(time_array), prediction_proba.shape[0]))
            return
        if top_n == -1:
            top_n = self.get_output_dims()
        # Perform back annotation
        for i in range(len(time_array)):
            sorted_index = np.argsort(prediction_proba[i, :])[::-1]
            if prediction_proba[i, sorted_index[0]] != -1:
                fp.write('%s' % time_array[i].strftime('%Y-%m-%d %H:%M:%S'))
                for j in range(top_n):
                    fp.write(', %s(%g)' % (self.get_activity_by_index(sorted_index[j]),
                                           prediction_proba[i, sorted_index[j]]))
                fp.write('\n')

    def _get_time_array(self, split_id=-1, split_name=None):
        """Get Time Array based for specified splits

        Args:
            split_id (:obj:`int`): The index of split set to be annotated (required if split_name not specified).
            split_name (:obj:`str`): The name of the split set to be annotated (required if split_id is not specified).

        Returns:
            :obj:`list` of :obj:`datetime.datetime`: List of event datetime objects of splits specified.
        """
        if split_id == -1:
            if type(split_name) is tuple:
                time_array = []
                for each_split in split_name:
                    each_split_id = self.info['split_sets'].index(each_split)
                    if 0 < each_split_id < len(self.info['split_sets']):
                        time_array += self.info['split_timearray'][each_split_id]
            else:
                if split_name in self.info['split_sets']:
                    split_id = self.info['split_sets'].index(split_name)
                    if 0 < split_id < len(self.info['split_sets']):
                        time_array = self.info['split_timearray'][split_id]
                else:
                    logger.error('Failed to find split set with name %s.' % split_name)
                    return None
        elif 0 < split_id < len(self.info['split_sets']):
            time_array = self.info['split_timearray'][split_id]
        else:
            logger.error('Split set index %d out of bound.' % split_id)
            return None
        return time_array

    @staticmethod
    def files_exist(dir_name):
        """Check if the CASAS Fuel dataset files exist under dir_name
        """
        data_filename = os.path.join(dir_name, 'data.hdf5')
        info_filename = os.path.join(dir_name, 'info.pkl')
        return os.path.isfile(data_filename) and os.path.isfile(info_filename)
