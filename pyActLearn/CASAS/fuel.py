import os
import pickle
import logging
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

