import h5py
import logging
import dateutil.parser
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CASASHDF5:
    """CASASHDF5 Class to create and retrieve CASAS smart home data from h5df file 
    
    The data saved to or retrieved from a H5PY data file are pre-calculated features by
    :class:`CASASData` class. The H5PY data file also contains meta-data about the
    dataset, which include description for each feature, splits by week and/or splits
    by days.
    
    Attributes:
        _file (:class:`h5py.File`): :class:`h5py.File` object that represents root group.
    
    Args:
        filename (:obj:`str`): HDF5 File Name
        mode (:obj:`str`): 'r' for load from the file, and 'w' for create a new h5py data
    """
    def __init__(self, filename, mode='r', driver=None):
        self._file = h5py.File(filename, mode=mode, driver=driver)
        if mode == 'w':
            self._sources = []
            self._weeks = OrderedDict()
            self._days = OrderedDict()
            self._feature_description = []
            self._target_description = []
            self._target_colors = []
            self._sensors = []
            self._comment = ''
            self._bg_target = ''
        elif mode == 'r':
            self._load_dataset_info()
        else:
            raise ValueError('mode should be \'w\' or \'r\', but got %s.' % mode)

    def fetch_data(self, start_split=None, stop_split=None, pre_load=0):
        """Fetch data between start and stop splits
        
        Args:
            start_split (:obj:`str`): Begin of data
            stop_split (:obj:`str`): End of data
            pre_load (:obj:`int`): Load extra number of data before start split.
        
        Returns:
            :obj:`tuple` of :obj:`numpy.ndarray`: Returns a tuple of all sources sliced by the split defined.
                The sources should be in the order of ('time', 'feature', 'target')
        """
        start, stop = self._get_split_range(start_split, stop_split, pre_load)
        # Get time into a array of datetime
        if 'time' in self._sources:
            time_list = [dateutil.parser.parse(date_string.decode('utf-8'))
                         for date_string in self._file['time'][start:stop]]
        else:
            time_list = None
        # Get feature array
        if 'features' in self._sources:
            features = self._file['features'][start:stop]
        else:
            features = None
        # Get label array
        if 'targets' in self._sources:
            targets = self._file['targets'][start:stop]
        else:
            targets = None
        return time_list, features, targets

    # region Metadata Auxiliary Functions
    def num_sensors(self):
        """Return the number of sensors in the sensor list
        """
        return len(self._sensors)

    def get_sensor_by_index(self, i):
        """Get sensor name by index
        
        Args:
            i (:obj:`int`): Index to sensor
        """
        return self._sensors[i]

    def num_features(self):
        """Get number of features in the dataset
        """
        return len(self._feature_description)

    def get_feature_description_by_index(self, i):
        """Get the description of feature column :math:`i`.
        
        Args:
            i (:obj:`int`): Column index.
        
        Returns:
            :obj:`str`: Corresponding column description.
        """
        return self._feature_description[i]

    def num_targets(self):
        """Total number of target classes.
        
        Returns:
            :obj:`int`: Total number of target classes.
        """
        return len(self._target_description)

    def get_target_descriptions(self):
        """Get list of target descriptions
        
        Returns:
            :obj:`list` of :obj:`str`: List of target class description strings.
        """
        return self._target_description

    def get_target_description_by_index(self, i):
        """Get target description by class index :math:`i`.
        
        Args:
            i (:obj:`int`): Class index.
            
        Returns:
            :obj:`str`: Corresponding target class description.
        """
        return self._target_description[i]

    def get_target_colors(self):
        return self._target_colors

    def get_target_color_by_index(self, i):
        """Get the color string of target class :math:`i`.
        
        Args:
            i (:obj:`int`): Class index.
            
        Returns:
            :obj:`str`: Corresponding target class color string.        
        """
        return self._target_colors[i]

    def is_bg_target(self, i=None, label=None):
        """Check if the target class given by :param:`i` or :param:`label` is considered background
        
        Args:
            i (:obj:`int`): Class index.
            label (:obj:`str`): Class name.
            
        Returns:
            :obj:`bool`: True if it is considered background.
        """
        if i is not None:
            return i == self._target_description.index(self._bg_target)
        if label is not None:
            return label == self._bg_target
        return False

    def get_bg_target(self):
        """Get the description of the target class considered background in the dataset.
        
        Returns:
            :obj:`str`: Name of the class which is considered background in the dataset. Usually it is 'Other_Activity'.
        """
        return self._bg_target

    def get_bg_target_id(self):
        """Get the id of the target class considered background.
        
        Returns:
            :obj:`int`: The index of the target class which is considered background in the dataset.
        """
        return self._target_description.index(self._bg_target)

    def num_between_splits(self, start_split=None, stop_split=None):
        """Get the number of item between splits
        
        Args:
            start_split (:obj:`str`): Begin of data
            stop_split (:obj:`str`): End of data
        
        Returns:
            :obj:`int`: The number of items between two splits.
        """
        start, stop = self._get_split_range(start_split, stop_split)
        return stop - start

    def get_weeks_info(self):
        """Get splits by week.
        
        Returns:
            :obj:`List` of :obj:`tuple`: List of (key, value) tuple, where key is the name of the split and value is
                number of items in that split.
        """
        return [(week, self._weeks[week][1] - self._weeks[week][0]) for week in self._weeks]

    def get_days_info(self):
        """Get splits by day.

        Returns:
            :obj:`List` of :obj:`tuple`: List of (key, value) tuple, where key is the name of the split and value is
                number of items in that split.
        """
        return [(day, self._days[day][1] - self._days[day][0]) for day in self._days]
    # endregion

    # region CASASH5PY Dataset Creation
    def create_features(self, feature_array, feature_description):
        """ Create Feature Dataset
        
        Args:
            feature_array (:obj:`numpy.ndarray`): Numpy array holding calculated feature vectors
            feature_description (:obj:`list` of :obj:`str`): List of strings that describe each column of
                feature vectors.
        """
        if 'features' in self._sources:
            logger.error('Feature array already exists in the dataset.')
            return
        self._sources.append('features')
        self._feature_description = feature_description
        # Create feature array
        dset = self._file.create_dataset('features', data=feature_array,
                                         chunks=True, compression="gzip", compression_opts=9)
        dset.dims[0].label = 'batch'
        dset.dims[1].label = 'feature'
        # Add Feature Description as attributes
        self._file.attrs['features'] = [description.encode('utf-8')
                                        for description in feature_description]

    def create_targets(self, target_array, target_description, target_colors):
        """ Create Target Dataset
        
        Args:
            target_array (:obj:`numpy.ndarray`): Numpy array holding target labels
            target_description (:obj:`list` of :obj:`str`): List of strings that describe each each target class.
            target_colors (:obj:`list` of :obj:`str`): List of color values corresponding to each target class.
        """
        if 'targets' in self._sources:
            logger.error('Target array already exists in the dataset.')
            return
        self._sources.append('targets')
        self._target_description = target_description
        self._target_colors = target_colors
        # Create feature array
        dset = self._file.create_dataset('targets', data=target_array.reshape((target_array.size, 1)))
        dset.dims[0].label = 'batch'
        dset.dims[1].label = 'target'
        # Add Target Description as attributes
        self._file.attrs['targets'] = [description.encode('utf-8')
                                       for description in target_description]
        # Add Target Color as attributes
        self._file.attrs['target_colors'] = [color_string.encode('utf-8')
                                             for color_string in target_colors]

    def create_time_list(self, time_array):
        """ Create Time List
        
        Args:
            time_array (:obj:`list` of :obj:`datetime`): datetime corresponding to each feature vector in feature
                dataset.
        """
        if 'time' in self._sources:
            logger.error('Time list already exists in the dataset.')
            return
        self._sources.append('time')
        # Create Time lists
        num_items = len(time_array)
        dt = h5py.special_dtype(vlen=bytes)
        dset = self._file.create_dataset('time', (num_items,), dtype=dt)
        for i in range(num_items):
            dset[i] = time_array[i].isoformat().encode('utf-8')

    def create_splits(self, days, weeks):
        """ Create splits by days and weeks
        
        Args:
            days (:obj:`list` of :obj:`int`): Start index for each day
            weeks (:obj:`list` of :obj:`int`): Start index for week
        """
        if len(self._days) != 0 or len(self._weeks) != 0:
            logger.error('Splits already exist.')
            return
        self._days = OrderedDict()
        self._weeks = OrderedDict()
        max_name_len = len('week_%d' % len(days))
        # Create days numpy array
        days_array = np.empty(
            len(days) - 1,
            dtype=np.dtype([
                ('name', 'a', max_name_len),
                ('start', np.int64, 1),
                ('stop', np.int64, 1)]
            ))
        # Create days numpy array
        weeks_array = np.empty(
            len(weeks) - 1,
            dtype=np.dtype([
                ('name', 'a', max_name_len),
                ('start', np.int64, 1),
                ('stop', np.int64, 1)]
            ))
        # Populate days_array
        for i in range(len(days) - 1):
            days_array[i]['name'] = ('day_%d' % i).encode('utf-8')
            days_array[i]['start'] = days[i]
            days_array[i]['stop'] = days[i+1]
            self._days[('day_%d' % i)] = [days[i], days[i+1]]
        # Populate weeks array
        for i in range(len(weeks) - 1):
            weeks_array[i]['name'] = ('week_%d' % i).encode('utf-8')
            weeks_array[i]['start'] = weeks[i]
            weeks_array[i]['stop'] = weeks[i+1]
            self._weeks[('week_%d' % i)] = [weeks[i], weeks[i+1]]
        # Set attributes
        self._file.attrs['days'] = days_array
        self._file.attrs['weeks'] = weeks_array

    def create_comments(self, comment):
        """ Add comments to dataset
        
        Args:
            comment (:obj:`str`): Comments to the dataset
        """
        self._file.attrs['comment'] = comment.encode('utf-8')

    def create_sensors(self, sensors):
        """ Add sensors list to attributes
        
        If the sensor IDs in the dataset is not binary coded, there is a need to provide the sensor list to go along
        with the feature vectors.
        
        Args:
            sensors (:obj:`list` of :obj:`str`): List of sensor name corresponds to the id in the feature array.
        """
        self._file.attrs['sensors'] = [sensor.encode('utf-8') for sensor in sensors]

    def set_background_target(self, target_name):
        """ Set 'target_name' as background target
        
        Args:
            target_name (:obj:`str`): Name of background target
        """
        if self._bg_target != '':
            logger.error('background target label has been set to %s.' % self._bg_target)
            return
        self._bg_target = target_name
        self._file.attrs['bg_target'] = target_name.encode('utf-8')

    def flush(self):
        """ Write To File
        """
        self._file.attrs['sources'] = [source.encode('utf-8') for source in self._sources]
        self._file.flush()
    # endregion

    def close(self):
        """ Close Dataset
        """
        self._file.close()

    # region InternalSupportRoutines
    def _get_split_range(self, start_split=None, stop_split=None, pre_load=0):
        """Get the requested splits range

        Args:
            start_split (:obj:`str`): Begin of data
            stop_split (:obj:`str`): End of data
            pre_load (:obj:`int`): Load extra number of data before start split.

        Returns:
            :obj:`tuple` of :obj:`int`: Returns a tuple of the start and stop index.            
        """
        # Determine the start index
        if start_split is None:
            start = 0
            stop = self._file[self._sources[0]].shape[0]
        elif start_split in self._weeks:
            start = self._weeks[start_split][0]
            stop = self._weeks[start_split][1]
        elif start_split in self._days:
            start = self._days[start_split][0]
            stop = self._days[start_split][1]
        else:
            raise ValueError('start_split error: Cannot find %s in splitting array.' % start_split)
        # Determine the stop index
        if stop_split is not None:
            if stop_split in self._weeks:
                stop = self._weeks[stop_split][1]
            elif stop_split in self._days:
                stop = self._weeks[stop_split][1]
            else:
                raise ValueError('stop_split error: Cannot find %s in splitting array.' % stop_split)
        # Compensate pre-load
        start = start - pre_load
        if start < 0:
            start = 0
        return start, stop

    def _load_dataset_info(self):
        """Populate attributes of current class based on meta-data from h5py file
        """
        attrs = self._file.attrs.keys()
        # Check sources set
        if 'sources' in attrs:
            self._sources = [source.decode('utf-8') for source in self._file.attrs['sources']]
        else:
            self._sources = []
        # Parse splits
        self._weeks = OrderedDict()
        self._days = OrderedDict()
        if 'weeks' in attrs and 'days' in attrs:
            for row in self._file.attrs['weeks']:
                self._weeks[row['name'].decode('utf-8')] = [row['start'], row['stop']]
            for row in self._file.attrs['days']:
                self._days[row['name'].decode('utf-8')] = [row['start'], row['stop']]
        # Meta-data about dataset
        if 'features' in attrs:
            self._feature_description = [description.decode('utf-8')
                                         for description in self._file.attrs['features']]
        else:
            self._feature_description = []
        if 'targets' in attrs:
            self._target_description = [description.decode('utf-8')
                                        for description in self._file.attrs['targets']]
        else:
            self._target_description = []
        if 'target_colors' in attrs:
            self._target_colors = [color_string.decode('utf-8')
                                   for color_string in self._file.attrs['target_colors']]
        else:
            self._target_colors = []
        if 'sensors' in attrs:
            self._sensors = [sensor.decode('utf-8') for sensor in self._file.attrs['sensors']]
        else:
            self._sensors = []
        # Load Comments and Background task
        if 'bg_target' in attrs:
            self._bg_target = self._file.attrs['bg_target'].decode('utf-8')
        else:
            self._bg_target = ''
        if 'comment' in attrs:
            self._comment = self._file.attrs['comment'].decode('utf-8')
        else:
            self._comment = ''
    # endregion
