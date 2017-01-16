import os
import sys
import time
import math
import h5py
import pickle
import logging
import datetime
import xlsxwriter
import numpy as np
import scipy.sparse as sp
from fuel.datasets.hdf5 import H5PYDataset

from .home import CASASHome
from .stat_features import EventHour, EventSeconds, LastSensor, WindowDuration, \
                           SensorCount, DominantSensor, SensorElapseTime

logger = logging.getLogger(__name__)


class CASASData(object):
    """A class to load activity data from CASAS datasets

    This class is used to load activity data from CASAS datasets, compute data statistics and basic visualization
    of the statistics

    Args:
        path (:obj:`str`): path to a dataset directory, the dataset event file for dataset in legacy format, or
            a pickle file that stored a CASASData structure.

    Attributes:
        sensor_list (:obj:`dict`): A dictionary containing sensor information
        activity_list (:obj:`dict`): A dictionary containing activity information
        event_list (:obj:`list` of :obj:`dict`): List of data used to store raw events
        x (:obj:`numpy.ndarray`): 2D numpy array that contains calculated feature data
        y (:obj:`numpy.ndarray`): 2D numpy array that contains activity label corresponding to feature array
        data_path (:obj:`str`): path to data file
        home (:class:`pyActLearn.CASAS.home.CASASHome`): :class:`CASAS.home.CASASHome` object that stores
            the home information associated with the dataset.
        is_legacy (:obj:`bool`): Defaults to False. If the dataset loaded is in legacy format or not.
        is_stat_feature (:obj:`bool`): Calculate statistical features or use raw data in ``x``
        is_labeled (:obj:`bool`): If given dataset is labeled
        time_list (:obj:`list` of :class:`datetime.datetime`): Datetime of each entry in ``x``. Used for back annotation,
            and splitting dataset by weeks or days.
        feature_list (:obj:`dict`): A dictionary of statistical features used in statistical feature calculation
        routines (:obj:`dict`): Function routines that needs to run every time when calculating features.
            Excluded from pickling.
        num_enabled_features (:obj:`int`): Number of enabled features.
        num_static_features (:obj:`int`): Number of features related to window
        num_per_sensor_features (:obj:`int`): Number of features that needs to be calculated per enabled sensor
        events_in_window (:obj:`int`): Number of sensor events (or statistical features of a sliding window)
            grouped in a feature vector.
    """

    def __init__(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            logger.error('Cannot find %s' % path)
            raise FileNotFoundError('Cannot find %s' % path)
        # Initialize Default Values
        self.x = None
        self.y = None
        self.is_labeled = True
        self.activity_list = {}
        self.sensor_list = {}
        self.event_list = []
        self.events_in_window = 1
        self.time_list = []
        # Statistical Features and flag
        self.is_stat_feature = False
        self.max_window_size = 30
        self.feature_list = {}
        self.routines = {}
        self.num_enabled_features = 0
        self.num_static_features = 0
        self.num_per_sensor_features = 0
        # From which source to construct CASAS data
        if os.path.isdir(path):
            logger.debug('Load CASAS data from directory %s' % path)
            self.home = CASASHome(directory=path)
            self.is_legacy = False
            self.data_path = path
            # Populate sensor list, activity list with data from self.home
            for sensor in self.home.get_all_sensors():
                self._add_sensor(sensor)
            for activity in self.home.get_all_activities():
                self._add_activity(activity)
            # Load Events
            logger.debug('Load CASAS sensor events from %s' % self.data_path)
            self._load_events_from_dataset(os.path.join(path, './events.csv'))
        else:
            filename, file_ext = os.path.splitext(path)
            if file_ext == '.pkl':
                # A pickle file - unpickle it - but if this is the case, user can directly
                # get the class from pickle.load function
                logger.debug('Load from pickle file %s' % path)
            else:
                self.home = None
                self.is_legacy = True
                self.data_path = ""

    def __getstate__(self):
        """Return state to be pickled
        """
        state = self.__dict__.copy()
        if self.x is not None:
            density_count = np.count_nonzero(self.x)
            density = float(density_count) / self.x.size
            if density < 0.5:
                state['x'] = sp.csr_matrix(state['x'])
        return self.__dict__

    def __setstate__(self, state):
        """Set state from pickled file
        """
        if sp.issparse(state['x']):
            state['x'] = state['x'].todense()
        self.__dict__.update(state)

    def _load_events_from_legacy(self, filename):
        """Load casas data from annotated event logs

        Args:
            filename (:obj:`str`): absolute path to file
        """
        self.event_list = []
        if os.path.isfile(filename):
            self.data_path = filename
            f = open(filename, 'r')
            line_number = 0
            for line in f:
                line_number += 1
                word_list = str(str(line).strip()).split()
                if len(word_list) > 3:
                    # date, time, sensor ID, sensor status, annotated label
                    date_list = word_list[0].split('-')
                    time_list = word_list[1].split(':')
                    sec_list = time_list[2].split('.')
                    event_time = datetime.datetime(int(date_list[0]),
                                                   int(date_list[1]),
                                                   int(date_list[2]),
                                                   int(time_list[0]),
                                                   int(time_list[1]),
                                                   int(sec_list[0]),
                                                   int(sec_list[1]))
                    cur_data_dict = {
                        'datetime': event_time,
                        'sensor_id': word_list[2],
                        'sensor_status': word_list[3],
                    }
                    self._add_sensor(cur_data_dict['sensor_id'])
                    self.is_labeled = False
                    if len(word_list) > 4:
                        self.is_labeled = True
                        # Add Corresponding Labels
                        cur_data_dict['activity'] = word_list[4]
                        self._add_activity(cur_data_dict['activity'])
                    self.event_list.append(cur_data_dict)
                else:
                    logger.error('Error parsing %s:%d' % (filename, line_number))
                    logger.error('  %s' % line)
        else:
            raise FileNotFoundError('Cannot find file %s' % filename)

    def _load_events_from_dataset(self, filename):
        """Load events from CASAS dataset

        Args:
            filename (:obj:`str`): path to ``event.csv`` file in the dataset
        """
        self.event_list = []
        self.is_labeled = False
        if os.path.isfile(filename):
            f = open(filename, 'r')
            line_number = 0
            for line in f:
                line_number += 1
                word_list = str(str(line).strip()).split(',')
                if len(word_list) < 6:
                    logger.error('Error parsing %s:%d' % (filename, line_number))
                    logger.error('  %s' % line)
                    continue
                # date, time, sensor ID, sensor status, annotated label
                if '/' in word_list[0]:
                    event_time = datetime.datetime.strptime(word_list[0] + ' ' + word_list[1], "%m/%d/%Y %H:%M:%S")
                else:
                    event_time = datetime.datetime.strptime(word_list[0] + ' ' + word_list[1], "%Y-%m-%d %H:%M:%S")
                # Remove OFF - no use
                if word_list[3] == "OFF":
                    continue
                # Remove Continuous Firing
                # if len(self.event_list) > 0 and
                # word_list[2] == self.event_list[len(self.event_list) - 1]['sensor_id']:
                #     continue
                cur_data_dict = {
                    'datetime': event_time,
                    'sensor_id': word_list[2],
                    'sensor_status': word_list[3],
                    'resident_name': word_list[4],
                    'activity': word_list[5]
                }
                if len(word_list[5]) > 0:
                    self.is_labeled = True
                    if not cur_data_dict['activity'] in self.activity_list:
                        logger.warn('Activity %s not found in activity list. Added it now.'
                                    % cur_data_dict['activity'])
                        self._add_activity(cur_data_dict['activity'])
                # Add Corresponding Labels
                self.event_list.append(cur_data_dict)
        else:
            logger.error('Cannot find data file %s\n' % filename)

    def populate_feature(self, method='raw', normalized=True, per_sensor=True):
        """Populate the feature vector in ``x`` and activities in `y`

        Args:
            method (:obj:`str`): The method to convert sensor events into feature vector.
                Available methods are ``'raw'`` and ``'stat'``.
            normalized (:obj:`bool`): Will each feature be normalized between 0 and 1?
            per_sensor (:obj:`bool`): For features related with sensor ID, are they
        """
        if method == 'raw':
            self._calculate_raw_features(normalized, per_sensor)
        else:
            self._add_feature(EventHour(normalized=normalized))
            self._add_feature(EventSeconds(normalized=normalized))
            self._add_feature(LastSensor(per_sensor=per_sensor))
            self._add_feature(WindowDuration(normalized=normalized))
            self._add_feature(SensorCount(normalized=normalized))
            self._add_feature(DominantSensor(per_sensor=per_sensor))
            self._add_feature(SensorElapseTime(normalized=normalized))
            self._calculate_stat_features()

    def _calculate_raw_features(self, normalized=True, per_sensor=True):
        """Populate the feature vector with raw sensor data

        Args:
            normalized (:obj:`bool`): Will each feature be normalized between 0 and 1?
            per_sensor (:obj:`bool`): For features related with sensor ID, are they
        """
        num_events = len(self.event_list)
        events_in_window = self.events_in_window
        self.y = np.zeros((num_events - events_in_window + 1,))
        self.time_list = []
        if per_sensor:
            len_per_event = 1 + len(self.get_enabled_sensors())
        else:
            len_per_event = 2
        num_col = len_per_event * events_in_window
        self.x = np.zeros((num_events - events_in_window + 1, num_col))
        for i in range(num_events - events_in_window + 1):
            self.y[i] = self.get_activity_index(self.event_list[i + events_in_window - 1]['activity'])
            for j in range(events_in_window):
                # Datetime is represented in seconds
                event_time = self.event_list[i + events_in_window - 1 - j]['datetime']
                seconds = event_time.timestamp() - \
                    datetime.datetime.combine(event_time.date(), datetime.time.min).timestamp()
                if normalized:
                    self.x[i, j*len_per_event] = seconds/(24*3600)
                else:
                    self.x[i, j*len_per_event] = seconds
                # Sensor id
                sensor_index = self.get_sensor_index(self.event_list[i + events_in_window - 1 - j]['sensor_id'])
                if per_sensor:
                    self.x[i, j * len_per_event + sensor_index + 1] = 1
                else:
                    self.x[i, j * len_per_event + 1] = sensor_index
            self.time_list.append(self.event_list[i + events_in_window - 1]['datetime'])
        return num_events

    def _calculate_stat_features(self):
        """Populate the feature vector with statistical features using sliding window
        """
        num_feature_columns = self._count_feature_columns()
        num_feature_rows = self._count_samples()
        self.x = np.zeros((num_feature_rows, num_feature_columns), dtype=np.float)
        self.y = np.zeros(num_feature_rows, dtype=np.int)
        cur_row_id = self.max_window_size - 1
        cur_sample_id = 0
        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.clear()
        while cur_row_id < len(self.event_list):
            cur_sample_id += self._calculate_window_feature(cur_row_id, cur_sample_id)
            cur_row_id += 1
        # Due to sensor event discontinuity, the sample size will be smaller than the num_feature_rows calculated
        self.x = self.x[0:cur_sample_id, :]
        self.y = self.y[0:cur_sample_id]
        self.is_stat_feature = True
        logger.debug('Total amount of feature vectors calculated: %d' % cur_sample_id)

    def _count_samples(self):
        """Count the maximum possible samples in data_list
        """
        num_events = len(self.event_list)
        if num_events < self.max_window_size - 1:
            logger.error('data size is %d smaller than window size %d' %
                         (len(self.event_list), self.max_window_size))
            return 0
        num_sample = 0
        if self.is_labeled:
            # If labeled, count enabled activity entry after the first
            # max_window_size event
            for event in self.event_list:
                if num_sample < self.max_window_size + self.events_in_window - 2:
                    num_sample += 1
                else:
                    """ ToDo: Need to check sensor enable status to make count sample count """
                    if self.activity_list[event['activity']]['enable']:
                        num_sample += 1
            num_sample -= self.max_window_size + self.events_in_window - 2
        else:
            # If not labeled, we need to calculate for each window
            # and finally find which catalog it belongs to
            num_sample = num_events - self.max_window_size - self.events_in_window + 2
        return num_sample

    def _calculate_window_feature(self, cur_row_id, cur_sample_id):
        """Calculate feature vector for current window specified by cur_row_id

        Args:
            cur_row_id (:obj:`int`): Row index of current window (last row)
            cur_sample_id (:obj:`int`): Row index of current sample in self.x

        Returns:
            :obj:`int`: number of feature vector added
        """
        # Default Window Size to 30
        window_size = self.max_window_size
        num_enabled_sensors = len(self.get_enabled_sensors())
        # Skip current window if labeled activity is ignored
        if self.is_labeled:
            activity_label = self.event_list[cur_row_id]['activity']
            window_size = self.activity_list[activity_label]['window_size']
            if not self.activity_list[activity_label]['enable']:
                return 0
        if cur_row_id > self.max_window_size - 1:
            if cur_sample_id == 0:
                for i in range(self.num_enabled_features * (self.events_in_window - 1)):
                    self.x[cur_sample_id][self.num_enabled_features*self.events_in_window-i-1] = \
                        self.x[cur_sample_id][self.num_enabled_features * (self.events_in_window-1)-i-1]
            else:
                for i in range(self.num_enabled_features * (self.events_in_window - 1)):
                    self.x[cur_sample_id][self.num_enabled_features*self.events_in_window-i-1] = \
                        self.x[cur_sample_id-1][self.num_enabled_features * (self.events_in_window-1)-i-1]
        # Execute feature update routine
        for (key, routine) in self.routines.items():
            if routine.enabled:
                routine.update(data_list=self.event_list, cur_index=cur_row_id,
                               window_size=window_size, sensor_info=self.sensor_list)
        # Get Feature Data and Put into arFeature array
        for (key, feature) in self.feature_list.items():
            if feature.enabled:
                # If it is per Sensor index, we need to iterate through all sensors to calculate
                if feature.per_sensor:
                    for sensor_name in self.sensor_list.keys():
                        if self.sensor_list[sensor_name]['enable']:
                            column_index = self.num_static_features + \
                                           feature.index * num_enabled_sensors + \
                                           self.sensor_list[sensor_name]['index']
                            self.x[cur_sample_id][column_index] = \
                                feature.get_feature_value(data_list=self.event_list,
                                                          cur_index=cur_row_id,
                                                          window_size=window_size,
                                                          sensor_info=self.sensor_list,
                                                          sensor_name=sensor_name)
                else:
                    self.x[cur_sample_id][feature.index] = \
                        feature.get_feature_value(data_list=self.event_list,
                                                  cur_index=cur_row_id,
                                                  window_size=window_size,
                                                  sensor_info=self.sensor_list,
                                                  sensor_name=None)
                if not feature.is_value_valid:
                    return 0
        if cur_row_id < self.max_window_size + self.events_in_window - 2:
            return 0
        if self.is_labeled:
            self.y[cur_sample_id] = self.activity_list[self.event_list[cur_row_id]['activity']]['index']
        self.time_list.append(self.event_list[cur_row_id]['datetime'])
        return 1

    _COLORS = ('#b20000, #56592d, #acdae6, #cc00be, #591616, #d5d9a3, '
               '#007ae6, #4d0047, #a67c7c, #2f3326, #00294d, #b35995, '
               '#ff9180, #1c330d, #73b0e6, #f2b6de, #592400, #6b994d, '
               '#1d2873, #ff0088, #cc7033, #50e639, #0000ff, #7f0033, '
               '#e6c3ac, #00d991, #c8bfff, #592d3e, #8c5e00, #80ffe5, '
               '#646080, #d9003a, #332200, #397367, #6930bf, #33000e, '
               '#ffbf40, #3dcef2, #1c0d33, #8c8300, #23778c, #ba79f2, '
               '#e6f23d, #203940, #302633').split(',')

    # region Activity List
    def _add_activity(self, label):
        """Add activity to :attr:`activity_list`

        Args:
            label (:obj:`str`): activity label

        Returns:
            :obj:`int`: activity index
        """
        if label not in self.activity_list:
            logger.debug('add activity class %s' % label)
            if self.is_legacy:
                self.activity_list[label] = {'name': label}
            else:
                self.activity_list[label] = self.home.get_activity(label)
                if self.activity_list[label] is None:
                    logger.warn('Failed to find information about activity %s' % label)
                    self.activity_list[label] = {'name': label}
            self.activity_list[label]['index'] = -1
            self.activity_list[label]['enable'] = True
            self.activity_list[label]['window_size'] = 30
            self._assign_activity_indices()
        return self.activity_list[label]['index']

    def _assign_activity_indices(self):
        """Assign index number to each activity enabled

        Returns:
            :obj:`int`: Number of enabled activities
        """
        num_enabled_activities = 0
        for label in self.activity_list.keys():
            activity = self.activity_list[label]
            if activity['enable']:
                activity['index'] = num_enabled_activities
                num_enabled_activities += 1
            else:
                activity['index'] = -1
        logger.debug('Finished assigning index to activities. %d Activities enabled' % num_enabled_activities)
        return num_enabled_activities

    def get_activities_by_indices(self, activity_ids):
        """Get a group of activities by their corresponding indices

        Args:
            activity_ids (:obj:`list` of :obj:`int`): A list of activity indices

        Returns:
            :obj:`list` of :obj:`str`: A list of activity labels in the same order
        """
        return [self.get_activity_by_index(cur_id) for cur_id in activity_ids]

    def get_activity_by_index(self, activity_id):
        """Get Activity name by their index

        Args:
            activity_id (:obj:`int`): Activity index

        Returns:
            :obj:`str`: Activity label
        """
        for activity_label in self.activity_list.keys():
            if activity_id == self.activity_list[activity_label]['index']:
                return activity_label
        logger.error('Failed to find activity with index %d' % activity_id)
        return ""

    def get_activity_index(self, activity_label):
        """Get Index of an activity

        Args:
            activity_label (:obj:`str`): Activity label

        Returns:
            :obj:`int`: Activity index (-1 if not found or not enabled)
        """
        if activity_label in self.activity_list:
            return self.activity_list[activity_label]['index']
        else:
            return -1

    def get_enabled_activities(self):
        """Get label list of all enabled activities

        Returns:
            :obj:`list` of :obj:`str`: list of activity labels
        """
        enabled_activities_list = []
        for activity_label in self.activity_list.keys():
            if self.activity_list[activity_label]['enable']:
                enabled_activities_list.append(activity_label)
        return enabled_activities_list

    def get_activity_color(self, activity_label):
        """Find the color string for the activity.

        Args:
            activity_label (:obj:`str`): activity label

        Returns:
            :obj:`str`: RGB color string
        """
        if self.is_legacy:
            # Pick the color from color list based on the activity index
            activity_index = self.get_activity_index(activity_label)
            if activity_index >= 0:
                return self._COLORS[activity_index % len(self._COLORS)]
            else:
                return '#C8C8C8'   # returns grey
        else:
            return self.home.get_activity_color(activity_label)

    def enable_activity(self, activity_label):
        """Enable an activity

        Args:
            activity_label (:obj:`str`): Activity label

        Returns:
            :obj:`int`: The index of the enabled activity
        """
        if activity_label in self.activity_list:
            logger.debug('Enable Activity %s' % activity_label)
            self.activity_list[activity_label]['enable'] = True
            self._assign_activity_indices()
            return self.activity_list[activity_label]['index']
        else:
            logger.error('Activity %s not found' % activity_label)
            return -1

    def disable_activity(self, activity_label):
        """Disable an activity

        Args:
            activity_label (:obj:`str`): Activity label
        """
        if activity_label in self.activity_list:
            logger.debug('Disable Activity %s' % activity_label)
            self.activity_list[activity_label]['enable'] = False
            self.activity_list[activity_label]['index'] = -1
            self._assign_activity_indices()
        else:
            logger.error('Activity %s not found' % activity_label)
    # endregion

    # region Sensor List
    def _add_sensor(self, name):
        """Add Sensor to :attr:`sensor_list`

        Args:
            name (:obj:`str`): sensor name

        Returns:
            (:obj:`int`): sensor index
        """
        if name not in self.sensor_list:
            logger.debug('Add sensor %s to sensor list' % name)
            if self.is_legacy:
                self.sensor_list[name] = {'name': name}
            else:
                self.sensor_list[name] = self.home.get_sensor(name)
                if self.sensor_list[name] is None:
                    logger.error('Failed to find information about sensor %s' % name)
                    self.sensor_list[name] = {'name': name}
            self.sensor_list[name]['index'] = -1
            self.sensor_list[name]['enable'] = True
            self.sensor_list[name]['lastFireTime'] = None
            self._assign_sensor_indices()
        return self.sensor_list[name]['index']

    def _assign_sensor_indices(self):
        """Assign index to each enabled sensor

        Returns
            :obj:`int`: The number of enabled sensor
        """
        sensor_id = 0
        for sensor_label in self.sensor_list.keys():
            if self.sensor_list[sensor_label]['enable']:
                self.sensor_list[sensor_label]['index'] = sensor_id
                sensor_id += 1
            else:
                self.sensor_list[sensor_label]['index'] = -1
        return sensor_id

    def enable_sensor(self, sensor_name):
        """Enable a sensor

        Args:
            sensor_name (:obj:`str`): Sensor Name

        Returns
            :obj:`int`: The index of the enabled sensor
        """
        if sensor_name in self.sensor_list:
            logger.debug('Enable Sensor %s' % sensor_name)
            self.sensor_list[sensor_name]['enable'] = True
            self._assign_sensor_indices()
            return self.sensor_list[sensor_name]['index']
        else:
            logger.error('Failed to find sensor %s' % sensor_name)
            return -1

    def disable_sensor(self, sensor_name):
        """Disable a sensor

        Args:
            sensor_name (:obj:`str`): Sensor Name
        """
        if sensor_name in self.sensor_list:
            logger.debug('Disable Sensor %s' % sensor_name)
            self.sensor_list[sensor_name]['enable'] = False
            self.sensor_list[sensor_name]['index'] = -1
            self._assign_sensor_indices()
        else:
            logger.error('Failed to find sensor %s' % sensor_name)

    def get_sensor_by_index(self, sensor_id):
        """Get the name of sensor by index

        Args:
            sensor_id (:obj:`int`): Sensor index

        Returns:
            :obj:`str`: Sensor name
        """
        for sensor_name in self.sensor_list.keys():
            if self.sensor_list[sensor_name]['index'] == sensor_id:
                return sensor_name
        logger.error('Failed to find sensor with index %d' % sensor_id)
        return ''

    def get_sensor_index(self, sensor_name):
        """Get Sensor Index

        Args:
            sensor_name (:obj:`str`): Sensor Name

        Returns:
            :obj:`int`: Sensor index (-1 if not found or not enabled)
        """
        if sensor_name in self.sensor_list:
            return self.sensor_list[sensor_name]['index']
        else:
            return -1

    def get_enabled_sensors(self):
        """Get the names of all enabled sensors

        Returns:
            :obj:`list` of :obj:`str`: List of sensor names
        """
        enabled_sensor_array = []
        for sensor_label in self.sensor_list.keys():
            if self.sensor_list[sensor_label]['enable']:
                enabled_sensor_array.append(sensor_label)
        return enabled_sensor_array
    # endregion

    # region Stat Feature Routine Update Management
    def _add_routine(self, routine):
        """Add routine to feature update routine list

        Args:
            routine (:class:`pyActLearn.CASAS.stat_features.FeatureRoutineTemplate`): routine to be added
        """
        if routine.name in self.routines.keys():
            logger.debug('feature routine %s already existed.' % routine.name)
        else:
            logger.debug('Add feature routine %s: %s' % (routine.name, routine.description))
            self.routines[routine.name] = routine

    def disable_routine(self, routine):
        """ Disable a routine

        Check all enabled feature list and see if the routine is used by other features.
        If no feature need the routine, disable it

        Args:
            routine (:class:`pyActLearn.CASAS.stat_features.FeatureRoutineTemplate`): routine to be disabled
        """
        if routine.name in self.routines.keys():
            for feature_name in self.feature_list.keys():
                if self.feature_list[feature_name].enabled:
                    if self.feature_list[feature_name].routine == routine:
                        logger.debug('routine %s is used by feature %s.' % (routine.name, feature_name))
                        return
            logger.debug('routine %s is disabled.' % routine.name)
            self.routines[routine.name].enabled = False
        else:
            logger.error('routine %s not added to routine list' % routine.name)

    def enable_routine(self, routine):
        """Enable a given routine

        Args:
            routine (:class:`pyActLearn.CASAS.stat_features.FeatureRoutineTemplate`): routine to be disabled
        """
        if routine.name in self.routines.keys():
            logger.debug('routine %s is enabled.' % routine.name)
            routine.enabled = True
        else:
            logger.error('routine %s not added to routine list' % routine.name)
    # endregion

    # region Stat Feature Management
    def _add_feature(self, feature):
        """Add Feature to feature list

        Args:
            feature (:class:`pyActlearn.CASAS.stat_features`): FeatureTemplate Object
        """
        if feature.name in self.feature_list.keys():
            logger.warn('feature: %s already existed. Add Feature Function ignored.' % feature.name)
        else:
            logger.debug('Add Feature %s: %s' % (feature.name, feature.description))
            self.feature_list[feature.name] = feature
            if feature.routine is not None:
                self._add_routine(feature.routine)
            self._assign_feature_indexes()

    def disable_feature(self, feature_name):
        """Disable a feature

        Args:
            feature_name (:obj:`str`): Feature name.
        """
        if feature_name in self.feature_list.keys():
            logger.debug('Disable Feature %s: %s' % (feature_name, self.feature_list[feature_name]['description']))
            self.feature_list[feature_name].enabled = True
            self.feature_list[feature_name].index = -1
            self._assign_feature_indexes()
            if self.feature_list[feature_name].routine is not None:
                self.disable_routine(self.feature_list[feature_name].routine)
        else:
            logger.error('Feature %s Not Found' % feature_name)

    def enable_feature(self, feature_name):
        """Enable a feature

        Args:
            feature_name (:obj:`str`): Feature name.
        """
        if feature_name in self.feature_list.keys():
            logger.debug('Enable Feature %s: %s' % (feature_name, self.feature_list[feature_name]['description']))
            self.feature_list[feature_name].enabled = True
            self._assign_feature_indexes()
            if self.feature_list[feature_name].routine is not None:
                self.enable_routine(self.feature_list[feature_name].routine)
        else:
            logger.error('Feature %s Not Found' % feature_name)

    def _assign_feature_indexes(self):
        """Assign index to features
        """
        static_id = 0
        per_sensor_id = 0
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.enabled:
                if feature.per_sensor:
                    feature.index = per_sensor_id
                    per_sensor_id += 1
                else:
                    feature.index = static_id
                    static_id += 1
            else:
                feature.index = -1
        self.num_static_features = static_id
        self.num_per_sensor_features = per_sensor_id
        logger.debug('Finished assigning index to features. %d Static Features, %d Per Sensor Features' %
                     (static_id, per_sensor_id))

    def get_feature_by_index(self, index):
        """Get Feature Name by Index

        Args:
            index (:obj:`int`): column index of feature

        Returns:
            :obj:`tuple` of :obj:`str`: (feature name, sensor name) tuple.
                If it is not per-sensor feature, the sensor name is None.
        """
        max_id = self.num_enabled_features
        num_enabled_sensors = len(self.get_enabled_sensors())
        if index > max_id:
            logger.error('index %d is greater than the number of feature columns %d' %
                         (index, max_id))
        if index > self.num_static_features:
            # It is per_sensor Feature
            sensor_id = (index - self.num_static_features) % num_enabled_sensors
            feature_id = math.floor((index - self.num_static_features) / num_enabled_sensors)
            per_sensor = True
        else:
            # It is a generic feature
            sensor_id = -1
            feature_id = index
            per_sensor = False
        # Find Corresponding feature name and sensor label
        feature_name = None
        for featureLabel in self.feature_list.keys():
            feature = self.feature_list[featureLabel]
            if feature.index == feature_id and feature.per_sensor == per_sensor:
                feature_name = featureLabel
                break
        sensor_name = 'Window'
        if sensor_id > 0:
            for sensor_label in self.sensor_list.keys():
                sensor = self.sensor_list[sensor_label]
                if sensor['index'] == sensor_id:
                    sensor_name = sensor_label
                    break
        return feature_name, sensor_name

    def get_feature_string_by_index(self, index):
        """Get the string describing the feature specified by column index

        Args:
            index (:obj:`int`): column index of feature

        Returns:
            :obj:`str`: Feature string
        """
        # Check if it is a statistical feature
        if self.is_stat_feature:
            # It is stat feature
            feature_name, sensor_name = self.get_feature_by_index(index)
            if feature_name is None or sensor_name is None:
                logger.error('Failed to find feature/sensor name for feature %d' % index)
                return 'None'
            else:
                return sensor_name + ": " + feature_name
        else:
            # It is a windowed event feature
            if self.x.shape[1] == 2 * self.events_in_window:
                # Sensor ID is presented as integer
                entry_num = int(index / 2)
                index_in_entry = index % 2
                if index_in_entry == 0:
                    return "-%d Entry: Time" % entry_num
                else:
                    return "-%d Entry: Sensor ID" % entry_num
            else:
                # Sensor ID is presented as a binary array
                num_sensors = len(self.get_enabled_sensors())
                entry_num = int(index / (num_sensors + 1))
                index_in_entry = int(index % (num_sensors + 1))
                if index_in_entry == 0:
                    return "-%d Entry: Time" % entry_num
                else:
                    return "-%d Entry: %s" % (entry_num, self.get_sensor_by_index(index_in_entry - 1))

    def _update_feature_count(self):
        """Update feature count values
        """
        self.num_enabled_features = 0
        self.num_static_features = 0
        self.num_per_sensor_features = 0
        for name, feature in self.feature_list.items():
            if feature.enabled:
                self.num_enabled_features += 1
                if feature.per_sensor:
                    self.num_per_sensor_features += 1
                else:
                    self.num_static_features += 1

                def count_feature_columns(self):
                    """
                    Count the size of feature columns
                    :return: integer, size of feature columns
                    """
                    self.num_enabled_features = 0
                    num_enabled_sensors = len(self.get_enabled_sensors())
                    for feature_name in self.feature_list.keys():
                        if self.feature_list[feature_name].enabled:
                            if self.feature_list[feature_name].per_sensor:
                                self.num_enabled_features += num_enabled_sensors
                            else:
                                self.num_enabled_features += 1
                    return self.num_enabled_features * self.events_in_window

    def _count_feature_columns(self):
        """Count the size of feature columns

        Returns:
            :obj:`int`: size of feature columns
        """
        self.num_enabled_features = 0
        num_enabled_sensors = len(self.get_enabled_sensors())
        for feature_name in self.feature_list.keys():
            if self.feature_list[feature_name].enabled:
                if self.feature_list[feature_name].per_sensor:
                    self.num_enabled_features += num_enabled_sensors
                else:
                    self.num_enabled_features += 1
        return self.num_enabled_features * self.events_in_window
    # endregion

    # region Summary
    def summary(self):
        """Print summary of loaded datasets
        """
        print('Dataset Path: %s' % self.data_path)
        print('Sensors: %d' % len(self.sensor_list))
        print('Sensors enabled: %d' % len(self.get_enabled_sensors()))
        print('Activities: %d' % len(self.activity_list))
        print('Activities enabled: %d' % len(self.get_enabled_activities()))
        print('loaded events: %d' % len(self.event_list))
        if self.x is not None:
            print('feature array: (%d, %d)' % (self.x.shape[0], self.x.shape[1]))
            print('activity array: (%d, )' % self.y.shape[0])
    # endregion

    def _break_by_day(self):
        """Find the split point of the dataset by day

        Returns:
            :obj:`list` of :obj:`int`: List of indices of the event at the beginning of each day
        """
        day_index_list = [0]
        start_date = self.time_list[0].date()
        for i in range(len(self.time_list)):
            cur_date = self.time_list[i].date()
            if cur_date > start_date:
                day_index_list.append(i)
                start_date = cur_date
        day_index_list.append(len(self.time_list))
        return day_index_list

    def _break_by_week(self):
        """Find the split point of the dataset by week

        Returns:
            :obj:`list` of :obj:`int`: List of indices of the event at the beginning of each week
        """
        week_index_list = [0]
        start_date = self.time_list[0].date()
        for i in range(len(self.time_list)):
            cur_date = self.time_list[i].date()
            # Monday - then not the same day as start_date
            # Else, if more than 7 days apart
            if (cur_date.weekday() == 0 and cur_date > start_date) or (cur_date - start_date).days >= 7:
                week_index_list.append(i)
                start_date = cur_date
        week_index_list.append(len(self.time_list))
        return week_index_list

    def export_hdf5(self, directory, break_by='week', comments=''):
        """Export feature and label vector into hdf5 file and store the class information in a pickle file

        Args:
            directory (:obj:`str`): The directory to save hdf5 and complementary dataset information
            break_by (:obj:`str`): Select the way to split the data, either by ``'week'`` or ``'day'``
            comments (:obj:`str`): Additional comments to add
        """
        if os.path.exists(directory):
            if os.path.isdir(directory):
                overwrite = ' '
                while overwrite not in ['n', 'N', 'y', 'Y']:
                    # ask if overwrite
                    overwrite = input('Directory %s found. Overwrite? [Y/n] ' % directory)
                    if overwrite == 'n' or overwrite == 'N':
                        return
                    elif overwrite == '':
                        break
            else:
                sys.stderr.write('%s is not a directory. Abort.')
                return
        else:
            os.mkdir(directory)
        # Create HDF5 File
        f = h5py.File(directory + '/data.hdf5', mode='w')
        # Create features and targets array
        features = f.create_dataset('features', self.x.shape, dtype='float32')
        targets = f.create_dataset('targets', (self.y.shape[0], 1), dtype='uint8')
        features[...] = self.x
        targets[...] = self.y.reshape((self.y.shape[0], 1))
        features.dims[0].label = 'batch'
        features.dims[1].label = 'feature'
        targets.dims[0].label = 'batch'
        targets.dims[1].label = 'index'
        # Find Split Locations
        if break_by == 'day':
            break_list = self._break_by_day()
        else:
            break_list = self._break_by_week()
        # Construct split dict
        split_dict = {}
        split_set = []
        split_timearray = []
        num_break_point = len(break_list) - 1
        for i in range(num_break_point):
            start = break_list[i]
            stop = break_list[i + 1]
            split_name = break_by + ' ' + str(i)
            split_dict[split_name] = {
                'features': (start, stop),
                'targets': (start, stop)
            }
            split_set.append(split_name)
            split_timearray.append(self.time_list[start:stop])
        f.attrs['split'] = H5PYDataset.create_split_array(split_dict=split_dict)
        # Save to file
        f.flush()
        f.close()
        # Save Complementary Information
        f = open(directory + '/info.pkl', 'wb')
        dataset_info = {
            'index_to_activity': {i: self.get_activity_by_index(i) for i in range(len(self.get_enabled_activities()))},
            'index_to_feature': {i: self.get_feature_string_by_index(i) for i in range(self.x.shape[1])},
            'activity_info': self.activity_list,
            'sensor_info': self.sensor_list,
            'split_sets': split_set,
            'split_timearray': split_timearray,
            'comments': comments
        }
        pickle.dump(dataset_info, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def write_to_xlsx(self, filename, start=0, end=-1):
        """Write to file in xlsx format

        Args:
            filename (:obj:`str`): xlsx file name.
            start (:obj:`int`): start index.
            end (:obj:`int`): end index.
        """
        workbook = xlsxwriter.Workbook(filename)
        # Dump Activities
        activity_sheet = workbook.add_worksheet("Activities")
        c = 0
        for item in self.activity_list[list(self.activity_list.keys())[0]].keys():
            activity_sheet.write(0, c, str(item))
            c += 1
        r = 1
        for activity in self.activity_list.keys():
            c = 0
            for item in self.activity_list[activity].keys():
                activity_sheet.write(r, c, str(self.activity_list[activity][item]))
                c += 1
            r += 1
        # Dump Sensors
        sensor_sheet = workbook.add_worksheet("Sensors")
        c = 0
        for item in self.sensor_list[list(self.sensor_list.keys())[0]].keys():
            sensor_sheet.write(0, c, str(item))
            c += 1
        r = 1
        for sensor in self.sensor_list.keys():
            c = 0
            for item in self.sensor_list[sensor].keys():
                sensor_sheet.write(r, c, str(self.sensor_list[sensor][item]))
                c += 1
            r += 1
        # Dump Calculated Features
        if self.is_stat_feature:
            # Feature Description Sheet
            feature_sheet = workbook.add_worksheet('Features')
            feature_list_title = ['name', 'index', 'enabled', 'per_sensor', 'description', 'routine']
            for c in range(0, len(feature_list_title)):
                feature_sheet.write(0, c, str(feature_list_title[c]))
            r = 1
            for feature in self.feature_list:
                feature_sheet.write(r, 0, str(self.feature_list[feature].name))
                feature_sheet.write(r, 1, str(self.feature_list[feature].index))
                feature_sheet.write(r, 2, str(self.feature_list[feature].enabled))
                feature_sheet.write(r, 3, str(self.feature_list[feature].per_sensor))
                feature_sheet.write(r, 4, str(self.feature_list[feature].description))
                if self.feature_list[feature].routine is None:
                    feature_sheet.write(r, 5, 'None')
                else:
                    feature_sheet.write(r, 5, str(self.feature_list[feature].routine.name))
                r += 1
        # Dump Events
        if len(self.event_list) != 0:
            event_sheet = workbook.add_worksheet('Events')
            c = 0
            for item in self.event_list[0].keys():
                event_sheet.write(0, c, str(item))
                c += 1
            r = 1
            for event in self.event_list[0:100]:
                c = 0
                for item in event.keys():
                    event_sheet.write(r, c, str(event[item]))
                    c += 1
                r += 1
        # Dump Data
        if self.x is not None:
            data_sheet = workbook.add_worksheet('Data')
            # Export self.x feature
            if self.is_stat_feature:
                data_sheet.write(0, 0, 'activity')
                # Calculate enabled sensor size
                num_sensors = len(self.get_enabled_sensors())
                # Add Feature Title
                for feature_name in self.feature_list.keys():
                    if self.feature_list[feature_name].enabled:
                        if self.feature_list[feature_name].per_sensor:
                            # Calculate Start Position
                            start_col = self.num_static_features + \
                                        self.feature_list[feature_name].index * num_sensors + 1
                            data_sheet.merge_range(0, start_col, 0, start_col + num_sensors - 1, feature_name)
                        else:
                            data_sheet.write(0, self.feature_list[feature_name].index + 1, feature_name)
                for c in range(1, self.num_static_features + 1):
                    data_sheet.write(1, c, 'window')
                for f in range(0, self.num_per_sensor_features):
                    for sensor in self.sensor_list.keys():
                        start_col = f * num_sensors + self.num_static_features + self.sensor_list[sensor]['index'] + 1
                        data_sheet.write(1, start_col, sensor)
                # Add Data from Data Array
                r = 2
                (num_samples, num_features) = self.x.shape
                if end == -1:
                    end = num_samples
                if start < num_samples and start < end:
                    for i in range(start, end):
                        data_sheet.write(r, 0, str(self.y[i]))
                        c = 1
                        for item in self.x[i]:
                            data_sheet.write(r, c, str(item))
                            c += 1
                        r += 1
        workbook.close()
