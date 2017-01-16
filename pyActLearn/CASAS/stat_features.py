import abc
import logging
import numpy as np
from ..logging import logging_name

logger = logging.getLogger(__name__)


# region Abstract FeatureRoutineTemplate Class
class FeatureRoutineTemplate(metaclass=abc.ABCMeta):
    """Feature Routine Class

    A routine that calculate statistical features every time the window slides.

    Attributes:
        name (:obj:`str`): Feature routine name.
        description (:obj:`str`): Feature routine description.
        enabled (:obj:`str`): Feature routine enable flag.
    """
    def __init__(self, name, description, enabled=True):
        """
        Initialization of Template Class
        :return:
        """
        # Name
        self.name = name
        # Description
        self.description = description
        # enable
        self.enabled = enabled

    @abc.abstractmethod
    def update(self, data_list, cur_index, window_size, sensor_info):
        """Abstract update method

        For some features, we will update some statistical data every time
        we move forward a data record, instead of going back through the whole
        window and try to find the answer. This function will be called every time
        we advance in data record.

        Args:
            data_list (:obj:`list`): List of sensor data.
            cur_index (:obj:`int`): Index of current data record.
            window_size (:obj:`int`): Sliding window size.
            sensor_info (:obj:`dict`): Dictionary containing sensor index information.
        """
        return NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        """Clear Internal Data Structures if recalculation is needed
        """
        return NotImplementedError()
# endregion


# region Abstract FeatureTemplate Class
class FeatureTemplate(metaclass=abc.ABCMeta):
    """Statistical Feature Template

    Args:
        name (:obj:`str`): Feature name.
        description (:obj:`str`): Feature description.
        per_sensor (:obj:`bool`): If the feature is calculated for each sensor.
        enabled (:obj:`bool`): If the feature is enabled.
        routine (:obj:`.FeatureRoutineTemplate`): Routine structure.
        normalized (:obj:`bool`): If the value of feature needs to be normalized.

    Attributes:
        name (:obj:`str`): Feature name.
        description (:obj:`str`): Feature description.
        index (:obj:`int`): Feature index.
        normalized (:obj:`bool`): If the value of feature needs to be normalized.
        per_sensor (:obj:`bool`): If the feature is calculated for each sensor.
        enabled (:obj:`bool`): If the feature is enabled.
        routine (:obj:`.FeatureRoutineTemplate`): Routine structure.
        _is_value_valid (:obj:`bool`): If the value calculated is valid
    """
    def __init__(self, name, description, enabled=True, normalized=True, per_sensor=False, routine=None):
        self.name = name
        self.description = description
        self.index = -1
        self.normalized = normalized
        self.per_sensor = per_sensor
        self.enabled = enabled
        self._is_value_valid = False
        # update Routine
        # For some feature, we will update statistical data every time we move forward
        # a data record. Instead of going back through previous window, the update function
        # in this routine structure will be called each time we advance to next data record
        self.routine = routine

    @abc.abstractmethod
    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Abstract method to get feature value

        Args:
            data_list (:obj:`list`): List of sensor data.
            cur_index (:obj:`int`): Index of current data record.
            window_size (:obj:`int`): Sliding window size.
            sensor_info (:obj:`dict`): Dictionary containing sensor index information.
            sensor_name (:obj:`str`): Sensor Name.

        Returns:
            :obj:`double`: feature value
        """
        return NotImplementedError()

    @property
    def is_value_valid(self):
        """Statistical feature value valid check

        Due to errors and failures of sensors, the statistical feature calculated
        may go out of bound. This abstract method is used to check if the value
        calculated is valid. If not, it will not be inserted into feature vectors.

        Returns:
            :obj:`bool`: True if the result is valid.
        """
        return self._is_value_valid
# endregion


class EventHour(FeatureTemplate):
    """Show the hour of the time of current event
    """
    def __init__(self, normalized=False):
        super().__init__(name='lastEventHour',
                         description='Time of the last sensor event in window (hour)',
                         normalized=normalized,
                         per_sensor=False,
                         enabled=True,
                         routine=None)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Get the hour when the last sensor event in the window occurred

        Note:
            Please refer to :meth:`~.FeatureTemplate.get_feature_value` for information about
            parameters.
        """
        self._is_value_valid = True
        if self.normalized:
            return np.float(data_list[cur_index]['datetime'].hour)/24
        else:
            return np.float(data_list[cur_index]['datetime'].hour)


class EventSeconds(FeatureTemplate):
    """Feature that shows the time (min, sec) of current event in seconds
    """
    def __init__(self, normalized=False):
        super().__init__(
            name='lastEventSeconds',
            description='Time of the last sensor event in window in seconds',
            normalized=normalized,
            per_sensor=False,
            enabled=True,
            routine=None)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Get the time within an hour when the last sensor event in the window occurred (in seconds)

        Note:
            Please refer to :meth:`~.FeatureTemplate.get_feature_value` for information about
            parameters.
        """
        self._is_value_valid = True
        time = data_list[cur_index]['datetime']
        if self.normalized:
            return np.float((time.minute * 60) + time.second)/3600
        else:
            return np.float((time.minute * 60) + time.second)


class WindowDuration(FeatureTemplate):
    """Length of the window in seconds
    """
    def __init__(self, normalized=False):
        super().__init__(name='windowDuration',
                         description='Duration of current window in seconds',
                         normalized=normalized,
                         per_sensor=False,
                         enabled=True,
                         routine=None)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Get the duration of the window in seconds. Invalid if the duration is greater than half a day.

        Note:
            Please refer to :meth:`~.FeatureTemplate.get_feature_value` for information about
            parameters.
        """
        self._is_value_valid = True
        timedelta = data_list[cur_index]['datetime'] - data_list[cur_index - window_size + 1]['datetime']
        window_duration = timedelta.total_seconds()
        if window_duration > 3600 * 12:
            self._is_value_valid = False
            # Window Duration is greater than a day - not possible
            # print('Warning: curIndex: %d; windowSize: %d; windowDuration: %f' %
            # (curIndex, windowSize, window_duration))
            window_duration -= 3600 * 12 * (int(window_duration) / (3600 * 12))
            # print('Fixed window duration %f' % window_duration)
            if data_list[cur_index]['datetime'].month != data_list[cur_index - 1]['datetime'].month or \
                    data_list[cur_index]['datetime'].day != data_list[cur_index - 1]['datetime'].day:
                date_advanced = (data_list[cur_index]['datetime'] - data_list[cur_index - 1]['datetime']).days
                hour_advanced = data_list[cur_index]['datetime'].hour - data_list[cur_index - 1]['datetime'].hour
                logger.warn(logging_name(self) + ': line %d - %d: %s' %
                            (cur_index, cur_index + 1, data_list[cur_index - 1]['datetime'].isoformat()))
                logger.warn(logging_name(self) + ': Date Advanced: %d; hour gap: %d' % (date_advanced, hour_advanced))
        if self.normalized:
            # Normalized to 12 hours
            return np.float(window_duration) / (3600 * 12)
        else:
            return np.float(window_duration)


class LastSensor(FeatureTemplate):
    """Get the last sensor in the window
    """
    def __init__(self, per_sensor=False):
        super().__init__(name='lastSensorInWindow',
                         description='Sensor ID in the current window',
                         per_sensor=per_sensor,
                         enabled=True,
                         routine=None)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Get the sensor which fired the last event in the sliding window.

        If it is configured as per-sensor feature, it returns 1 if the sensor specified
        triggers the last event in the window. Otherwise returns 0.
        If it is configured as a non-per-sensor feature, it returns the index of the
        index corresponding to the dominant sensor name that triggered the last event.

        Note:
            Please refer to :meth:`~.FeatureTemplate.get_feature_value` for information about
            parameters.
        """
        self._is_value_valid = True
        sensor_label = data_list[cur_index]['sensor_id']
        if self.per_sensor:
            if sensor_name is not None:
                if sensor_name == sensor_label:
                    return 1
                else:
                    return 0
        else:
            if sensor_info.get(sensor_label, None) is None:
                self._is_value_valid = False
                logger.warn(logging_name(self) + ': Cannot find sensor %s in sensor_info' % sensor_label)
                logger.debug(logging_name(self) + ': Available sensors are: ' + str(sensor_info.keys()))
                return 0
            else:
                return sensor_info[sensor_label]['index']


class SensorCountRoutine(FeatureRoutineTemplate):
    """Routine to count occurance of each sensor

    Attributes:
        sensor_count (:obj:`dict`): Dictionary that counts the occurrance of each sensor
    """
    def __init__(self):
        super().__init__(
            name='SensorCountRoutine',
            description='Count Occurrence of all sensors in current event window',
            enabled=True
        )
        # Dominant Sensor
        self.sensor_count = {}

    def update(self, data_list, cur_index, window_size, sensor_info):
        """Record the number of occurrence of each sensor in the sensor count dictionary.
        """
        self.sensor_count = {}
        for sensor_label in sensor_info.keys():
            if sensor_info[sensor_label]['enable']:
                self.sensor_count[sensor_label] = 0
        for index in range(0, window_size):
            if data_list[cur_index - index]['sensor_id'] in self.sensor_count.keys():
                self.sensor_count[data_list[cur_index - index]['sensor_id']] += 1

    def clear(self):
        self.sensor_count = {}

sensor_count_routine = SensorCountRoutine()


class SensorCount(FeatureTemplate):
    """Counts the occurrence of each sensor
    """
    def __init__(self, normalized=False):
        super().__init__(name='sensorCount',
                         description='Number of Events in the window related to the sensor',
                         normalized=normalized,
                         per_sensor=True,
                         enabled=True,
                         routine=sensor_count_routine)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Counts the number of occurrence of the sensor specified in current window.
        """
        count = self.routine.sensor_count.get(sensor_name, None)
        if count is None:
            logger.error(logging_name(self) + ': Cannot find sensor %s in sensor list' % sensor_name)
            self._is_value_valid = False
        else:
            self._is_value_valid = True
            if self.normalized:
                return float(count)/(window_size * 2)
            else:
                return float(count)


class SensorElapseTimeRoutine(FeatureRoutineTemplate):
    """Routine to record last occurrence of each sensor

    Attributes:
        sensor_fire_log (:obj:`dict`): Dictionary that record the last firing state of each sensor
    """
    def __init__(self):
        super().__init__(name='SensorElapseTimeUpdateRoutine',
                         description='Update Sensor Elapse Time for all enabled sensors',
                         enabled=True)
        # Sensor Fire Log
        self.sensor_fire_log = {}

    def update(self, data_list, cur_index, window_size, sensor_info):
        """Record the number of occurrence of each sensor in the sensor count dictionary.
        """
        if not self.sensor_fire_log:
            for sensor_label in sensor_info.keys():
                self.sensor_fire_log[sensor_label] = data_list[cur_index - window_size + 1]['datetime']
            for i in range(0, window_size):
                self.sensor_fire_log[data_list[cur_index - i]['sensor_id']] = data_list[cur_index - i]['datetime']
        self.sensor_fire_log[data_list[cur_index]['sensor_id']] = data_list[cur_index]['datetime']

    def clear(self):
        self.sensor_fire_log = {}

sensor_elapse_time_routine = SensorElapseTimeRoutine()


class SensorElapseTime(FeatureTemplate):
    """The time elapsed since last firing (in seconds)
    """
    def __init__(self, normalized=False):
        super().__init__(name='sensorElapseTime',
                         description='Time since each sensor fired (in seconds)',
                         normalized=normalized,
                         per_sensor=True,
                         enabled=True,
                         routine=sensor_elapse_time_routine)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """Get elapse time of specified sensor in seconds
        """
        self._is_value_valid = True
        timedelta = data_list[cur_index]['datetime'] - self.routine.sensor_fire_log[sensor_name]
        sensor_duration = timedelta.total_seconds()
        if self.normalized:
            elapse_time = float(sensor_duration)/(12*3600)
            # If the sensor is not fired in past 12 hours, just round it up to 12 hours
            if elapse_time > 1:
                elapse_time = 1.
            return elapse_time
        else:
            return float(sensor_duration)


class DominantSensorRoutine(FeatureRoutineTemplate):
    """Routine to record the occurance of each sensor within the sliding window

    Attributes:
        dominant_sensor_list (:obj:`dict`): Dictionary that record the last firing state of each sensor
    """
    def __init__(self):
        super().__init__(name='DominantSensorRoutine',
                         description='DominantSensorUpdateRoutine',
                         enabled=True)
        # Dominant Sensor
        self.dominant_sensor_list = {}

    def update(self, data_list, cur_index, window_size, sensor_info):
        """Calculate the dominant sensor of current window and store
        the name of the sensor in the dominant sensor array. The
        information is fetched by dominant sensor features.
        """
        if cur_index < window_size:
            logger.warn(logging_name(self) + ': current index %d is smaller than window size %d.' % (cur_index, window_size))
        sensor_count = {}
        for index in range(0, window_size):
            if data_list[cur_index - index]['sensor_id'] in sensor_count.keys():
                sensor_count[data_list[cur_index - index]['sensor_id']] += 1
            else:
                sensor_count[data_list[cur_index - index]['sensor_id']] = 1
        # Find the Dominant one
        max_count = 0
        for sensor_label in sensor_count.keys():
            if sensor_count[sensor_label] > max_count:
                max_count = sensor_count[sensor_label]
                self.dominant_sensor_list[cur_index] = sensor_label

    def clear(self):
        self.dominant_sensor_list = {}

dominant_sensor_routine = DominantSensorRoutine()


class DominantSensor(FeatureTemplate):
    """Dominant Sensor of current window
    """
    def __init__(self, per_sensor=False):
        super().__init__(name='DominantSensor',
                         description='Dominant Sensor in the window',
                         normalized=True,
                         per_sensor=per_sensor,
                         enabled=True,
                         routine=dominant_sensor_routine)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """If per_sensor is True, returns 1 with corresponding sensor Id.
        otherwise, return the index of last sensor in the window
        """
        self._is_value_valid = True
        dominant_sensor_label = self.routine.dominant_sensor_list.get(cur_index, None)
        if dominant_sensor_label is None:
            logger.warn(logging_name(self) + ': cannot find dominant sensor label for window index %d' % cur_index)
        if self.per_sensor:
            if sensor_name is not None:
                if sensor_name == dominant_sensor_label:
                    return 1
                else:
                    return 0
        else:
            return sensor_info[dominant_sensor_label]['index']


class DominantSensorPreviousWindow(FeatureTemplate):
    """Dominant Sensor of previous window
    """
    def __init__(self, per_sensor=False):
        super().__init__(name='DominantSensorPreviousWindow',
                         description='Dominant Sensor in the previous window',
                         normalized=True,
                         per_sensor=per_sensor,
                         enabled=True,
                         routine=dominant_sensor_routine)

    def get_feature_value(self, data_list, cur_index, window_size, sensor_info, sensor_name=None):
        """If per_sensor is True, returns 1 with corresponding sensor Id.
        otherwise, return the index of last sensor in the window
        """
        dominant_sensor_label = self.routine.dominant_sensor_list.get([cur_index-1], None)
        if dominant_sensor_label is None:
            logger.warn(logging_name(self) + ': cannot find dominant sensor label for window index %d' % cur_index)
        if self.per_sensor:
            if sensor_name is not None:
                if sensor_name == dominant_sensor_label:
                    return 1
                else:
                    return 0
        else:
            return sensor_info[dominant_sensor_label]['index']
