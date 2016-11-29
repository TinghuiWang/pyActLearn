import json
import os
import logging

logger = logging.getLogger(__name__)


class CASASHome:
    """Load Home Data Structure from JSON file

    Attributes:
        data_dict (:obj:`dict`): A dictionary contains information about smart home.

    Parameters:
        directory (:obj:`str`): Directory that stores CASAS smart home data
    """
    def __init__(self, directory):
        dataset_json_fname = directory + '/dataset.json'
        if os.path.exists(dataset_json_fname):
            f = open(dataset_json_fname, 'r')
            self.data_dict = json.load(f)
        else:
            logger.error('Smart home metadata file %s does not exist. Create an empty CASASHome Structure'
                        % dataset_json_fname)
            raise FileNotFoundError('File %s not found.' % dataset_json_fname)
            # self.data_dict = {
            #     'name': '',
            #     'floorplan': '',
            #     'sensors': [],
            #     'activities': [],
            #     'residents': []
            # }

    def get_name(self):
        """Get the smart home name

        Returns:
            :obj:`str`: smart home name
        """
        return self.data_dict['name']

    def get_all_activities(self):
        """Get All Activities

        Returns:
            :obj:`list` of :obj:`str`: list of activity names
        """
        names = [activity['name'] for activity in self.data_dict['activities']]
        return names

    def get_activity(self, label):
        """Find the information about the activity

        Parameters:
            label (:obj:`str`): activity label

        Returns:
            :obj:`dict`: A dictionary containing activity information
        """
        for activity in self.data_dict['activities']:
            if activity['name'] == label:
                return activity
        return None

    def get_activity_color(self, label):
        """Find the color string of the activity

        Parameters:
            label (:obj:`str`): activity label

        Returns:
            :obj:`str`: RGB color string
        """
        activity = self.get_activity(label)
        if activity is not None:
            return "#" + activity['color'][3:9]
        else:
            raise ValueError('Activity %s Not Found' % label)

    def get_sensor(self, name):
        """Get the information about the sensor

        Parameters:
            name (:obj:`str`): name of the sensor

        Returns:
            :obj:`dict`: A dictionary that stores sensor information
        """
        for sensor in self.data_dict['sensors']:
            if sensor['name'] == name:
                return sensor
        return None

    def get_all_sensors(self):
        """Get All Sensor Names

        Returns:
            :obj:`list` of :obj:`str`: a list of sensor names
        """
        names = [sensor['name'] for sensor in self.data_dict['sensors']]
        return names

    def get_resident(self, name):
        """Get Information about the resident

        Parameters:
            name (:obj:`str`): name of the resident

        Returns:
            :obj:`dict`: A Dictionary that stores resident information
        """
        for resident in self.data_dict['residents']:
            if resident['name'] == name:
                return resident
        return None

    def get_resident_color(self, name):
        """Get the color string for the resident

        Parameters:
            name (:obj:`str`): name of the resident

        Returns:
            :obj:`str`: RGB color string representing the resident
        """
        resident = self.get_resident(name)
        if resident is not None:
            return "#" + resident['color'][3:9]
        else:
            raise ValueError('Resident %s Not Found' % name)

    def get_all_residents(self):
        """Get All Resident Names

        Returns:
            :obj:`list` of :obj:`str`: A list of resident names
        """
        names = [resident['name'] for resident in self.data_dict['residents']]
        return names
