import json
import os
import logging
import numpy as np
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..utils.LabeledLine import LabeledLine

logger = logging.getLogger(__name__)


# noinspection PyPackageRequirements
class CASASHome:
    """Load Home Data Structure from JSON file

    Attributes:
        data_dict (:obj:`dict`): A dictionary contains information about smart home.
        directory (:obj:`str`): Directory that stores CASAS smart home data

    Parameters:
        directory (:obj:`str`): Directory that stores CASAS smart home data
    """
    def __init__(self, directory):
        self.directory = directory
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

    def _prepare_floorplan(self):
        """Prepare the floorplan for drawing

        Returns:
            :obj:`dict`: A dictionary contains all the pieces needed to draw the floorplan
        """
        floorplan_dict = {}
        img = mimg.imread(os.path.join(self.directory, self.data_dict['floorplan']))
        img_x = img.shape[1]
        img_y = img.shape[0]
        # Create Sensor List/Patches
        sensor_boxes = {}
        sensor_texts = {}
        sensor_centers = {}
        # Check Bias
        for sensor in self.data_dict['sensors']:
            loc_x = sensor['locX'] * img_x
            loc_y = sensor['locY'] * img_y
            size_x = sensor['sizeX'] * img_x
            size_y = sensor['sizeY'] * img_y
            sensor_center_x = loc_x + size_x / 2
            sensor_center_y = loc_y + size_y / 2
            sensor_boxes[sensor['name']] = \
                patches.Rectangle((loc_x, loc_y), size_x, size_y,
                                  edgecolor='grey', facecolor='orange', linewidth=1,
                                  zorder=2)
            sensor_texts[sensor['name']] = (loc_x + size_x / 2, loc_y + size_y / 2, sensor['name'])
            sensor_centers[sensor['name']] = (sensor_center_x, sensor_center_y)
        # Populate dictionary
        floorplan_dict['img'] = img
        floorplan_dict['width'] = img_x
        floorplan_dict['height'] = img_y
        floorplan_dict['sensor_centers'] = sensor_centers
        floorplan_dict['sensor_boxes'] = sensor_boxes
        floorplan_dict['sensor_texts'] = sensor_texts
        return floorplan_dict

    def draw_floorplan(self, filename=None):
        """Draw the floorplan of the house, save it to file or display it on screen

        Args:
            filename (:obj:`str`): Name of the file to save the floorplan to
        """
        floorplan_dict = self._prepare_floorplan()
        self._plot_floorplan(floorplan_dict, filename)

    @staticmethod
    def _plot_floorplan(floorplan_dict, filename=None):
        fig, (ax) = plt.subplots(1, 1)
        fig.set_size_inches(18, 18)
        ax.imshow(floorplan_dict['img'])
        # Draw Sensor block patches
        for key, patch in floorplan_dict['sensor_boxes'].items():
            ax.add_patch(patch)
        # Draw Sensor name
        for key, text in floorplan_dict['sensor_texts'].items():
            ax.text(*text, color='black',
                    horizontalalignment='center', verticalalignment='center',
                    zorder=3)
        if floorplan_dict.get('sensor_lines', None) is not None:
            for key, line in floorplan_dict['sensor_lines'].items():
                ax.add_line(line)
        if filename is None:
            # Show image
            fig.show()
        else:
            fig.savefig(filename)
            plt.close(fig)

    def plot_sensor_distance(self, sensor_name, distance_matrix, max_sensors=None, filename=None):
        """Plot distance in distance_matrix
        """
        sensor_index = self.get_all_sensors().index(sensor_name)
        num_sensors = len(self.data_dict['sensors'])
        floorplan_dict = self._prepare_floorplan()
        x1 = floorplan_dict['sensor_centers'][sensor_name][0]
        y1 = floorplan_dict['sensor_centers'][sensor_name][1]
        # Draw Lines, and Set alpha for each sensor box
        sensor_lines ={}
        for i in range(num_sensors):
            sensor = self.data_dict['sensors'][i]
            if sensor_name != sensor['name']:
                x2 = floorplan_dict['sensor_centers'][sensor['name']][0]
                y2 = floorplan_dict['sensor_centers'][sensor['name']][1]
                line = LabeledLine([x1, x2], [y1, y2], linewidth=1,
                                   linestyle='--', color='b', zorder=10,
                                   label='%.5f' % distance_matrix[sensor_index, i],
                                   alpha=(1 - distance_matrix[sensor_index, i]) * 0.9 + 0.1)
                sensor_lines[sensor['name']] = line
                floorplan_dict['sensor_boxes'][sensor['name']].set_alpha(1 - distance_matrix[sensor_index, i])
        # Only show up to `max_lines` of sensors
        if max_sensors is not None and max_sensors < num_sensors:
            sorted_index = np.argsort(distance_matrix[sensor_index, :])
            for i in range(max_sensors + 1, num_sensors):
                sensor_lines.pop(self.data_dict['sensors'][sorted_index[i]]['name'], None)
        floorplan_dict['sensor_lines'] = sensor_lines
        self._plot_floorplan(floorplan_dict, filename)
