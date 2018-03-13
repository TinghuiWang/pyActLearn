"""Learning Methods regarding sensors
"""
import numpy as np
import math


def sensor_distance(sensor_list, normalize=True):
    """Calculate direct distance between sensors

    Args:
        sensor_list (:obj:`list` of :obj:`dict`): List of dictionary containing
            sensor information
        normalize (:obj:`bool`): Normalize to the largest sensor distances

    Return:
        :class:`numpy.array`: Sensor distance matrix

    Here is an example of ``sensor_list``::

        [{'sizeY': 0.032722513089005235,
          'locY': 0.7697478270655527,
          'lastFireTime': None,
          'description': '',
          'type': 'Motion',
          'enable': True,
          'name': 'M012',
          'locX': 0.23419495147148758,
          'index': 23,
          'sizeX': 0.03038897817876984},
         {'sizeY': 0.01963350785340314,
          'locY': -0.0022175574177846114,
          'lastFireTime': None,
          'description': '',
          'type': 'Door',
          'enable': True,
          'name': 'D030',
          'locX': 0.6419911140802685,
          'index': 9,
          'sizeX': 0.03160453730592064},
        ...]

    """
    num_sensors = len(sensor_list)
    distance_matrix = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors - 1):
        x1 = sensor_list[i]['locX'] + sensor_list[i]['sizeX'] / 2
        y1 = sensor_list[i]['locY'] + sensor_list[i]['sizeY'] / 2
        for j in range(i+1, num_sensors):
            x2 = sensor_list[j]['locX'] + sensor_list[j]['sizeX'] / 2
            y2 = sensor_list[j]['locY'] + sensor_list[j]['sizeY'] / 2
            # Calculate distance
            distance = math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    # Normalize
    if normalize:
        distance_matrix = distance_matrix / np.max(distance_matrix, axis=1)[:,None]
    # Return distance matrix
    return distance_matrix


def sensor_mi_distance(sensor_list, event_list):
    r"""Compute distance between sensors based on mutual information

    Mutual information is defined as the quantity that measures the mutual dependence
    of two random variables. For sensor-based activity recognition, it is then defined
    as the chance of these two sensors occurring successively in the entire sensor stream.

    Formally, given two sensors :math:`S_i` and :math:`S_j`, the mutual information between
    them, :math:`MI(i, j)`, calculated over a series of :math:`N` events is defined as:

    .. math::

       M(i,j) = \frac{1}{N} \sum_{k=1}^{N-1} \delta \left( S_k, S_i \right) \delta \left( S_{k+1}, S_j) \right)

    where

    .. math::

        \delta \left( S_k, S_i \right) = \begin{cases}
            0,& \text{if } S_k \neq S_i\\
            1,& \text{if } S_k = S_i
        \end{cases}

    D. Cook, Wileybook.

    Args:
        sensor_list (:obj:`list` of :obj:`dict`): List of dictionary containing
            sensor information
        event_list (:obj:`list` of :obj:`dict`): List of events.

    Return:
        :class:`numpy.array`: Sensor distance matrix

    Here is an example of ``event_list``::

        [{'activity': 'Other_Activity',
          'sensor_status': 'ON',
          'resident_name': '',
          'sensor_id': 'M015',
          'datetime': datetime.datetime(2009, 7, 17, 15, 49, 51)},
         {'activity': 'Other_Activity',
          'sensor_status': 'ON',
          'resident_name': '',
          'sensor_id': 'M003',
          'datetime': datetime.datetime(2009, 7, 17, 15, 49, 52)},
         {'activity': 'Other_Activity',
          'sensor_status': 'ON',
          'resident_name': '',
          'sensor_id': 'M007',
          'datetime': datetime.datetime(2009, 7, 17, 15, 49, 58)},
          ...]

    Please refer to :func:`sensor_distance` for an example of ``sensor_list``.
    """
    num_sensors = len(sensor_list)
    distance_matrix = np.zeros((num_sensors, num_sensors))
    num_events = len(event_list)
    sensor_dict = {}
    # Store sensor ID in hash table for faster access
    for i in range(num_sensors):
        sensor_dict[sensor_list[i]['name']] = i
    # Count mutual information
    for i in range(num_events - 1):
        sensor_1 = sensor_dict[event_list[i]['sensor_id']]
        sensor_2 = sensor_dict[event_list[i + 1]['sensor_id']]
        if sensor_1 != sensor_2:
            distance_matrix[sensor_1, sensor_2] += 1
            distance_matrix[sensor_2, sensor_1] += 1
    distance_matrix = distance_matrix / np.max(distance_matrix, axis=1)[:,None]
    distance_matrix = 1. - distance_matrix
    # Diagal is 0
    for i in range(num_sensors):
        distance_matrix[i, i] = 0
    # Return distance matrix
    return distance_matrix


def sensor_vector_distance(sensor_list, window_size=10, normalize=True):
    """Use word2vec technique to map sensors into a high-dimensional space

    Use cosine distance to measure the distance between sensors in the high-dimensional space.
    """
    num_sensors = len(sensor_list)
    distance_matrix = np.zeros((num_sensors, num_sensors))
    return distance_matrix

