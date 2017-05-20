.. _casas_data_doc_master:

CASAS.data
==========

:ref:`casas_data_doc_master` implements :class:`pyActLearn.CASAS.data.CASASData`.

Sensor Event File Format
------------------------

:class:`pyActLearn.CASAS.data.CASASData` can load the smart home raw sensor event logs in raw text (legacy) format,
and comma separated (.csv) files.

Legacy format
^^^^^^^^^^^^^

Here is a snip of sensor event logs of a smart home in raw text format::

   2009-06-01 17:51:20.055202	M046	ON
   2009-06-01 17:51:22.036689	M046	OFF
   2009-06-01 17:51:28.053264	M046	ON
   2009-06-01 17:51:30.072223	M046	OFF
   2009-06-01 17:51:35.046958	M045	OFF
   2009-06-01 17:51:41.096098	M045	ON
   2009-06-01 17:51:44.096236	M046	ON
   2009-06-01 17:51:45.053722	M045	OFF
   2009-06-01 17:51:46.015612	M045	ON
   2009-06-01 17:51:47.005712	M046	OFF
   2009-06-01 17:51:48.004619	M046	ON
   2009-06-01 17:51:49.076356	M046	OFF
   2009-06-01 17:51:50.035392	M046	ON

The following is an example to load the sensor event logs in legacy text format into class
:class:`pyActLearn.CASAS.data.CASASData`.

.. code-block:: python

   from pyActLearn.CASAS.data import CASASData
   data = CASASData(path='twor.summer.2009/annotate')

CSV format
^^^^^^^^^^

Some of the smart home data set are updated to CSV format. Those datasets usually come with meta-data about the smart
home including floorplan, sensor location, activities annotated, and other information.

The binary sensor events are logged inside file ``event.csv``. Here is a snip of it::

   2/1/2009,8:00:38 AM,M048,OFF,,,
   2/1/2009,8:00:38 AM,M049,OFF,,,
   2/1/2009,8:00:39 AM,M028,ON,,,
   2/1/2009,8:00:39 AM,M042,ON,,,
   2/1/2009,8:00:40 AM,M029,ON,,,
   2/1/2009,8:00:40 AM,M042,OFF,,,
   2/1/2009,8:00:40 AM,L003,OFF,,,
   2/1/2009,8:00:42 AM,M043,OFF,,,
   2/1/2009,8:00:42 AM,M037,ON,,,
   2/1/2009,8:00:42 AM,M050,OFF,,,
   2/1/2009,8:00:42 AM,M044,OFF,,,
   2/1/2009,8:00:42 AM,M028,OFF,,,
   2/1/2009,8:00:43 AM,M029,OFF,,,

The metadata about the smart home is in a json file format. Here is a snip of the metadata for twor dataset:

.. code-block:: json

   {
   "name": "TWOR_2009_test",
   "floorplan": "TWOR_2009.png",
   "sensors": [
      {
         "name": "M004",
         "type": "Motion",
         "locX": 0.5605087077755726,
         "locY": 0.061440840882448416,
         "sizeX": 0.0222007722007722,
         "sizeY": 0.018656716417910446,
         "description": ""
      },
   ],
   "activities": [
      {
         "name": "Meal Preparation",
         "color": "#FF8A2BE2",
         "is_noise": false,
         "is_ignored": false
      },
   ]}

To load such a dataset, provide the directory path to the constructor of :class:`pyActLearn.CASAS.data.CASASData`.

.. code-block:: python

   from pyActLearn.CASAS.data import CASASData
   data = CASASData(path='twor.summer.2009/')

.. note::

   The constructor of :class:`pyActLearn.CASAS.data.CASASData` differentiates the format of sensor log by
   determining whether the path is a directory or file. If it is a file, it assumes that it is in legacy raw
   text format. If it is a directory, the constructor looks for ``event.csv`` file within the directory for
   binary sensor events, and ``dataset.json`` for mete-data about the smart home.

Event Pre-processing
--------------------

Raw sensor event data may need to be pre-processed before the learning algorithm can consume them. For algorithms
like Hidden Markov Model, only raw sensor series are needed. For algorithms like decision tree, random forest,
multi-layer perceptron, etc., statistic features within a sliding window of fixed length or variable length are
calculated. For data used in stacked auto-encoder, the input needs to be normalized between 0 to 1.

:func:`pyActLearn.CASAS.data.CASASData.populate_feature` function handles the pre-processing of all binary sensor
events. The statistical features implemented in this function includes

- :ref:`feature_window_duration`
- :ref:`feature_last_sensor`
- :ref:`feature_event_hour`
- :ref:`feature_event_seconds`
- :ref:`feature_sensor_count`
- :ref:`feature_elapsed_time`
- :ref:`feature_dominant_sensor`

Methods to enable and disable specific features or activities are provided as well.
Please refer to :class:`pyActLearn.CASAS.data.CASASData` API reference for more information.

Export Data
-----------

After the data are pre-processed, the features and labels can be exported to excel file (.xlsx) via function
:func:`pyActLearn.CASAS.data.CASASData.write_to_xlsx`.

:func:`pyActLearn.CASAS.data.CASASData.export_hdf5` will save the pre-processed features and target labels in
hdf5 format. The meta-data is saved as attributes of the root node of hdf5 dataset.
The hdf5 file can be viewed using hdfviewer_.

.. _hdfviewer: https://support.hdfgroup.org/products/java/hdfview/

Here is an example loading raw sensor events and save to hdf5 dataset file.

.. code-block:: python

   from pyActLearn.CASAS.data import CASASData
   data = CASASData(path='datasets/twor.2009/')
   data.populate_feature(method='stat', normalized=True, per_sensor=True)
   data..export_hdf5(filename='hdf5/twor_2009_stat.hdf5', comments='')

API Reference
-------------

.. automodule:: pyActLearn.CASAS.data
   :members:
   :show-inheritance:
