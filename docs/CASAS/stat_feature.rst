.. _casas_stat_feature_doc_master:

Statistical Features
====================

For activity recognition based on learning algorithms like support vector machine (SVM), decision tree, random forest,
multi-layer perceptron. [Krishnan2014]_ investigated various sliding window approaches to generate such statistical
features.

.. _feature_window_duration:

Window Duration
---------------

.. autoclass:: pyActLearn.CASAS.stat_features.WindowDuration
   :members:
   :show-inheritance:

.. _feature_last_sensor:

Last Sensor
-----------

.. autoclass:: pyActLearn.CASAS.stat_features.LastSensor
   :members:
   :show-inheritance:

.. _feature_event_hour:

Hour of the Event
-----------------

.. autoclass:: pyActLearn.CASAS.stat_features.EventHour
   :members:
   :show-inheritance:

.. _feature_event_seconds:

Seconds of the Event
--------------------

.. autoclass:: pyActLearn.CASAS.stat_features.EventSeconds
   :members:
   :show-inheritance:

.. _feature_sensor_count:

Sensor Count
------------

.. autoclass:: pyActLearn.CASAS.stat_features.SensorCount
   :members:
   :show-inheritance:

.. _feature_elapsed_time:

Sensor Elapse Time
------------------

.. autoclass:: pyActLearn.CASAS.stat_features.SensorElapseTime
   :members:
   :show-inheritance:

.. _feature_dominant_sensor:

Dominant Sensor
---------------

.. autoclass:: pyActLearn.CASAS.stat_features.DominantSensor
   :members:
   :show-inheritance:

Feature Template
----------------

.. autoclass:: pyActLearn.CASAS.stat_features.FeatureTemplate
   :members:
   :show-inheritance:

Feature Update Routine Template
-------------------------------

.. autoclass:: pyActLearn.CASAS.stat_features.FeatureRoutineTemplate
   :members:
   :show-inheritance:
