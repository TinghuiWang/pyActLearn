.. pyActLearn documentation master file, created by
   sphinx-quickstart on Mon Nov 14 10:34:33 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyActLearn's documentation!
======================================

pyActLearn is an activity recognition platform designed to recognize ADL
(Activities of Daily Living) in smart homes equipped with less intrusive
passive monitoring sensors, such as motion detectors, door sensors,
thermometers, light switches, etc.

Components
----------

:ref:`casas_doc_master`
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`casas_doc_master` contains classes and functions that load and pre-process smart home sensor
event data. The pre-processed data are stored in an hdf5 data format with smart home information
stored as attributes of the dataset. The processed data are splitted into weeks and days.
Class :class:`pyActLearn.CASAS.hdf5.CASASHDF5` can load the hdf5 dataset and use as a feeder for
activity recognition learning algorithm.

:ref:`learning_doc_master`
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`learning_doc_master` contains classes and functions that implement supervised and unsupervised
learning algorithms for activity recognition. Some of the classes refers to models provided by other
python packages such as hmmlearn (for multinomial hidden markov models) and sklearn (for support
vector machine, decision tree, and random forest).

:ref:`performance_doc_master`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`performance_doc_master` contains classes and functions that implement multiple performance
metrics for activity recognition, including confusion matrix, multi-class classification metrics,
event-based scoring, and activity timeliness.

Roadmap
-------

- Data Loading

  - [X] Load event list from legacy raw text files
  - [X] Load event list from csv event files
  - [X] Load sensor information from JSON meta-data file
  - [X] Divide event list by days or weeks

- Pre-processing

  - [X] Statistical feature extraction using sliding window approach
  - [X] Raw interpretation

- Algorithm implementations

  - Supervised Learning

    - [X] Decision Tree
    - [X] HMM
    - [X] SVM
    - [X] Multi-layer Perceptron
    - [X] Stacked De-noising Auto-encoder with fine tuning
    - [X] Recurrent Neural Network with LSTM Cell
    - [ ] Recurrent Neural Network with GRU

  - Un-supervised Learning

    - [X] Stacked De-noising Auto-encoder
    - [X] k-skip-2-gram with Negative Sampling (word2vec)

  - Transfer Learning

- Evaluation

  - [ ] n-Fold Cross-validation
  - [X] Traditional Multi-class Classification Metrics
  - [X] Event-based Continuous Evaluation Metrics
  - [X] Event-based Activity Diagram

- Annotation

  - [X] Back annotate dataset with predicted results
  - [X] Back annotate with probability

- Visualization

  - [X] Sensor distance on floor plan

.. toctree::
   :maxdepth: 2
   :hidden:

   Installation
   CASAS/index
   learning/index
   performance/index
   reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
