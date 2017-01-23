import os
import pickle
import logging
import argparse
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from pyActLearn.CASAS.data import CASASData
from pyActLearn.CASAS.fuel import CASASFuel
from pyActLearn.performance.record import LearningResult
from pyActLearn.performance import get_confusion_matrix

logger = logging.getLogger(__file__)


def training_and_test(token, train_data, test_data, num_classes, result):
    """Train and test

    Args:
        token (:obj:`str`): token representing this run
        train_data (:obj:`tuple` of :obj:`numpy.array`): Tuple of training feature and label
        test_data (:obj:`tuple` of :obj:`numpy.array`): Tuple of testing feature and label
        num_classes (:obj:`int`): Number of classes
        result (:obj:`pyActLearn.performance.record.LearningResult`): LearningResult object to hold learning result
    """
    model = RandomForestClassifier(n_estimators=20, criterion="entropy")
    model.fit(train_data[0], train_data[1].flatten())
    # Test
    predicted_y = model.predict(test_data[0])
    predicted_proba = model.predict_proba(test_data[0])
    # Evaluate the Test and Store Result
    confusion_matrix = get_confusion_matrix(num_classes=num_classes,
                                            label=test_data[1].flatten(), predicted=predicted_y)
    result.add_record(model.get_params(), key=token, confusion_matrix=confusion_matrix)
    # In case any label is missing, populate it
    if predicted_proba.shape[1] != num_classes:
        temp_array = np.zeros((predicted_proba.shape[0], num_classes), np.float32)
        for i in range(len(model.classes_)):
            temp_array[:, model.classes_[i]] = predicted_proba[:, i]
        predicted_proba = temp_array
    return predicted_y, predicted_proba


def load_and_test(token, test_data, num_classes, result):
    """Load and test

    Args:
        token (:obj:`str`): token representing this run
        test_data (:obj:`tuple` of :obj:`numpy.array`): Tuple of testing feature and label
        num_classes (:obj:`int`): Number of classes
        result (:obj:`pyActLearn.performance.record.LearningResult`): LearningResult object to hold learning result
    """
    model = RandomForestClassifier(n_estimators=20, criterion="entropy")
    model.set_params(result.get_record_by_key(token)['model'])
    # Test
    predicted_y = model.predict(test_data[0])
    predicted_proba = model.predict_proba(test_data[0])
    return predicted_y, predicted_proba

if __name__ == '__main__':
    args_ok = False
    parser = argparse.ArgumentParser(description='Run Random Forest on single resident CASAS datasets.')
    parser.add_argument('-d', '--dataset', help='Directory to original datasets')
    parser.add_argument('-o', '--output', help='Output folder')
    parser.add_argument('--h5py', help='HDF5 dataset folder')
    args = parser.parse_args()
    # Default parameters
    log_filename = os.path.basename(__file__).split('.')[0] + \
                   '-%s.log' % datetime.now().strftime('%y%m%d_%H:%M:%S')
    # Setup output directory
    output_dir = args.output
    if output_dir is not None:
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if os.path.exists(output_dir):
            # Found output_dir, check if it is a directory
            if not os.path.isdir(output_dir):
                exit('Output directory %s is found, but not a directory. Abort.' % output_dir)
        else:
            # Create directory
            os.makedirs(output_dir)
    else:
        output_dir = '.'
    log_filename = os.path.join(output_dir, log_filename)
    # Setup Logging as early as possible
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] %(name)s:%(levelname)s:%(message)s',
                        handlers=[logging.FileHandler(log_filename),
                                  logging.StreamHandler()])
    # If dataset is specified, update h5py
    casas_data_dir = args.dataset
    if casas_data_dir is not None:
        casas_data_dir = os.path.abspath(os.path.expanduser(casas_data_dir))
        if not os.path.isdir(casas_data_dir):
            exit('CASAS dataset at %s does not exist. Abort.' % casas_data_dir)
    # Find h5py dataset first
    h5py_dir = args.h5py
    if h5py_dir is not None:
        h5py_dir = os.path.abspath(os.path.expanduser(h5py_dir))
    else:
        # Default location
        h5py_dir = os.path.join(output_dir, 'h5py')
    if os.path.exists(h5py_dir):
        if not os.path.isdir(h5py_dir):
            exit('h5py dataset location %s is not a directory. Abort.' % h5py_dir)
    if not CASASFuel.files_exist(h5py_dir):
        # Finish check and creating all directory needed - now load datasets
        if casas_data_dir is not None:
            casas_data = CASASData(path=casas_data_dir)
            casas_data.summary()
            # SVM needs to use statistical feature with per-sensor and normalization
            casas_data.populate_feature(method='stat', normalized=False, per_sensor=False)
            casas_data.export_hdf5(h5py_dir)
    casas_fuel = CASASFuel(dir_name=h5py_dir)
    # Prepare learning result
    result_pkl_file = os.path.join(output_dir, 'result.pkl')
    result = None
    if os.path.isfile(result_pkl_file):
        f = open(result_pkl_file, 'rb')
        result = pickle.load(f)
        f.close()
        if result.data != h5py_dir:
            logger.error('Result pickle file found for different dataset %s' % result.data)
            exit('Cannot save learning result at %s' % result_pkl_file)
    else:
        result = LearningResult(name='DecisionTree', data=h5py_dir, mode='by_week')
    num_classes = casas_fuel.get_output_dims()
    # Open Fuel and get all splits
    split_list = casas_fuel.get_set_list()
    train_name = split_list[0]
    train_set = casas_fuel.get_dataset((train_name,), load_in_memory=True)
    (train_set_data) = train_set.data_sources
    # Prepare Back Annotation
    fp_back_annotated = open(os.path.join(output_dir, 'back_annotated.txt'), 'w')
    fp_back_probability = open(os.path.join(output_dir, 'back_annotated_proba.txt'), 'w')
    for i in range(1, len(split_list)):
        test_name = split_list[i]
        test_set = casas_fuel.get_dataset((test_name,), load_in_memory=True)
        (test_set_data) = test_set.data_sources
        # run svm
        logger.info('Training on %s, Testing on %s' % (train_name, test_name))
        if result.get_record_by_key(test_name) is None:
            prediction, prediction_proba = training_and_test(test_name, train_set_data, test_set_data, num_classes, result)
        else:
            prediction, prediction_proba = load_and_test(test_name, test_set_data, num_classes, result)
        casas_fuel.back_annotate(fp_back_annotated, prediction=prediction, split_id=i)
        casas_fuel.back_annotate_with_proba(fp_back_probability, prediction_proba, split_id=i)
        train_name = test_name
        train_set_data = test_set_data
    fp_back_annotated.close()
    fp_back_probability.close()
    f = open(result_pkl_file, 'wb')
    pickle.dump(obj=result, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    result.export_to_xlsx(os.path.join(output_dir, 'result.xlsx'))

