import h5py
import pickle
import logging
import xlsxwriter
import collections
from collections import OrderedDict
from datetime import datetime
import numpy as np
from . import overall_performance_index, per_class_performance_index, get_performance_array, get_confusion_matrix
from .event import score_segment
from ..logging import logging_name
from ..CASAS.fuel import CASASFuel

logger = logging.getLogger(__name__)


class LearningResult:
    """LearningResult is a class that stores results of a learning run.
    
    It may be a single-shot run or a time-based analysis. The result structure holds the parameters for the model 
    as well as the evaluation result for easy plot.
    
    The parameters need to be set at the time of creation, such as number of total events, splits, class description,
    feature array. However, the prediction, event.rst-based scoring can be added and modified at run-time - in case
    of failure at run-time.

    Parameters:
        name (:obj:`str`): Name of the learning run.
        description (:obj:`str`): Description of the learning result.
        classes (:obj:`list` of :obj:`str`): List of description of target classes.
        num_events (:obj:`int`): Number of total entries in the test set.
        bg_class (:obj:`str`): Name of the class that is considered background.
        splits (:obj:`OrderedDict`): List of splits with name of splits as key and the size of each split as value.

    Attributes:
        name (:obj:`str`): Name of the learning run
        data (:obj:`str`): Path to the h5py dataset directory
        mode (:obj:`str`): valid choices are `single_shot`, `by_week` or `by_day`
        created_time (:obj:`float`): created time since Epoch in seconds
        modified_time (:obj:`float`): record modified time since Epoch in seconds
        overall_performance (:class:`numpy.array`): overall performance of the learning
        per_class_performance (:class:`numpy.array`): overall per-class performance of the learning
        confusion_matrix (:class:`numpy.array`): overall confusion matrix
        records (:obj:`collections.OrderedDict`): Ordered dictionary storing all records
    """
    def __init__(self, name, classes, num_events, bg_class=None, splits=None, description=''):
        cur_time = datetime.now()
        self.name = name
        self.description = description
        self.classes = classes
        self.created_time = cur_time
        self.modified_time = cur_time
        self.performance = {}
        self.splits = OrderedDict()
        if splits is not None:
            index = 0
            for name, length in splits:
                self.splits[name] = {
                    'start': index,
                    'stop': index+length,
                    'model_path': ''
                }
                index += length
        else:
            self.splits['None'] = {
                'start': 0,
                'stop': num_events,
                'model_path': ''
            }
        self.truth = np.empty(shape=(num_events, ), dtype=np.int)
        self.prediction = np.empty(shape=(num_events, ), dtype=np.int)
        self.time = np.empty(shape=(num_events, ), dtype='datetime64[ns]')
        self.num_events = num_events
        if bg_class is None:
            self.bg_class_id = -1
        elif bg_class in self.classes:
            self.bg_class_id = self.classes.index(bg_class)
        else:
            raise ValueError('Background class %s not in the target classes list.' % bg_class)

    def record_result(self, model_file, time, truth, prediction, split=None):
        """Record the result of a split

        Args:
            model_file (:obj:`str`): Path to the file that stores the model parameters
            split (:obj:`str`): Name of the split the record is for
            time (:obj:`list` of :obj:`datetime`): Corresponding datetime
            truth (:obj:`numpy.ndarray`): Array that holds the ground truth for the targeting split
            prediction (:obj:`numpy.ndarray`): Array that holds the prediction for the targeting split
        """
        split_name = str(split)
        if split_name not in self.splits.keys():
            return ValueError('Split %s not found in the result.' % split)
        start_pos = self.splits[split_name]['start']
        stop_pos = self.splits[split_name]['stop']
        self.truth[start_pos:stop_pos] = truth.astype(dtype=np.int)
        self.prediction[start_pos:stop_pos] = prediction.astype(dtype=np.int)
        self.time[start_pos:stop_pos] = time
        self.splits[split_name]['model_path'] = model_file
        # Calculate performance metrics for the split
        confusion_matrix = get_confusion_matrix(len(self.classes),
                                                self.truth[start_pos:stop_pos],
                                                self.prediction[start_pos:stop_pos]
                                                )
        self.splits[split_name]['confusion_matrix'] = confusion_matrix
        # After confusion metrix, one can calculate traditional multi-class performance
        overall_performance, per_class_performance = get_performance_array(confusion_matrix)
        self.splits[split_name]['overall_performance'] = overall_performance
        self.splits[split_name]['per_class_performance'] = per_class_performance
        # Note: Event-based scoring can be done after all split are logged in.

    def get_record_of_split(self, split):
        """Get result corresponding to specific split
        
        Args:
            split (:obj:`str`): Name of the split.
            
        Returns:
            :obj:`dict`: 
        """
        if split in self.splits.keys():
            return self.splits[split]
        else:
            logger.error('Cannot find split %s.' % split)
            return None

    def get_time_list(self):
        time_list = [datetime.utcfromtimestamp(item.astype(datetime) * 1e-9) for item in self.time]
        return time_list

    def event_based_scoring(self):
        """Event based segment scoring
        """
        self.performance['event_scoring'] = score_segment(self.truth, self.prediction, bg_label=self.bg_class_id)

    def calculate_overall_performance(self):
        """Calculate overall performance
        """
        confusion_matrix = get_confusion_matrix(len(self.classes), self.truth, self.prediction)
        overall_performance, per_class_performance = get_performance_array(confusion_matrix)
        self.performance['overall_performance'] = overall_performance
        self.performance['per_class_performance'] = per_class_performance

    def save_to_file(self, filename):
        """Pickle to file
        """
        f = open(filename, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load_from_file(filename):
        """Load LearningResult from file
        
        Args:
            filename (:obj:`str`): Path to the file that stores the result.
        
        Returns:
            :class:`pyActLearn.performance.record.LearningResult`: LearningResult object.
        """
        f = open(filename, 'rb')
        result = pickle.load(f)
        f.close()
        return result

    def export_to_xlsx(self, filename, home_info=None):
        """Export to XLSX

        Args:
            filename (:obj:`str`): path to the file
            home_info (:class:`pyActLearn.CASAS.fuel.CASASFuel`): dataset information
        """
        workbook = xlsxwriter.Workbook(filename)
        num_performance = len(per_class_performance_index)
        num_classes = len(self.classes)
        # Overall Performance Summary
        overall_sheet = workbook.add_worksheet('overall')
        overall_sheet.merge_range(0, 0, 0, len(overall_performance_index) - 1, 'Overall Performance')
        for c in range(len(overall_performance_index)):
            overall_sheet.write(1, c, str(overall_performance_index[c]))
            overall_sheet.write(2, c, self.overall_performance[c])
        overall_sheet.merge_range(4, 0, 4, len(per_class_performance_index), 'Per-Class Performance')
        overall_sheet.write(5, 0, 'Activities')
        for c in range(len(per_class_performance_index)):
            overall_sheet.write(5, c + 1, str(per_class_performance_index[c]))
        for r in range(num_classes):
            label = home_info.get_activity_by_index(r)
            overall_sheet.write(r + 6, 0, label)
            for c in range(num_performance):
                overall_sheet.write(r + 6, c + 1, self.per_class_performance[r][c])
        overall_sheet.merge_range(8 + num_classes, 0, 8 + num_classes, num_classes, 'Confusion Matrix')
        for i in range(num_classes):
            label = home_info.get_activity_by_index(i)
            overall_sheet.write(9 + num_classes, i + 1, label)
            overall_sheet.write(10 + num_classes + i, 0, label)
        for r in range(num_classes):
            for c in range(num_classes):
                overall_sheet.write(10 + num_classes + r, c + 1, self.confusion_matrix[r][c])

        records = self.get_record_keys()

        # Weekly Performance Summary
        weekly_sheet = workbook.add_worksheet('weekly')
        weekly_list_title = ['dataset', '#week'] + overall_performance_index
        for c in range(len(weekly_list_title)):
            weekly_sheet.write(0, c, str(weekly_list_title[c]))
        r = 1
        for record_id in records:
            weekly_sheet.write(r, 0, 'b1')
            weekly_sheet.write(r, 1, record_id)
            for c in range(len(overall_performance_index)):
                weekly_sheet.write(r, c + 2, '%.5f' % self.get_record_by_key(record_id)['overall_performance'][c])
            r += 1
        dataset_list_title = ['activities'] + per_class_performance_index
        # Per Week Per Class Summary
        for record_id in self.get_record_keys():
            cur_sheet = workbook.add_worksheet(record_id)
            for c in range(0, len(dataset_list_title)):
                cur_sheet.write(0, c, str(dataset_list_title[c]))
            for r in range(num_classes):
                label = home_info.get_activity_by_index(r)
                cur_sheet.write(r+1, 0, label)
                for c in range(num_performance):
                    cur_sheet.write(r + 1, c + 1, self.get_record_by_key(record_id)['per_class_performance'][r][c])
        workbook.close()

    def export_annotation(self, filename):
        """Export back annotation to file
        """
        f = open(filename, 'w')
        for i in range(self.num_events):
            f.write('%s %s\n' % (
                datetime.utcfromtimestamp(self.time[i].astype(datetime) * 1e-9).strftime('%Y-%m-%d %H:%M:%S'),
                self.classes[self.prediction[i]]
            ))
        f.close()
