import time
import logging
import xlsxwriter
import collections
from . import overall_performance_index, per_class_performance_index, get_performance_array
from ..logging import logging_name
from ..CASAS.fuel import CASASFuel

logger = logging.getLogger(__name__)


class LearningResult:
    """LearningResult is a class that stores results of a learning run.
    It may be a single-shot run or a time-based analysis
    The result structure holds the parameters for the model as well as
    the evaluation result for easy plot.

    Parameters:
        name (:obj:`str`): Name of the learning run
        data (:obj:`str`): Name of the dataset or description of the dataset
        mode (:obj:`str`): valid choices are `single_shot`, `by_week` or `by_day`

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
    def __init__(self, name='', data='', mode='single_shot'):
        cur_time = time.time()
        self.name = name
        self.data = data
        self.mode = mode
        self.created_time = cur_time
        self.modified_time = cur_time
        self.overall_performance = None
        self.per_class_performance = None
        self.confusion_matrix = None
        self.records = collections.OrderedDict()

    def get_num_records(self):
        """Get the length of result records in current instance
        """
        if self.records is None:
            return 0
        else:
            return len(self.records)

    def get_record_keys(self):
        """Get List of keys to all the records
        """
        if self.records is None:
            return []
        else:
            return self.records.keys()

    def add_record(self, model, key='single_shot', confusion_matrix=None):
        """Add a learning milestone record

        Args:
            model (:obj:`object`): snap shot of learning model parameters
            key (:obj:`str`): key string to represent current record
            confusion_matrix (:obj:`numpy.array`): Confusion Matrix
        """
        if self.get_num_records() == 0:
            self.confusion_matrix = confusion_matrix.copy()
        else:
            # Check confusion matrix size
            if confusion_matrix.shape != self.confusion_matrix.shape:
                logger.error(logging_name(self) + ': confusion matrix shape mismatch. Original shape %s. New shape %s'
                             % (str(self.confusion_matrix.shape), str(confusion_matrix.shape)))
            else:
                self.confusion_matrix += confusion_matrix
        self.overall_performance, self.per_class_performance = get_performance_array(self.confusion_matrix)
        overall_performance, per_class_performance = get_performance_array(confusion_matrix)
        cur_result = {
            'model': model,
            'confusion_matrix': confusion_matrix,
            'per_class_performance': per_class_performance,
            'overall_performance': overall_performance
        }
        self.records[key] = cur_result

    def get_record_by_key(self, key):
        """
        Get result corresponding to specific key
        :param key:
        :return:
        """
        if key in self.records.keys():
            return self.records[key]
        else:
            logger.error(logging_name(self) + ': Cannot find record %s' % key)
            return None

    def export_to_xlsx(self, filename, home_info=None):
        """Export to XLSX

        Args:
            filename (:obj:`str`): path to the file
            home_info (:class:`pyActLearn.CASAS.fuel.CASASFuel`): dataset information
        """
        if home_info is None:
            home_info = CASASFuel(dir_name=self.data)
        workbook = xlsxwriter.Workbook(filename)
        records = self.get_record_keys()
        num_performance = len(per_class_performance_index)
        num_classes = self.confusion_matrix.shape[0]
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
