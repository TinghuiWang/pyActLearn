"""Reference Document:
    Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks.
    Information Processing and Management, 45, p. 427-437
"""

import logging
import numpy as np

logger = logging.getLogger(__file__)

per_class_performance_index = ['true_positive', 'true_negative', 'false_positive', 'false_negative',
                               'accuracy', 'misclassification', 'recall', 'false positive rate',
                               'specificity', 'precision', 'prevalence', 'f-1 measure', 'g-measure']

overall_performance_index = ['average accuracy', 'weighed accuracy',
                             'precision (micro)', 'recall (micro)', 'f-1 score (micro)',
                             'precision (macro)', 'recall (macro)', 'f-1 score (macro)',
                             'exact matching ratio']


def get_confusion_matrix_by_activity(num_classes, label, predicted):
    """Calculate confusion matrix based on activity accuracy

    Instead of calculating confusion matrix by comparing ground truth and predicted
    result one by one, it compares if a segment of activity is correctly predicted.
    It also logs the shift of activity predicted versus labeled.
    """
    return


def get_confusion_matrix(num_classes, label, predicted):
    """Calculate confusion matrix based on ground truth and predicted result

    Args:
        num_classes (:obj:`int`): Number of classes
        label (:obj:`list` of :obj:`int`): ground truth labels
        predicted (:obj:`list` of :obj:`int`): predicted labels

    Returns:
        :class:`numpy.array`: Confusion matrix (`numpy_class` by `numpy_class`)
    """
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(label)):
        matrix[label[i]][predicted[i]] += 1
    return matrix


def get_performance_array(confusion_matrix):
    r"""Calculate performance matrix based on the given confusion matrix
    
    [Sokolova2009]_ provides a detailed analysis for multi-class performance metrics.
    
    Per-class performance metrics:
    
    0. **True_Positive**: number of samples that belong to class and classified correctly
    1. **True_Negative**: number of samples that correctly classified as not belonging to class
    2. **False_Positive**: number of samples that belong to class and not classified correctMeasure:
    3. **False_Negative**: number of samples that do not belong to class but classified as class
    4. **Accuracy**: Overall, how often is the classifier correct? (TP + TN) / (TP + TN + FP + FN)
    5. **Misclassification**: Overall, how often is it wrong? (FP + FN) / (TP + TN + FP + FN)
    6. **Recall**: When it's actually yes, how often does it predict yes? TP / (TP + FN)
    7. **False Positive Rate**: When it's actually no, how often does it predict yes? FP / (FP + TN)
    8. **Specificity**: When it's actually no, how often does it predict no? TN / (FP + TN)
    9. **Precision**: When it predicts yes, how often is it correct? TP / (TP + FP)
    10. **Prevalence**: How often does the yes condition actually occur in our sample? Total(class) / Total(samples)
    11. **F(1) Measure**: 2 * (precision * recall) / (precision + recall)
    12. **G Measure**:  sqrt(precision * recall)

    Gets Overall Performance for the classifier
    
    0. **Average Accuracy**: The average per-class effectiveness of a classifier
    1. **Weighed Accuracy**: The average effectiveness of a classifier weighed by prevalence of each class
    2. **Precision (micro)**: Agreement of the class labels with those of a classifiers if calculated from sums of per-text
       decision
    3. **Recall (micro)**: Effectiveness of a classifier to identify class labels if calculated from sums of per-text
       decisions
    4. **F-Score (micro)**: Relationship between data's positive labels and those given by a classifier based on a sums of
       per-text decisions
    5. **Precision (macro)**: An average per-class agreement of the data class labels with those of a classifiers
    6. **Recall (macro)**: An average per-class effectiveness of a classifier to identify class labels
    7. **F-Score (micro)**: Relations between data's positive labels and those given by a classifier based on a per-class
       average
    8. **Exact Matching Ratio**: The average per-text exact classification

    .. note:: 
       
       In Multi-class classification, Micro-Precision == Micro-Recall == Micro-FScore == Exact Matching Ratio
       (Multi-class classification: each input is to be classified into one and only one class)

    Args:
        num_classes (:obj:`int`): Number of classes
        confusion_matrix (:class:`numpy.array`): Confusion Matrix (numpy array of num_class by num_class)

    Returns:
        :obj:`tuple` of :class:`numpy.array`: tuple of overall performance and per class performance
    """
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        logger.error("confusion matrix with shape " + str(confusion_matrix.shape) + " is not square.")
        return None, None

    num_classes = confusion_matrix.shape[0]

    per_class = np.zeros((num_classes, len(per_class_performance_index)), dtype=float)
    overall = np.zeros((len(overall_performance_index),), dtype=float)

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        true_negative = np.sum(confusion_matrix)\
            - np.sum(confusion_matrix[i, :])\
            - np.sum(confusion_matrix[:, i])\
            + confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - confusion_matrix[i][i]
        false_negative = np.sum(confusion_matrix[i, :]) - confusion_matrix[i][i]
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        per_class_accuracy = (true_positive + true_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Mis-classification: (FP + FN) / (TP + TN + FP + FN)
        per_class_misclassification = (false_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # Recall: TP / (TP + FN)
        if true_positive + false_negative == 0:
            per_class_recall = 0.
        else:
            per_class_recall = true_positive / (true_positive + false_negative)
        # False Positive Rate: FP / (FP + TN)
        if false_positive + true_negative == 0:
            per_class_fpr = 0.
        else:
            per_class_fpr = false_positive / (false_positive + true_negative)
        # Specificity: TN / (FP + TN)
        if false_positive + true_negative == 0:
            per_class_specificity = 0.
        else:
            per_class_specificity = true_negative / (false_positive + true_negative)
        # Precision: TP / (TP + FP)
        if true_positive + false_positive == 0:
            per_class_precision = 0.
        else:
            per_class_precision = true_positive / (true_positive + false_positive)
        # prevalence
        per_class_prevalence = (true_positive + false_negative)\
            / (true_positive + true_negative + false_positive + false_negative)
        # F-1 Measure: 2 * (precision * recall) / (precision +
        if per_class_precision + per_class_recall == 0:
            per_class_fscore = 0.
        else:
            per_class_fscore = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall)
        # G Measure: sqrt(precision * recall)
        per_class_gscore = np.sqrt(per_class_precision * per_class_recall)
        per_class[i][0] = true_positive
        per_class[i][1] = true_negative
        per_class[i][2] = false_positive
        per_class[i][3] = false_negative
        per_class[i][4] = per_class_accuracy
        per_class[i][5] = per_class_misclassification
        per_class[i][6] = per_class_recall
        per_class[i][7] = per_class_fpr
        per_class[i][8] = per_class_specificity
        per_class[i][9] = per_class_precision
        per_class[i][10] = per_class_prevalence
        per_class[i][11] = per_class_fscore
        per_class[i][12] = per_class_gscore

    # Average Accuracy: Sum{i}{Accuracy{i}} / num_class
    overall[0] = np.sum(per_class[:, per_class_performance_index.index('accuracy')]) / num_classes
    # Weighed Accuracy: Sum{i}{Accuracy{i} * Prevalence{i}} / num_class
    overall[1] = np.dot(per_class[:, per_class_performance_index.index('accuracy')],
                        per_class[:, per_class_performance_index.index('prevalence')])
    # Precision (micro): Sum{i}{TP_i} / Sum{i}{TP_i + FP_i}
    overall[2] = np.sum(per_class[:, per_class_performance_index.index('true_positive')]) / \
                 np.sum(per_class[:, per_class_performance_index.index('true_positive')] +
                        per_class[:, per_class_performance_index.index('false_positive')])
    # Recall (micro): Sum{i}{TP_i} / Sum{i}{TP_i + FN_i}
    overall[3] = np.sum(per_class[:, per_class_performance_index.index('true_positive')]) / \
                 np.sum(per_class[:, per_class_performance_index.index('true_positive')] +
                        per_class[:, per_class_performance_index.index('false_negative')])
    # F_Score (micro): 2 * Precision_micro * Recall_micro / (Precision_micro + Recall_micro)
    overall[4] = 2 * overall[2] * overall[3] / (overall[2] + overall[3])
    # Precision (macro): Sum{i}{Precision_i} / num_class
    overall[5] = np.sum(per_class[:, per_class_performance_index.index('precision')]) / num_classes
    # Recall (macro): Sum{i}{Recall_i} / num_class
    overall[6] = np.sum(per_class[:, per_class_performance_index.index('recall')]) / num_classes
    # F_Score (macro): 2 * Precision_macro * Recall_macro / (Precision_macro + Recall_macro)
    overall[7] = 2 * overall[5] * overall[6] / (overall[5] + overall[6])
    # Exact Matching Ratio:
    overall[8] = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return overall, per_class

