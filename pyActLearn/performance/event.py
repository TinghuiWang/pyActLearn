""" Event-based Performance Metrics

This file implements event-based performance metrics for activity recognition.

Reference:

    - Minnen, David, Tracy Westeyn, Thad Starner, J. Ward, and Paul Lukowicz. Performance metrics and evaluation
      issues for continuous activity recognition. Performance Metrics for Intelligent Systems 4 (2006).
    - Ward, J. A., Lukowicz, P. & Gellersen, H. W. Performance metrics for activity recognition. ACM Trans. Intell. 
      Syst. Technol. 2, 6:16:23 (2011).
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def per_class_event_scoring(num_classes, truth, prediction, truth_scoring, prediction_scoring):
    """Create per-class event scoring to identify the contribution of event-based errors to the traditional recall
    and false-positive rate.
    
    Instead of doing an EAD as proposed in previous two papers, we look at **Recall** and **FPR** separately.
    
    **Recall** is defined as TP/(TP + FN). In another word, how often does it predict yes when it's actually yes?
    The errors in the false negatives, such as Deletion, Fragmenting, and Underfill, adds up to the FP. A Deletion
    means a total miss of an activity. Underfill represents an error on the begin and end boundary of the event.
    Fragmenting represents a glitch in the prediction.
    
    **Precision** is defined as TP/(TP + FP). In another word, how often is it a yes when it is predicted yes?
    The error in the false positives, such as Insertion, Merge and Overfill, adds up to the
    FP. In the task of ADL recognition, insertion may be caused by human error in labeling. Overfill represents a
    disagreement of the begin/end boundary of an activity, but the merge is a glitch in the prediction.
    
    The function goes through the scoring of prediction and ground truth - and returns two dictionary that summaries
    the contribution of all those errors to **Recall** and **False Positive Rate** scores.
    
    Args:
        num_classes (:obj:`int`): Total number of target classes
        truth (:obj:`numpy.ndarray`): Ground truth array, shape (num_samples, )
        prediction (:obj:`numpy.ndarray`): Prediction array, shape (num_samples, )
        truth_scoring (:obj:`numpy.ndarray`): Event scoring with respect to ground truth labels (i.e. false negatives
            are further divided into Deletion, Fragmenting, and Underfill). The information in this array is used to 
            fill **Recall** measurement.
        prediction_scoring (:obj:`numpy.ndarray`): Event scoring with respect to prediction labels (i.e. false positives
            are further divided into Insertion, Merging and Overfill). The information in this array is used to fill
            **Precision** measurement.

    Returns:
        :obj:`tuple` of :obj:`numpy.ndarray`: 
            Tuple of event-based scoring summarie for recall and precision.
            Each summary array has a shape of (num_classes, ).
    """
    recall_array = np.zeros((num_classes,),
                           dtype=np.dtype([
                               ('C', np.int, 1),
                               ('D', np.int, 1),
                               ('F', np.int, 1),
                               ('U', np.int, 1),
                               ('u', np.int, 1)])
                           )
    fpr_array = np.zeros((num_classes,),
                        dtype=np.dtype([
                            ('C', np.int, 1),
                            ('I', np.int, 1),
                            ('M', np.int, 1),
                            ('O', np.int, 1),
                            ('o', np.int, 1)])
                        )
    for i in range(truth_scoring.shape[0]):
        recall_array[truth[i]][truth_scoring[i]] += 1
        fpr_array[prediction[i]][prediction_scoring[i]] += 1
    return recall_array, fpr_array


def per_class_segment_scoring(num_classes, truth, prediction, truth_scoring, prediction_scoring):
    """Create per-class event scoring to identify the contribution of event-based errors to the traditional recall
    and false-positive rate. The count is based on each event segment instead of each sensor event.

    Args:
        num_classes (:obj:`int`): Total number of target classes
        truth (:obj:`numpy.ndarray`): Ground truth array, shape (num_samples, )
        prediction (:obj:`numpy.ndarray`): Prediction array, shape (num_samples, )
        truth_scoring (:obj:`numpy.ndarray`): Event scoring with respect to ground truth labels (i.e. false negatives
            are further divided into Deletion, Fragmenting, and Underfill). The information in this array is used to 
            fill **Recall** measurement.
        prediction_scoring (:obj:`numpy.ndarray`): Event scoring with respect to prediction labels (i.e. false positives
            are further divided into Insertion, Merging and Overfill). The information in this array is used to fill
            **Precision** measurement.

    Returns:
        :obj:`tuple` of :obj:`numpy.ndarray`: 
            Tuple of event-based scoring summarie for recall and precision.
            Each summary array has a shape of (num_classes, ).
    """
    # Total Segments
    total_segs = 0
    seg_logs = np.zeros((num_classes,))

    recall_array = np.zeros((num_classes,),
                            dtype=np.dtype([
                                ('C', np.int, 1),
                                ('D', np.int, 1),
                                ('F', np.int, 1),
                                ('U', np.int, 1),
                                ('u', np.int, 1)])
                            )
    fpr_array = np.zeros((num_classes,),
                         dtype=np.dtype([
                             ('C', np.int, 1),
                             ('I', np.int, 1),
                             ('M', np.int, 1),
                             ('O', np.int, 1),
                             ('o', np.int, 1)])
                         )
    prev_prediction = prediction[0]
    prev_prediction_scoring = prediction_scoring[0]
    prev_truth = truth[0]
    prev_truth_scoring = truth_scoring[0]
    seg_correct = 0
    seg_delete = 0
    for i in range(truth_scoring.shape[0]):
        cur_prediction = prediction[i]
        cur_prediction_scoring = prediction_scoring[i]
        cur_truth = truth[i]
        cur_truth_scoring = truth_scoring[i]
        # Update Counts
        if cur_truth_scoring != prev_truth_scoring or cur_truth != prev_truth:
            if prev_truth_scoring == 'C':
                seg_correct = 1
            elif prev_truth_scoring == 'D':
                seg_delete = 1
            else:
                recall_array[prev_truth][prev_truth_scoring] += 1
        # Add counts to array
        if cur_prediction != prev_prediction or cur_prediction_scoring != prev_prediction_scoring:
            fpr_array[prev_prediction][prev_prediction_scoring] += 1
        # Update array counts
        if cur_truth != prev_truth:
            # DEBUG
            total_segs += 1
            seg_logs[prev_truth] += 1
            if (seg_correct == 0 and seg_delete == 0) or (seg_correct == 1 and seg_delete == 1):
                if prev_truth != 7:
                    logger.debug('i: %d' % i)
                    logger.debug('truth      : %s' % str(truth[i-10:i+10]))
                    logger.debug('predi      : %s' % str(prediction[i-10:i+10]))
                    logger.debug('truth_score: %s' % str(truth_scoring[i - 10:i + 10]))
            # END_DEBUG
            recall_array[prev_truth]['C'] += seg_correct
            recall_array[prev_truth]['D'] += seg_delete
            seg_correct = 0
            seg_delete = 0
        prev_prediction = cur_prediction
        prev_prediction_scoring = cur_prediction_scoring
        prev_truth = cur_truth
        prev_truth_scoring = cur_truth_scoring
    # Final Update
    recall_array[prev_truth]['C'] += seg_correct
    fpr_array[prev_prediction][prev_prediction_scoring] += 1
    # Clear Underfill, Overfill, Segment and Merge
    for i in range(num_classes):
        recall_array[i]['U'] = 0
        recall_array[i]['u'] = 0
        recall_array[i]['F'] = 0
        fpr_array[i]['O'] = 0
        fpr_array[i]['o'] = 0
        fpr_array[i]['M'] = 0
    #DEBUG
    logger.debug('Total Seg: %d' % total_segs)
    logger.debug('seg_logs: %s' % str(seg_logs))
    logger.debug('seg_logs_added: %s' % str([
        recall_array[i]['C'] + recall_array[i]['D'] for i in range(num_classes)
    ]))
    return recall_array, fpr_array


def score_segment(truth, prediction, bg_label=-1):
    r""" Score Segments
    
    According to [Minnen2006]_ and [Ward2011]_, a segment is defined as the largest part of an event on which
    the comparison between the ground truth and the output of recognition system can be made in an unambiguous
    way. However, in this piece of code, we remove the limit where the segment is the largest part of an event.
    As long as there is a match between prediction and ground truth, it is recognized as a segment.
    
    There are four possible outcomes to be scored: TP, TN, FP and FN. In event-based performance scoring, the FP and
    FN are further divided to the following cases:
    
    - Insertion (I): A FP that corresponds exactly to an inserted return.
    - Merge (M): A FP that occurs between two TP segments within a merge return.
    - Overfill (O): A FP that occurs at the start or end of a partially matched return.
    - Deletion (D): A FN that corresponds exactly to a deleted evjmk, ent.
    - Fragmenting (F): A FN that corresponds exactly to a deleted event.
    - Underfill (U): A FN that occurs at the start or end of a detected event.
    
    Args:
        truth (:obj:`numpy.ndarray`): Ground truth
        prediction (:obj:`numpy.ndarray`): prediction
        bg_label (:obj:`numpy.ndarray`): Background label
    
    Returns:
        :obj:`numpy.ndarray`: An array with truth and event-based scoring labels
    """
    # Sanity Check
    assert(truth.shape == prediction.shape)
    # Prepare Scoring
    truth_score = np.empty(
        (truth.shape[0],),
        dtype=np.unicode
    )
    prediction_score = np.empty(
        (truth.shape[0],),
        dtype=np.unicode
    )
    # Find next segmentation
    seg_start = 0
    seg_stop = 0
    while seg_stop < truth.size:
        seg_stop = _next_segment(truth, seg_start)
        # Score the segment
        # 1. Find if there is correct labels there
        _score_specified_segment(truth, prediction, seg_start, seg_stop, truth_score, prediction_score, bg_label)
        seg_start = seg_stop
    return truth_score, prediction_score


def _next_segment(truth, start_index):
    """ Find the end of the segment
    
    Args:
        truth (:obj:`numpy.ndarray`): Ground truth
        start_index (:obj:`int`): start index of current segment
        
    Returns:
        :obj:`int`: end index of current segment
    """
    stop_index = start_index + 1
    while stop_index < truth.size:
        if truth[stop_index] == truth[start_index]:
            stop_index += 1
        else:
            return stop_index
    return stop_index


def _score_specified_segment(truth, prediction, start, stop, truth_score, prediction_score, bg_label=-1):
    """Score a given segment
    """
    # Find if the activity of this segment is correctly picked up by prediction
    seg_label = truth[start]
    # Label correct items
    num_correct_items = 0
    correct_seg_list = []
    correct_seg_start = start
    correct_seg_stop = start
    for i in range(start, stop):
        truth_score[i] = seg_label
        prediction_score[i] = prediction[i]
        if prediction[i] == seg_label:
            truth_score[i] = 'C'.encode('utf-8')
            prediction_score[i] = 'C'.encode('utf-8')
            num_correct_items += 1
            correct_seg_stop = i+1
        else:
            # For the truth class, it is a false negative - default to Deletion (D)
            truth_score[i] = 'D'.encode('utf-8')
            # For the prediction class, it is a false positive - default to Insertion (I)
            prediction_score[i] = 'I'.encode('utf-8')
            # Populate correct segment list
            if correct_seg_stop > correct_seg_start:
                correct_seg_list.append((correct_seg_start, correct_seg_stop))
            correct_seg_start = i + 1
    if truth_score[stop - 1] == 'C':
        # Add correct segment
        correct_seg_list.append((correct_seg_start, correct_seg_stop))
    # If the prediction got the segment completely wrong, scoring finished.
    if num_correct_items == 0 or seg_label == bg_label:
        return
    # Otherwise, go through the second time and identify the cause of the error
    # Overfill (O) is part of false positive (prediction label)
    if prediction[start] == seg_label:  # Check Overfill at the beginning
        i = start - 1
        while i >= 0:
            if prediction[i] == seg_label:
                prediction_score[i] = 'O'.encode('utf-8')
                i -= 1
            else:
                break
    if prediction[stop - 1] == seg_label:  # Check Overfill at the end
        i = stop
        while i < truth.size:
            if prediction[i] == seg_label:
                prediction_score[i] = 'o'.encode('utf-8')
                i += 1
            else:
                break
    # Underfill (U) is part of false negative (related to truth)
    if prediction[start] != seg_label:
        i = start
        while i < stop:
            if prediction[i] == seg_label:
                break
            else:
                truth_score[i] = 'U'.encode('utf-8')
                i += 1
    if prediction[stop - 1] != seg_label:
        i = stop - 1
        while i >= start:
            if prediction[i] == seg_label:
                break
            else:
                truth_score[i] = 'u'.encode('utf-8')
                i -= 1
    # Merge and Fragment occur between two TP segments
    # Handle Fragment and Merge
    if len(correct_seg_list) > 1:
        for i in range(len(correct_seg_list) - 1):
            tmp_start = correct_seg_list[i][1]
            tmp_stop = correct_seg_list[i+1][0]
            for j in range(tmp_start, tmp_stop):
                # For the truth class, it is a false negative - so it is Fragment (F)
                truth_score[j] = 'F'.encode('utf-8')
                # For the prediction class, it is a false positive - so it is Merge (M)
                prediction_score[j] = 'M'.encode('utf-8')
