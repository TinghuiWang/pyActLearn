import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

recall_scoring_labels = ['Correct', 'Fragmenting', 'Underfill-B', 'Underfill-E', 'Deletion']
fpr_scoring_labels = ['Correct', 'Merging', 'Overfill-B', 'Overfill-E', 'Insertion']
recall_scoring_indices = {'C': 0, 'D': 4, 'F': 1, 'U': 2, 'u': 3}
fpr_scoring_indices = {'C': 0, 'I': 4, 'M': 1, 'O': 2, 'o': 3}


def draw_per_class_recall(classes, class_colors, recall_array, filename=None):
    """Draw recall array
    """
    recall_np = np.empty((len(classes), len(recall_scoring_labels)),
                         dtype=np.float)
    for i, row in enumerate(recall_array):
        for key in recall_scoring_indices:
            recall_np[i, recall_scoring_indices[key]] = row[key]
    recall_np /= np.sum(recall_np, axis=1, keepdims=True)

    ind = np.arange(len(classes))
    width = 0.35
    bottom = np.zeros((len(classes),))
    bar_array = []
    for i in range(len(recall_scoring_labels)):
        bar_array.append(plt.bar(ind, recall_np[:, i], width,
                                 alpha=(1-1/len(recall_scoring_labels) * i),
                                 color=class_colors, bottom=bottom)[0])
        bottom += recall_np[:, i]
    plt.ylabel('Percentage')
    plt.xlabel('Classes')
    plt.xticks(ind, classes, rotation='vertical')
    plt.legend(bar_array, recall_scoring_labels)
    plt.show()


def _get_bg_class_id(classes, background_class):
    # Verify Background Class first
    if background_class is not None:
        bg_class_id = classes.index(background_class)
    else:
        bg_class_id = -1
    return bg_class_id


def _get_metric_label_dict(metric_name='recall'):
    if metric_name == 'recall':
        metric_labels = recall_scoring_labels
        metric_indices = recall_scoring_indices
    else:
        metric_labels = fpr_scoring_labels
        metric_indices = fpr_scoring_indices
    return metric_labels, metric_indices


def _gether_per_class_metrics(methods, classes, metric_arrays, as_percent, metric_labels, metric_indices):
    """Prepare metrics for bar plot
    """
    # Gather data for bar plot
    plot_metric_arrays = []
    for j in range(len(methods)):
        cur_metric = np.empty((len(classes), len(metric_labels)),
                              dtype=np.float)
        for i, row in enumerate(metric_arrays[j]):
            for key in metric_indices:
                cur_metric[i, metric_indices[key]] = row[key]
        # As percent
        if as_percent:
            cur_metric /= np.sum(cur_metric, axis=1, keepdims=True)
        # Append the metric for current methods
        plot_metric_arrays.append(cur_metric)
    return plot_metric_arrays


def _compare_per_class_metrics(methods, classes, class_colors, metric_arrays,
                               group_by='methods', filename=None, background_class=None,
                               as_percent=True, metric_name='recall'):
    """Compare per-class metrics between methods using bar-graph
    """
    metric_labels, metric_indices = _get_metric_label_dict(metric_name=metric_name)
    bg_class_id = _get_bg_class_id(classes, background_class)
    plot_metric_arrays = _gether_per_class_metrics(methods, classes, metric_arrays, as_percent,
                                                   metric_labels, metric_indices)
    # Prepare Data and x-label
    xtick_labels = []
    bar_colors = []
    if bg_class_id < 0:
        plot_data = np.empty((len(methods) * len(classes), len(metric_labels)))
    else:
        plot_data = np.empty((len(methods) * (len(classes) - 1), len(metric_labels)))
    # Fill plot data with values
    if group_by == 'methods':
        num_base_axis = len(methods)
        if bg_class_id < 0:
            num_sec_axis = len(classes)
        else:
            num_sec_axis = len(classes) - 1
        for j in range(len(classes)):
            if bg_class_id < 0 or j < bg_class_id:
                for i in range(len(methods)):
                    bar_colors.append(class_colors[j])
                    xtick_labels.append(methods[i])
                    plot_data[j * num_base_axis + i, :] = plot_metric_arrays[i][j, :]
            elif j > bg_class_id:
                for i in range(len(methods)):
                    bar_colors.append(class_colors[j])
                    xtick_labels.append(methods[i])
                    plot_data[(j-1) * num_base_axis + i, :] = plot_metric_arrays[i][j, :]
    else:
        if bg_class_id < 0:
            num_base_axis = len(classes)
        else:
            num_base_axis = len(classes) - 1
        num_sec_axis = len(methods)
        for j in range(len(methods)):
            xtick_labels.append(methods[j])
            for i in range(len(classes)):
                if bg_class_id < 0 or i < bg_class_id:
                    bar_colors.append(class_colors[i])
                    plot_data[j * num_base_axis + i, :] = plot_metric_arrays[j][i, :]
                elif i > bg_class_id:
                    bar_colors.append(class_colors[i])
                    plot_data[j * num_base_axis + i - 1, :] = plot_metric_arrays[j][i, :]
    # Calculate width and bar location
    width = 1/(num_base_axis + 1)
    ind = []
    for i in range(num_sec_axis):
        for j in range(num_base_axis):
            ind.append(i + j * width + width)
    bottom = np.zeros((num_base_axis * num_sec_axis,))
    # Set major and minor lines for y_axis
    if as_percent:
        minor_locator_value = 0.05
        major_locator_value = 0.2
    else:
        max_value = np.max(plot_data.sum(axis=1)) + 20
        minor_locator_value = int(max_value/20)
        major_locator_value = int(max_value/5)
    # Set up x_label location
    xlabel_ind = []
    if group_by == 'methods':
        xlabel_ind = [x + width/2 for x in ind]
        xlabel_rotation = 'vertical'
    else:
        xlabel_ind = [x + 0.5 for x in range(len(methods))]
        xlabel_rotation = 'horizontal'
    # Setup Figure
    fig, ax = plt.subplots()
    # Y-Axis
    minor_locator = MultipleLocator(minor_locator_value)
    major_locator = MultipleLocator(major_locator_value)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.grid(which="major", color='0.65', linestyle='-', linewidth=1)
    ax.yaxis.grid(which="minor", color='0.45', linestyle=' ', linewidth=1)
    # Plot Bar
    for i in range(len(metric_labels)):
        ax.bar(ind, plot_data[:, i], width,
               alpha=(1-1/len(metric_labels) * i),
               color=bar_colors, bottom=bottom)
        bottom += plot_data[:, i]
    if as_percent:
        plt.ylabel('Percentage')
    else:
        plt.ylabel('Count')
    plt.xlabel('Classes')
    plt.xticks(xlabel_ind, xtick_labels, rotation=xlabel_rotation, fontsize=6)
    # Prepare Legends
    patches = []
    legend_labels = []
    for i in range(len(metric_labels)):
        patches.append(Rectangle((0, 0), 0, 0, color='0.3', alpha=(1-1/len(metric_labels) * i)))
        legend_labels.append(metric_labels[i])
    for i in range(len(classes)):
        if i == bg_class_id:
            continue
        patches.append(Rectangle((0, 0), 0, 0, color=class_colors[i]))
        legend_labels.append(classes[i])
    plt.legend(patches, legend_labels, loc='center left', borderaxespad=0, bbox_to_anchor=(1.05, 0.5),
               prop={'size': 8})
    plt.tight_layout()
    plt.title('Event-based Activity Analysis - %s' % metric_name)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')


def compare_per_class_recall(methods, classes, class_colors, recall_arrays,
                             group_by='methods', filename=None, background_class=None,
                             as_percent=True):
    """Draw event.rst-based comparison between methods on Recall metric.
    
    Args:
        methods (:obj:`list` of :obj:`str`): List of names of different methods to be plotted.
        classes (:obj:`list` of :obj:`str`): List of target classes.
        class_colors (:obj:`list` of :obj:`str`): List of RGB color for corresponding classes in the ``classes`` list.
        recall_arrays (:obj:`list` of :obj:`numpy.ndarray`): List of recall arrays calculated for each methods.
        group_by (:obj:`str`): Group the bar graph of various 'methods' first or 'classes' first. Default to 'methods'.
        filename (:obj:`str`): The filename to save the plot. None if display on screen with pyplot.
        background_class (:obj:`str`): Background class. Usually it points to ``Other_Activity``. The statistics of
            background_class will be omitted from the plot.
        as_percent (:obj:`bool`): Whether or not to convert the accumulated value to percentage.
    """
    _compare_per_class_metrics(methods, classes, class_colors, recall_arrays,
                               group_by=group_by, filename=filename, background_class=background_class,
                               as_percent=as_percent, metric_name='recall')


def compare_per_class_precision(methods, classes, class_colors, precision_arrays,
                                group_by='methods', filename=None, background_class=None,
                                as_percent=True):
    """Draw event.rst-based comparison between methods on precision metric.

    Args:
        methods (:obj:`list` of :obj:`str`): List of names of different methods to be plotted.
        classes (:obj:`list` of :obj:`str`): List of target classes.
        class_colors (:obj:`list` of :obj:`str`): List of RGB color for corresponding classes in the ``classes`` list.
        recall_arrays (:obj:`list` of :obj:`numpy.ndarray`): List of recall arrays calculated for each methods.
        group_by (:obj:`str`): Group the bar graph of various 'methods' first or 'classes' first. Default to 'methods'.
        filename (:obj:`str`): The filename to save the plot. None if display on screen with pyplot.
        background_class (:obj:`str`): Background class. Usually it points to ``Other_Activity``. The statistics of
            background_class will be omitted from the plot.
        as_percent (:obj:`bool`): Whether or not to convert the accumulated value to percentage.
    """
    _compare_per_class_metrics(methods, classes, class_colors, precision_arrays,
                               group_by=group_by, filename=filename, background_class=background_class,
                               as_percent=as_percent, metric_name='precision')


def draw_timeliness_hist(classes, class_colors, truth, prediction, truth_scoring, prediction_scoring, time_list,
                         background_class):
    """Get Timeliness Histogram for underfill and overfill
    """
    start_mismatch, stop_mismatch = _get_timeliness_measures(classes, truth, prediction,
                                                             time_list)
    bg_id = _get_bg_class_id(classes, background_class)
    num_classes = len(classes)
    # Plot histogram
    stack_to_plot = []
    stack_of_colors = []
    stack_of_labels = []
    for i in range(num_classes):
        if i != bg_id:
            stack_to_plot.append(start_mismatch[i])
            stack_of_colors.append(class_colors[i])
            stack_of_labels.append(classes[i])
    # Histo stack
    bins = np.linspace(-300, 300, 100)
    plt.figure()
    patches = []
    for i in range(num_classes-1):
        patches.append(Rectangle((0, 0), 0, 0, color=stack_of_colors[i]))
    for i in range(num_classes-1):
        plt.subplot(num_classes-1, 1, i+1)
        plt.hist(stack_to_plot[i], bins=bins, alpha=0.7, color=stack_of_colors[i], label=stack_of_labels[i], lw=0)
    # plt.hist(stack_to_plot, bins=bins, alpha=0.7, color=stack_of_colors, label=stack_of_labels)
    plt.legend(patches, stack_of_labels, loc='center left', borderaxespad=0, bbox_to_anchor=(1.05, 0.5),
               prop={'size': 8})
    plt.show()


def _find_overlap_seg(seg_list, id):
    for seg_id in range(len(seg_list)):
        if seg_list[seg_id][1] < id:
            continue
        elif seg_list[seg_id][0] > id:
            return -1
        else:
            return seg_id
    return -1


def _find_seg_start_within(seg_list, start, stop):
    for seg_id in range(len(seg_list)):
        if seg_list[seg_id][0] < start:
            continue
        elif seg_list[seg_id][0] > stop:
            return -1
        else:
            return seg_id
    return -1


def _find_seg_end_within(seg_list, start, stop):
    found_seg_id = -1
    for seg_id in range(len(seg_list)):
        if seg_list[seg_id][1] < start:
            continue
        elif seg_list[seg_id][0] > stop:
            return found_seg_id
        else:
            found_seg_id = seg_id
    return found_seg_id


def _get_timeoffset_measures(classes, truth, prediction, time_list):
    num_classes = len(classes)
    start_mismatch = [list([]) for i in range(num_classes)]
    stop_mismatch = [list([]) for i in range(num_classes)]
    # Processing segmentation first!
    for j in range(num_classes):
        pred_segs = []
        truth_segs = []
        prev_pred = False
        prev_truth = False
        tseg_start = 0
        tseg_stop = 0
        pseg_start = 0
        pseg_stop = 0
        for i in range(truth.shape[0]):
            cur_truth = (int(truth[i]) == j)
            cur_pred = (int(prediction[i]) == j)
            # Truth segments
            if cur_truth != prev_truth:
                if cur_truth:
                    tseg_start = i
                elif tseg_stop != 0:
                    truth_segs.append((tseg_start, tseg_stop))
            tseg_stop = i
            # Prediction segments
            if cur_pred != prev_pred:
                if cur_pred:
                    pseg_start = i
                elif pseg_stop != 0:
                    pred_segs.append((pseg_start, pseg_stop))
            pseg_stop = i
            prev_truth = cur_truth
            prev_pred = cur_pred
        # Add compensated segments to predictions egments
        for ts, (tseg_start, tseg_stop) in enumerate(truth_segs):
            ps = _find_overlap_seg(pred_segs, tseg_start)
            if ps == -1:
                # potential underfill or deletion
                ps = _find_seg_start_within(pred_segs, tseg_start, tseg_stop)
                if ps != -1:
                    pseg_start = pred_segs[ps][0]
                    offset = (time_list[tseg_start] - time_list[pseg_start]).total_seconds()
                    if abs(offset) < 18000:
                        start_mismatch[j].append(offset)
            else:
                pseg_start = pred_segs[ps][0]
                # Check the end of previous truth
                if ts > 1 and truth_segs[ts-1][1] >= pseg_start:
                    continue
                else:
                    offset = (time_list[tseg_start] - time_list[pseg_start]).total_seconds()
                    if abs(offset) < 18000:
                        # Calculate overfill
                        start_mismatch[j].append((time_list[tseg_start] - time_list[pseg_start]).total_seconds())
        for ts, (tseg_start, tseg_stop) in enumerate(truth_segs):
            ps = _find_overlap_seg(pred_segs, tseg_stop)
            if ps == -1:
                # potential underfill or deletion
                ps = _find_seg_end_within(pred_segs, tseg_start, tseg_stop)
                if ps != -1:
                    pseg_stop = pred_segs[ps][1]
                    offset = (time_list[tseg_stop] - time_list[pseg_stop]).total_seconds()
                    if tseg_stop != pseg_stop and abs(offset) < 18000:
                        stop_mismatch[j].append(offset)
            else:
                pseg_stop = pred_segs[ps][1]
                # Check the end of previous truth
                if ts < len(truth_segs) - 1 and truth_segs[ts-1][0] <= pseg_stop:
                    continue
                else:
                    offset = (time_list[tseg_stop] - time_list[pseg_stop]).total_seconds()
                    if abs(offset) < 18000:
                        # Calculate overfill
                        stop_mismatch[j].append(offset)
        # print("class: %d" % j)
        # print("pred_segs: %d %s" % (len(pred_segs), str(pred_segs)))
        # print("truth_segs: %d %s" % (len(truth_segs), str(truth_segs)))
        # print("start_mismatch: %s" % start_mismatch)
        # print("stop_mismatch: %s" % stop_mismatch)
    return start_mismatch, stop_mismatch


def _get_timeliness_measures(classes, truth, prediction, time_list):
    num_classes = len(classes)
    start_mismatch = [list([]) for i in range(num_classes)]
    stop_mismatch = [list([]) for i in range(num_classes)]
    # Processing segmentation first!
    for j in range(num_classes):
        pred_segs = []
        truth_segs = []
        prev_pred = False
        prev_truth = False
        tseg_start = 0
        tseg_stop = 0
        pseg_start = 0
        pseg_stop = 0
        for i in range(truth.shape[0]):
            cur_truth = (int(truth[i]) == j)
            cur_pred = (int(prediction[i]) == j)
            # Truth segments
            if cur_truth != prev_truth:
                if cur_truth:
                    tseg_start = i
                elif tseg_stop != 0:
                    truth_segs.append((tseg_start, tseg_stop))
            tseg_stop = i
            # Prediction segments
            if cur_pred != prev_pred:
                if cur_pred:
                    pseg_start = i
                elif pseg_stop != 0:
                    pred_segs.append((pseg_start, pseg_stop))
            pseg_stop = i
            prev_truth = cur_truth
            prev_pred = cur_pred
        # Add compensated segments to predictions egments
        for ts, (tseg_start, tseg_stop) in enumerate(truth_segs):
            ps = _find_overlap_seg(pred_segs, tseg_start)
            if ps == -1:
                # potential underfill or deletion
                ps = _find_seg_start_within(pred_segs, tseg_start, tseg_stop)
                if ps != -1:
                    pseg_start = pred_segs[ps][0]
                    offset = (time_list[tseg_start] - time_list[pseg_start]).total_seconds()
                    if tseg_start != pseg_start and abs(offset) < 18000:
                        start_mismatch[j].append(offset)
            else:
                pseg_start = pred_segs[ps][0]
                # Check the end of previous truth
                if ts > 1 and truth_segs[ts-1][1] >= pseg_start:
                    continue
                else:
                    offset = (time_list[tseg_start] - time_list[pseg_start]).total_seconds()
                    if tseg_start != pseg_start and abs(offset) < 18000:
                        # Calculate overfill
                        start_mismatch[j].append((time_list[tseg_start] - time_list[pseg_start]).total_seconds())
        for ts, (tseg_start, tseg_stop) in enumerate(truth_segs):
            ps = _find_overlap_seg(pred_segs, tseg_stop)
            if ps == -1:
                # potential underfill or deletion
                ps = _find_seg_end_within(pred_segs, tseg_start, tseg_stop)
                if ps != -1:
                    pseg_stop = pred_segs[ps][1]
                    offset = (time_list[tseg_stop] - time_list[pseg_stop]).total_seconds()
                    if tseg_stop != pseg_stop and abs(offset) < 18000:
                        stop_mismatch[j].append(offset)
            else:
                pseg_stop = pred_segs[ps][1]
                # Check the end of previous truth
                if ts < len(truth_segs) - 1 and truth_segs[ts-1][0] <= pseg_stop:
                    continue
                else:
                    offset = (time_list[tseg_stop] - time_list[pseg_stop]).total_seconds()
                    if tseg_stop != pseg_stop and abs(offset) < 18000:
                        # Calculate overfill
                        stop_mismatch[j].append(offset)
        # print("class: %d" % j)
        # print("pred_segs: %d %s" % (len(pred_segs), str(pred_segs)))
        # print("truth_segs: %d %s" % (len(truth_segs), str(truth_segs)))
        # print("start_mismatch: %s" % start_mismatch)
        # print("stop_mismatch: %s" % stop_mismatch)
    return start_mismatch, stop_mismatch


def _get_timeliness_measures_depricated(classes, truth, prediction, truth_scoring, prediction_scoring, time_list):
    num_classes = len(classes)
    start_mismatch = [list([]) for i in range(num_classes)]
    stop_mismatch = [list([]) for i in range(num_classes)]
    # For each Underfill, Overfill
    prev_truth = -1
    for i in range(truth.shape[0]):
        cur_truth = int(truth[i])
        # Overfill/Underfill only occur at the boundary of any activity event, so look for the boundary first
        if cur_truth != prev_truth:
            truth_time = time_list[i]
            # Check the start boundary
            if truth[i] == prediction[i]:
                # If current prediction is correct, then it can only be overfill of current truth label.
                j = i - 1
                while j >= 0 and prediction_scoring[j] == 'O':
                    j -= 1
                # If there is no overfill for cur_truth, and the current truth and prediction are the same,
                # then there is no start_boundary mismatch.
                start_mismatch[cur_truth].append((time_list[j + 1] - truth_time).total_seconds())
            else:
                # If current prediction is incorrect, then it can only be underfill of current truth label at start
                # boundary.
                j = i
                while j < truth.shape[0] and truth_scoring[j] == 'U':
                    j += 1
                if j != i and j < truth.shape[0]:
                    start_mismatch[cur_truth].append((time_list[j-1] - truth_time).total_seconds())
            # Check the stop boundary
            if i > 0:
                if prediction[i-1] == truth[i-1]:
                    # Previous prediction is correct, then it can only be overfill of previous truth.
                    # If there is no overfill, the stop boundary is accurate
                    j = i
                    while prediction_scoring[j] == 'o':
                        j += 1
                    stop_mismatch[prev_truth].append((time_list[j-1] - truth_time).total_seconds())
                else:
                    # Check Underfill for prev_truth (at the stop boundary)
                    j = i - 1
                    while j >= 0 and truth_scoring[j] == 'u':
                        j -= 1
                    if j != i - 1:
                        stop_mismatch[prev_truth].append((time_list[j + 1] - truth_time).total_seconds())
            if prev_truth != -1:
                if len(stop_mismatch[prev_truth]) > 0 and abs(stop_mismatch[prev_truth][-1]) > 1800:
                    logger.warning('Stop mismatch is over half an hour: %s at %d (%s) - %f' %
                                   (classes[prev_truth], i, time_list[i],
                                    stop_mismatch[prev_truth][-1]))
                if len(start_mismatch[cur_truth]) > 0 and abs(start_mismatch[cur_truth][-1]) > 1800:
                    logger.warning('Start mismatch is over half an hour: %s at %d (%s) - %f' %
                                   (classes[cur_truth], i, time_list[i],
                                    start_mismatch[cur_truth][-1]))
        # Update prev truth
        prev_truth = cur_truth
    # Sort all arrays
    for i in range(num_classes):
        start_mismatch[i].sort()
        stop_mismatch[i].sort()
    # Return
    return start_mismatch, stop_mismatch


def generate_latex_table(methods, classes, recall_metrics, precision_matrics,
                          background_class=None, filename=None,
                          as_percent=True, metric_name='recall'):
    bg_class_id = _get_bg_class_id(classes, background_class)
    metric_labels, metric_indices = _get_metric_label_dict(metric_name='recall')
    rmp = _gether_per_class_metrics(methods, classes, recall_metrics, True,
                                    metric_labels, metric_indices)
    rmr = _gether_per_class_metrics(methods, classes, recall_metrics, False,
                                    metric_labels, metric_indices)
    metric_labels, metric_indices = _get_metric_label_dict(metric_name='precision')
    pmp = _gether_per_class_metrics(methods, classes, precision_matrics, True,
                                    metric_labels, metric_indices)
    pmr = _gether_per_class_metrics(methods, classes, precision_matrics, False,
                                    metric_labels, metric_indices)
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('\\multirow{2}{*}{Models} & \\multirow{2}{*}{Activities} & '
            '\\multirow{2}{*}{Total Truth} & \\multicolumn{2}{|c|}{Recall} & '
            '\\multirow{2}{*}{Total Prediction} & \\multicolumn{2}{|c|}{Precision}  \\\\ \\hline\n')
    f.write('& & & C only & U included & & C only & O included \\\\ \\hline \n')
    for i, method in enumerate(methods):
        f.write('\\multirow{%d}{*}{%s} & ' % (len(classes), method.replace('_', '\_')))
        for j, target in enumerate(classes):
            if j != 0:
                f.write('& ')
            f.write('%s & '
                    '%d & %d (%.2f) & %d (%.2f)  & '
                    '%d & %d (%.2f) & %d (%.2f)  \\\\ \n' %
                    (target.replace('_', '\_'),
                     rmr[i][j,:].sum(), rmr[i][j,0], rmp[i][j,0],
                     rmr[i][j,0]+rmr[i][j,1]+rmr[i][j,2], rmp[i][j,0]+rmp[i][j,1]+rmp[i][j,2],
                     pmr[i][j,:].sum(), pmr[i][j,0], pmp[i][j,0],
                     pmr[i][j,0]+pmr[i][j,1]+pmr[i][j,2], pmp[i][j,0]+pmp[i][j,1]+pmp[i][j,2],
                     )
                    )
        f.write('\\hline\n')
    f.close()


def generate_seg_latex_table(methods, classes, recall_metrics, precision_matrics,
                             background_class=None, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    metric_labels, metric_indices = _get_metric_label_dict(metric_name='recall')
    rmp = _gether_per_class_metrics(methods, classes, recall_metrics, True,
                                    metric_labels, metric_indices)
    rmr = _gether_per_class_metrics(methods, classes, recall_metrics, False,
                                    metric_labels, metric_indices)
    metric_labels, metric_indices = _get_metric_label_dict(metric_name='precision')
    pmp = _gether_per_class_metrics(methods, classes, precision_matrics, True,
                                    metric_labels, metric_indices)
    pmr = _gether_per_class_metrics(methods, classes, precision_matrics, False,
                                    metric_labels, metric_indices)
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('Metric & Activities')
    for method in methods:
        f.write('& %s' % method.replace('_', '\_'))
    f.write('\\\\ \\hline \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            if i == 0:
                f.write('\multirow{%d}{*}{Recall} & ' % (len(classes) - 1))
            else:
                f.write(' & ')
            f.write('%s ' % activity.replace('_', '\_'))
            # Find maximum and store index
            temp_array = np.array([rmp[j][i,0] for j in range(len(methods))])
            max_index = temp_array.argpartition(-2)[-2:]
            for j, method in enumerate(methods):
                if j in max_index:
                    f.write('& \\textbf{%d/%.2f\\%%} ' % (rmr[j][i,0], rmp[j][i,0]* 100))
                else:
                    f.write('& %d/%.2f\\%% ' % (rmr[j][i,0], rmp[j][i,0]* 100))
            f.write('\\\\ \n')
    f.write('\\hline \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            if i == 0:
                f.write('\multirow{%d}{*}{Precision} & ' % (len(classes) - 1))
            else:
                f.write(' & ')
            f.write('%s ' % activity.replace('_', '\_'))
            # Find maximum and store index
            temp_array = np.array([pmp[j][i,0] for j in range(len(methods))])
            max_index = temp_array.argpartition(-2)[-2:]
            for j, method in enumerate(methods):
                if j in max_index:
                    f.write('& \\textbf{%d/%.2f\\%%} ' % (pmr[j][i,0], pmp[j][i,0]* 100))
                else:
                    f.write('& %d/%.2f\\%% ' % (pmr[j][i,0], pmp[j][i,0]* 100))
            f.write('\\\\ \n')
    f.write('\\hline \n')


def generate_event_recall_table(methods, classes, recall_metrics,
                                background_class=None, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    metric_labels, metric_indices = _get_metric_label_dict(metric_name='recall')
    rmp = _gether_per_class_metrics(methods, classes, recall_metrics, True,
                                    metric_labels, metric_indices)
    rmr = _gether_per_class_metrics(methods, classes, recall_metrics, False,
                                    metric_labels, metric_indices)
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('Activities')
    for method in methods:
        f.write('& %s' % method.replace('_', '\_'))
    f.write('\\\\ \\hline \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            f.write(' & ')
            f.write('%s ' % activity.replace('_', '\_'))
            # Find maximum and store index
            temp_array = np.array([rmp[j][i, 0] for j in range(len(methods))])
            max_index = temp_array.argpartition(-2)[-2:]
            for j, method in enumerate(methods):
                if j in max_index:
                    f.write('& \\textbf{%.2f\\%%} ' % (rmp[j][i,0]* 100))
                else:
                    f.write('& %.2f\\%% ' % (rmp[j][i,0]* 100))
            f.write('\\\\ \n')
    f.write('\\hline \n')
    f.write('Recall (micro) &')
    total_correct = np.array([np.sum(rmr[j][:, 0]) - rmr[j][bg_class_id, 0] for j in range(len(methods))])
    total_events = np.array([total_correct[j] + np.sum(rmr[j][:, 4]) - rmr[j][bg_class_id, 4]
                             for j in range(len(methods))])
    max_index = total_correct.argpartition(-2)[-2:]
    for j, method in enumerate(methods):
        if j in max_index:
            f.write('& \\textbf{%.2f\\%%} ' % (total_correct[j] / total_events[j] * 100))
        else:
            f.write('& %.2f\\%% ' % (total_correct[j] / total_events[j] * 100))
    f.write('\\\\ \n')
    f.write('\\hline \n')
    logger.debug('Total Events: %s' % str(total_events))


def generate_timeliness_table(methods, classes, result_array,
                              background_class, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    timeliness_values = []
    for i, method in enumerate(methods):
        start_mismatch, stop_mismatch = _get_timeliness_measures(classes, result_array[i][0], result_array[i][1],
                                                                 result_array[i][4])
        cur_timeliness = [start_mismatch[j] + stop_mismatch[j] for j in range(len(classes))]
        timeliness_values.append([np.abs(np.array(cur_timeliness[j])) for j in range(len(classes))])
    # Average, <60, >60
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('Activities & Metrics ')
    for method in methods:
        f.write('& %s' % method.replace('_', '\_'))
    f.write('\\\\ \\hline \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            f.write('\multirow{3}{*}{%s} & ' % activity.replace('_', '\_'))
            f.write('Average ')
            # Find maximum and store index
            for j, method in enumerate(methods):
                if len(timeliness_values[j][i]) == 0:
                    average_time = 0.
                else:
                    average_time = np.average(timeliness_values[j][i])
                f.write('& %.2f s' % average_time)
            f.write('\\\\ \n')
            f.write(' & ')
            f.write('<60s ')
            for j, method in enumerate(methods):
                number = (timeliness_values[j][i] <= 60).sum()
                if len(timeliness_values[j][i]) == 0:
                    percentage = 0.0
                else:
                    percentage = float(number)/len(timeliness_values[j][i]) * 100
                f.write('& %d/%.2f\\%% ' % (number, percentage))
            f.write('\\\\ \n')
            f.write(' & ')
            f.write('>60s ')
            for j, method in enumerate(methods):
                number = (timeliness_values[j][i] > 60).sum()
                if len(timeliness_values[j][i]) == 0:
                    percentage = 0.0
                else:
                    percentage = float(number)/len(timeliness_values[j][i]) * 100
                f.write('& %d/%.2f\\%% ' % (number, percentage))
            f.write('\\\\ \\hline \n')


def generate_timeliness_within60_table(methods, classes, result_array,
                                       background_class, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    timeliness_values = []
    for i, method in enumerate(methods):
        start_mismatch, stop_mismatch = _get_timeoffset_measures(classes, result_array[i][0], result_array[i][1],
                                                                 result_array[i][4])
        cur_timeliness = [start_mismatch[j] + stop_mismatch[j] for j in range(len(classes))]
        timeliness_values.append([np.abs(np.array(cur_timeliness[j])) for j in range(len(classes))])
    # Average, <60, >60
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('\\textbf{Activities} ')
    for method in methods:
        f.write('& \\textbf{%s} ' % method.replace('_', ' '))
    f.write('\\\\ \\midrule \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            f.write('%s & ' % activity.replace('_', ' '))
            for j, method in enumerate(methods):
                number = (timeliness_values[j][i] <= 60).sum()
                if len(timeliness_values[j][i]) == 0:
                    percentage = 0.0
                else:
                    percentage = float(number)/len(timeliness_values[j][i]) * 100
                f.write('& %.2f\\%% ' % (percentage))
            f.write('\\\\ \n')
    f.write('\\bottomrule\n')


def generate_timeliness_avg_table(methods, classes, result_array,
                                  background_class, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    timeliness_values = []
    for i, method in enumerate(methods):
        start_mismatch, stop_mismatch = _get_timeliness_measures(classes, result_array[i][0], result_array[i][1],
                                                                 result_array[i][4])
        cur_timeliness = [start_mismatch[j] + stop_mismatch[j] for j in range(len(classes))]
        timeliness_values.append([np.abs(np.array(cur_timeliness[j])) for j in range(len(classes))])
    # Average, <60, >60
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('\\textbf{Activities} ')
    for method in methods:
        f.write('& \\textbf{%s} ' % method.replace('_', ' '))
    f.write('\\\\ \\midrule \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            f.write('%s ' % activity.replace('_', ' '))
            # Find maximum and store index
            for j, method in enumerate(methods):
                if len(timeliness_values[j][i]) == 0:
                    average_time = 0.
                else:
                    average_time = np.average(timeliness_values[j][i])
                f.write('& %.1f' % average_time)
            f.write('\\\\ \n')
    f.write('\\bottomrule \n')


def generate_offset_per_table(methods, classes, result_array,
                              background_class, filename=None):
    bg_class_id = _get_bg_class_id(classes, background_class)
    timeliness_values = []
    for i, method in enumerate(methods):
        start_mismatch, stop_mismatch = _get_timeoffset_measures(classes, result_array[i][0], result_array[i][1],
                                                                 result_array[i][4])
        cur_timeliness = [start_mismatch[j] + stop_mismatch[j] for j in range(len(classes))]
        timeliness_values.append([np.abs(np.array(cur_timeliness[j])) for j in range(len(classes))])
    # Average, <60, >60
    if filename is None:
        f = sys.stdout
    else:
        f = open(filename, 'w')
    f.write('\\textbf{Activities} ')
    for method in methods:
        f.write('& \\textbf{%s} ' % method.replace('_', ' '))
    f.write('\\\\ \\midrule \n')
    for i, activity in enumerate(classes):
        if i != bg_class_id:
            f.write('%s ' % activity.replace('_', ' '))
            # Find maximum and store index
            for j, method in enumerate(methods):
                total_num = len(timeliness_values[j][i])/2
                nonzero_num = np.count_nonzero(timeliness_values[j][i])
                f.write('& %d/%d' % (nonzero_num, total_num))
            f.write('\\\\ \n')
    f.write('\\bottomrule \n')

