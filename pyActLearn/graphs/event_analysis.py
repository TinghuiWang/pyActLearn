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
                                                             truth_scoring, prediction_scoring, time_list)
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


def _get_timeliness_measures(classes, truth, prediction, truth_scoring, prediction_scoring, time_list):
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
