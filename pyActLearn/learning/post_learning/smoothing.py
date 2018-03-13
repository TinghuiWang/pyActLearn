def activity_smoothing(label, threshold=2):
    """The function performs a smoothing on the activity labels

    Due to the continuity of daily activities, any event.rst may last for some time.
    To clean up glitches in activity labels, this function looks for a "threshold" amount of
    activity labels before it acknowledge the new activity is performed by the resident.

    Args:
        label (:obj:`numpy.ndarray`): Activity labels.
        threshold (:obj:`int`): Activity threshold value.
    """
    smoothed_label = label.copy()
    old_event = label[0]
    new_event = label[0]
    count = threshold
    # Go through the whole label sets
    for i in range(label.shape[0]):
        # Assign old event.rst to output
        smoothed_label[i] = old_event
        # If the current is the same as new_event, increase the count
        # otherwise, a new_event is detected, log it and reset count.
        if label[i] == new_event:
            count += 1
        else:
            new_event = label[i]
            count = 0
        # If new_event is not the same as the old one, and the count exceeds a threshold
        # Smooth it out.
        if new_event != old_event and count > threshold:
            old_event = new_event
            for j in range(threshold):
                smoothed_label[i - j] = new_event
    # Return the new label
    return smoothed_label

