
Event-based Scoring
===================

[Minnen2006]_ and [Ward2011]_ proposed a set of performance metrics and visualizations for continuous activity
recognition. In both papers, the authors examine the issues in continuous activity recognition and argued that the
traditional standard multi-class evaluation methods fail to capture common artefacts found in continuous AR.

In both papers, the false positives and false negatives are further divided into six categories to faithfully capture
the nature of those errors in the context of continuous AR.

Whenever an error occurs, it is both a false positive with respect to the prediction label and a false negative with
respect to the ground truth label.

False positive errors are divided into the following three categories:

- Insertion (I): A FP that corresponds exactly to an inserted return.
- Merge (M): A FP that occurs between two TP segments within a merge return.
- Overfill (O): A FP that occurs at the start or end of a partially matched return.

False negatives errors are divided into the following three categories:

- Deletion (D): A FN that corresponds exactly to a deleted event.
- Fragmenting (F): A FN that corresponds exactly to a deleted event.
- Underfill (U): A FN that occurs at the start or end of a detected event.

API Reference
-------------

.. autofunction:: pyActLearn.performance.event.score_segment

.. autofunction:: pyActLearn.performance.event.per_class_event_scoring

.. autofunction:: pyActLearn.performance.event.per_class_segment_scoring