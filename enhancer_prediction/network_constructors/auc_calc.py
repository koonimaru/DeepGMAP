import tensorflow as tf

def auc_pr(true, prob, threshold):

    pred = tf.where(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    tn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.logical_not(tf.cast(true, tf.bool)))
    FPR = tf.truediv(tf.reduce_sum(tf.cast(fp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tn, fp), tf.int32)))
    TPR = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    PPV = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))

    return FPR, TPR, PPV


def auc_pr2(true, prob, threshold):
    FPR, _ = tf.metrics.false_positives_at_thresholds(true, prob, [threshold])
    TPR, _ = tf.metrics.true_negatives_at_thresholds(true, prob, [threshold])
    PPV, _ = tf.metrics.precision_at_thresholds(true, prob, [threshold])

    return FPR, TPR, PPV