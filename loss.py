import tensorflow as tf

def multiclass_hinge_loss(scores, classes):
    """
    Computer svm loss for multi classification
    Args:
      labels: One hot label, int32 tensor of shape [batch_size, num_classes] or [batch_size, seq_len, num_classes].
      logits: A float32 tensor of shape [batch_size,num_classes] or [batch_size, seq_len, num_classes]..
    Returns:
      A tensor of the same shape as [batch_sizes] or [batch_size, seq_len].
    """
    true_scores = tf.where(classes > 0, scores, tf.zeros(tf.shape(scores)))
    true_scores = tf.reduce_sum(true_scores, axis=-1, keepdims=True)
    L = tf.nn.relu((1 - true_scores + scores) * (1 - classes))
    final_loss = tf.reduce_mean(L, axis=-1)
    return final_loss
