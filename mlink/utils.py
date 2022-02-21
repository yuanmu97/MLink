import tensorflow as tf 


class BBoxIoU(tf.keras.metrics.Metric):
    """BBoxIoU
    Mean Bounding Box IoU
    """
    def __init__(self, name='BBoxIoU', **kwargs):
        super(BBoxIoU, self).__init__(name=name, **kwargs)
        # sum of each samples' IoU
        self.iou = self.add_weight(name='iou', initializer='zeros')
        # sample count
        self.count = self.add_weight(name='count', initializer='zeros')
        

    def update_state(self, y_true, y_pred):
        """update_state
        Args:
            y_true, y_pred (array): [BATCHSIZE*[x, y, w, h]]
        """
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        true_area = y_true[:,2] * y_true[:,3]
        
        pred_area = y_pred[:,2] * y_pred[:,3]

        inter_x1 = tf.maximum(y_true[:,0], y_pred[:,0])
        inter_y1 = tf.maximum(y_true[:,1], y_pred[:,1])

        inter_x2 = tf.minimum(y_true[:,0]+y_true[:,2], y_pred[:,0]+y_pred[:,2])
        inter_y2 = tf.minimum(y_true[:,1]+y_true[:,3], y_pred[:,1]+y_pred[:,3])

        # using tf.maximum(0., *) to avoid negative area value
        inter_area = tf.maximum(0., (inter_x2-inter_x1)) * tf.maximum(0., (inter_y2-inter_y1))
        union_area = true_area + pred_area - inter_area

        # update the sum of IoU & count
        self.iou.assign_add(tf.math.reduce_sum(inter_area / union_area))
        self.count.assign_add(batch_size)


    def result(self):
        """result
        Returns:
            mean_iou (float): Mean IoU
        """
        # mean IoU
        return tf.math.divide_no_nan(self.iou, self.count)