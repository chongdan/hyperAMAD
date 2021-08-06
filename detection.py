import tensorflow as tf



def compute_75(pool2):
    tensor_0 = pool2[:, :, 0:3, :, :]
    tensor_2 = pool2[:, :, 18:21, :, :]
    tensor_5 = tf.concat([tensor_2, tensor_0], 2)
    tensor_5 = tf.reshape(tensor_5, [tensor_5.shape[0].value, tensor_5.shape[1].value, 1, tensor_5.shape[2].value * tensor_5.shape[3].value,
                                      tensor_5.shape[-1].value])
    tensor_1 = pool2[:, :, 3:18, 0:3, :]
    tensor_1 = tf.reshape(tensor_1, [tensor_1.shape[0].value, tensor_1.shape[1].value, 1, tensor_1.shape[2].value * tensor_1.shape[3].value,
                                      tensor_1.shape[-1].value])

    tensor_5 = tf.concat([tensor_5, tensor_1], 3)
    tensor_0 = pool2[:, :, 3:18, 18:21, :]
    tensor_1 = tf.reshape(tensor_0, [tensor_0.shape[0].value, tensor_0.shape[1].value, 1, tensor_0.shape[2].value * tensor_0.shape[3].value,
                                      tensor_0.shape[-1].value])
    tensor_5 = tf.concat([tensor_5, tensor_1], 3)

    tensor_1 = pool2[:, :, 10:11, 10:11, :]

    tensor_5 = tf.acos((tensor_5 * tensor_1) / (tf.norm(tensor_5) * tf.norm(tensor_1))) * tf.abs(tensor_5 - tensor_1)
    return tensor_5


def compute_73(pool2):
    tensor_0 = pool2[:, :, 0:6, :, :]
    tensor_2 = pool2[:,  :, 15:21, :, :]
    tensor_5 = tf.concat([tensor_2, tensor_0], 2)
    tensor_5 = tf.reshape(tensor_5, [tensor_5.shape[0].value, tensor_5.shape[1].value, 1, tensor_5.shape[2].value * tensor_5.shape[3].value,
                                      tensor_5.shape[-1].value])
    tensor_1 = pool2[:, :, 6:15, 0:6, :]
    tensor_1 = tf.reshape(tensor_1, [tensor_1.shape[0].value, tensor_1.shape[1].value, 1, tensor_1.shape[2].value * tensor_1.shape[3].value,
                                      tensor_1.shape[-1].value])

    tensor_5 = tf.concat([tensor_5, tensor_1], 3)
    tensor_0 = pool2[:, :, 6:15, 15:21, :]
    tensor_1 = tf.reshape(tensor_0, [tensor_0.shape[0].value, tensor_0.shape[1].value, 1, tensor_0.shape[2].value * tensor_0.shape[3].value,
                                      tensor_0.shape[-1].value])
    tensor_5 = tf.concat([tensor_5, tensor_1], 3)

    tensor_1 = pool2[:, :, 10:11, 10:11, :]

    tensor_5 = tf.acos((tensor_5 * tensor_1) / (tf.norm(tensor_5) * tf.norm(tensor_1))) * tf.abs(tensor_5 - tensor_1)
    return tensor_5

def compute_53(pool2):
    tensor_0 = pool2[:, :, 0:3, :, :]
    tensor_2 = pool2[:, :, 12:15, :, :]
    tensor_5 = tf.concat([tensor_2, tensor_0], 2)
    tensor_5 = tf.reshape(tensor_5, [tensor_5.shape[0].value,tensor_5.shape[1].value, 1, tensor_5.shape[2].value * tensor_5.shape[3].value,
                                      tensor_5.shape[-1].value])
    tensor_1 = pool2[:, :, 3:12, 0:3, :]
    tensor_1 = tf.reshape(tensor_1, [tensor_1.shape[0].value, tensor_1.shape[1].value, 1, tensor_1.shape[2].value * tensor_1.shape[3].value,
                                      tensor_1.shape[-1].value])

    tensor_5 = tf.concat([tensor_5, tensor_1], 3)
    tensor_0 = pool2[:, :, 3:12, 12:15, :]
    tensor_1 = tf.reshape(tensor_0, [tensor_0.shape[0].value, tensor_0.shape[1].value, 1, tensor_0.shape[2].value * tensor_0.shape[3].value,
                                      tensor_0.shape[-1].value])
    tensor_5 = tf.concat([tensor_5, tensor_1], 3)

    tensor_1 = pool2[:, :, 7:8, 7:8, :]

    tensor_5 = tf.acos((tensor_5 * tensor_1) / (tf.norm(tensor_5) * tf.norm(tensor_1))) * tf.abs(tensor_5 - tensor_1)
    return tensor_5

def compute_531(pool2):

    tensor_0 = pool2[:, 0:1, :, :, :]
    tensor_1 = pool2[:, 4:5, :, :, :, ]
    tensor_5 = tf.concat([tensor_1, tensor_0], 2)
    tensor_0 = pool2[:, 1:4, 0:1, :, :]
    tensor_0 = tf.reshape(tensor_0, (-1, 1, 3, pool2.shape[3].value, 20))
    tensor_5 = tf.concat([tensor_5, tensor_0], 2)
    tensor_0 = pool2[:, 1:4, 4:5, :, :]
    tensor_0 = tf.reshape(tensor_0, (-1, 1, 3, pool2.shape[3].value, 20))
    tensor_5 = tf.concat([tensor_5, tensor_0], 2)
    tensor_1 = pool2[:, 2:3, 2:3, :, :]

    tensor_5 = tf.acos((tensor_5 * tensor_1) / (tf.norm(tensor_5) * tf.norm(tensor_1))) * tf.abs(tensor_5 - tensor_1)


    return tensor_5