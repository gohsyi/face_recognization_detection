import tensorflow as tf


def dense(inputs, units, activation=None):
    """
    construct a fully-connected layer
    :param inputs: inputs of fc
    :param units: number of outputs
    :param activation: activation function
    :return: corresponding created FC layer
    """

    return tf.layers.dense(
        inputs, units,
        activation=activation,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.5)
    )


def mlp(inputs, latents, activation=None):
    """
    construct a multi-layer perception
    :param inputs: inputs
    :param latents: latent sizes
    :param activation: activation function
    :return: corresponding created MLP
    """

    last_latent = inputs
    for i, hdim in enumerate(latents):
        last_latent = dense(last_latent, hdim, activation)
    return last_latent


def conv2d(inputs, filters, ksize):
    """
    construct a 2-d convolutional layer
    :param inputs: inputs
    :param filters: filters
    :param ksize: kernel size
    :return: corresponding created Conv-2d layer
    """

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=ksize,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        padding='same',
    )


def max_pooling2d(inputs, psize, strides):
    """
    construct a 2-d max-pooling layer
    :param inputs: inputs
    :param psize: pool size
    :param strides: strides
    :return: corresponding created Max-pooling layer
    """

    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=psize,
        strides=strides,
        padding='same',
    )
