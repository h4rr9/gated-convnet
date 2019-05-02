import tensorflow as tf


def conv1d(inputs, filters, k_size, strides, padding, scope_name='conv1d'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channels = inputs.shape[-1]

        kernel = tf.get_variable('kernel', shape=[
                                 k_size, in_channels, filters], initializer=tf.truncated_normal_initializer())

        biases = tf.get_variable(
            'biases', shape=[filters], initializer=tf.random_normal_initializer())

        conv = tf.nn.conv1d(inputs, kernel, stride=strides,
                            padding=padding, use_cudnn_on_gpu=True)

        conv_output = tf.add(conv, biases, name=scope.name)

    return conv_output


def casual_conv(inputs, filters, k_size, strides, scope_name='casual_conv'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        paddings = tf.constant([[0, 0], [k_size - 1, 0], [0, 0]])

        padded_inputs = tf.pad(
            tensor=inputs, paddings=paddings, name=scope.name)

        output = conv1d(inputs=padded_inputs, filters=filters,
                        k_size=k_size, strides=strides, padding='VALID')

    return output


def sigmoid(inputs, scope_name='sigmoid'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        sigma = tf.math.sigmoid(inputs, name=scope.name)
    return sigma


def gate(input_1, input_2, scope_name='gated_linear'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        gate_output = tf.math.multiply(
            input_1, sigmoid(input_2), name=scope.name)
    return gate_output


def gate_block(inputs, k_size, n, bottleneck=False, n_bottleneck=None, scope_name='gate'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channel = inputs.shape[-1]
        if(in_channel != n):
            residual = conv1d(inputs, filters=n, k_size=1,
                            padding='VALID', strides=1, scope_name='residual_red')
        else:
            residual = inputs

        if bottleneck is False:
            A = casual_conv(inputs=inputs, filters=n, k_size=k_size,
                            strides=1, scope_name='convA')

            B = casual_conv(inputs=inputs, filters=n, k_size=k_size,
                            strides=1, scope_name='convB')

            gate_block_output = gate(A, B)
        else:
            red_input = conv1d(inputs, filters=n_bottleneck,
                               k_size=1, strides=1, padding='VALID', scope_name='conv1d_red')

            A = casual_conv(inputs=red_input, filters=n_bottleneck, k_size=k_size,
                            strides=1, scope_name='convA')

            B = casual_conv(inputs=red_input, filters=n_bottleneck, k_size=k_size,
                            strides=1, scope_name='convB')

            gate_output = gate(A, B)

            gate_block_output = conv1d(
                gate_output, filters=n, k_size=1, strides=1, padding='VALID', scope_name='conv1d_exp')

        output = tf.add(gate_block_output, residual, name=scope.name)
    return output


def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]

        w = tf.get_variable('weights',
                            [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())

        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))

        out = tf.add(tf.matmul(inputs, w), b, name=scope.name)
    return out


def flatten(inputs, scope_name='flatten'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        feature_dim = inputs.shape[1] * inputs.shape[2]

        flatten = tf.reshape(inputs, shape=[-1, feature_dim], name=scope.name)

    return flatten


def Dropout(inputs, rate, scope_name='dropout'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        dropout = tf.nn.dropout(inputs, keep_prob=1 - rate, name=scope.name)
    return dropout


def l2_norm(inputs, alpha, scope_name='l2_norm'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        norm = alpha * tf.divide(inputs,
                                 tf.norm(inputs, ord='euclidean'),
                                 name=scope.name)
    return norm