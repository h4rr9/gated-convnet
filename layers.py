import tensorflow as tf


def conv1d(inputs, filters, k_size, stride, padding, scope_name="conv", _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channels = inputs.shape[-1]

        if _weights is None:
            kernel = tf.get_variable(
                "kernel",
                [k_size, in_channels, filters],
                initializer=tf.truncated_normal_initializer(),
            )

            biases = tf.get_variable(
                "biases", [filters], initializer=tf.random_normal_initializer()
            )
        else:
            kernel = tf.get_variable(
                "kernel", initializer=tf.constant(_weights[0]))

            biases = tf.get_variable(
                "biases", initializer=tf.constant(_weights[1]))
        conv = tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding)

        output = tf.add(conv, biases, name=scope.name)

    return output


def casual_conv(inputs, filters, k_size, strides, scope_name='casual_conv', _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        paddings = tf.constant([[0, 0], [k_size - 1, 0], [0, 0]])

        padded_inputs = tf.pad(
            tensor=inputs, paddings=paddings, name=scope.name + '_padding')

        output = conv1d(inputs=padded_inputs, filters=filters,
                        k_size=k_size, stride=strides, padding='VALID', _weights=_weights)

    return output


def relu(inputs, scope_name="relu"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _relu = tf.nn.relu(inputs, name=scope.name)
    return _relu


def elu(inputs, scope_name="elu"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _elu = tf.nn.elu(inputs, name=scope.name)
    return _elu


def leaky_relu(inputs, scope_name="leaky_relu"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _leaky_relu = tf.nn.leaky_relu(inputs, name=scope.name)
    return _leaky_relu


def softplus(inputs, scope_name="softplus"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _softplus = tf.nn.softplus(inputs, name=scope.name)
    return _softplus


def softsign(inputs, scope_name="softsign"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _softsign = tf.nn.softsign(inputs, name=scope.name)
    return _softsign


def sigmoid(inputs, scope_name="sigmoid"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _sigmoid = tf.nn.sigmoid(inputs, name=scope.name)
    return _sigmoid


def tanh(inputs, scope_name="tanh"):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        _tanh = tf.nn.tanh(inputs, name=scope.name)
    return _tanh


def gate(input_1, input_2, scope_name='gated_linear'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        gate_output = tf.multiply(
            input_1, sigmoid(input_2), name=scope.name + '_output')
    return gate_output


def gate_block(inputs, k_size, filters, scope_name='gate', _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channel = inputs.shape[-1]

        if(in_channel != filters):
            residual = conv1d(inputs, filters=filters, k_size=1,
                              padding='VALID', stride=1, scope_name='residual_red', _weights=_weights)
        else:
            residual = inputs

        A = casual_conv(inputs=inputs, filters=filters, k_size=k_size,
                        strides=1, scope_name='convA', _weights=_weights)

        B = casual_conv(inputs=inputs, filters=filters, k_size=k_size,
                        strides=1, scope_name='convB', _weights=_weights)

        gate_block_output = gate(A, B)

        output = tf.add(gate_block_output, residual,
                        name=scope.name + '_output')
    return output


def gate_block_b(inputs, k_size, filters, bottleneck, scope_name='gate_b', _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channel = inputs.shape[-1]

        if(in_channel != filters):
            residual = conv1d(inputs, filters=filters, k_size=1,
                              padding='VALID', stride=1, scope_name='residual_red', _weights=_weights)
        else:
            residual = inputs

        red_inputs = conv1d(inputs, filters=bottleneck,
                            k_size=1, strides=1, padding='VALID', scope_name='conv1d_red', _weights=_weights)

        A = casual_conv(inputs=red_inputs, filters=bottleneck, k_size=k_size,
                        strides=1, scope_name='convA', _weights=_weights)

        B = casual_conv(inputs=red_inputs, filters=bottleneck, k_size=k_size,
                        stride=1, scope_name='convB', _weights=_weights)

        gate_output = gate(A, B)

        gate_block_output = conv1d(
            gate_output, filters=filters, k_size=1, stride=1, padding='VALID', scope_name='conv1d_exp', _weights=_weights)

        output = tf.add(gate_block_output, residual,
                        name=scope.name + '_output')

    return output


def one_maxpool(inputs, padding='VALID', scope_name='one-pool1d'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        height, in_channel = inputs.shape[-2:]

        pool = tf.nn.pool(input=inputs, window_shape=[
                          height], pooling_type='MAX', padding=padding, strides=[1], name=scope.name)

    return pool


def maxpool1d(inputs, k_size, stride=None, padding='VALID', scope_name='pool1d'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        if stride is None:
            stride = k_size

        pool = tf.nn.pool(input=inputs, window_shape=[
                          k_size], pooling_type='MAX', padding=padding, strides=[stride], name=scope.name)

        return pool


def fully_connected(inputs, out_dim, scope_name="fc", _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        if _weights is None:
            in_dim = inputs.shape[-1]
            w = tf.get_variable(
                "weights",
                [in_dim, out_dim],
                initializer=tf.truncated_normal_initializer(),
            )

            b = tf.get_variable(
                "biases", [out_dim], initializer=tf.constant_initializer(0.0)
            )
        else:
            w = tf.get_variable(
                "weights", initializer=tf.constant(_weights[0]))

            biases = tf.get_variable(
                "biases", initializer=tf.constant(_weights[1]))

        out = tf.add(tf.matmul(inputs, w), b, name=scope.name)
    return out


def flatten(inputs, scope_name='flatten'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        feature_dim = inputs.shape[1] * inputs.shape[2]

        flatten = tf.reshape(
            inputs, shape=[-1, feature_dim], name=scope.name + '_output')

    return flatten


def concatinate(inputs, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        concat = tf.concat(inputs, 1, name=scope.name)

    return concat


def Dropout(inputs, rate, scope_name='dropout'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        dropout = tf.nn.dropout(inputs, keep_prob=1 -
                                rate, name=scope.name + '_output')
    return dropout


def l2_norm(inputs, alpha, scope_name='l2_norm'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        norm = alpha * tf.divide(inputs,
                                 tf.norm(inputs, ord='euclidean'),
                                 name=scope.name + '_output')
    return norm
