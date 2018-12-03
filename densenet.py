import tensorflow as tf

def conv2d(input, num_filters, kernel_size, strides=1, layer_name="conv"):

    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(inputs=input, filters=num_filters, kernel_size=kernel_size, strides=strides, padding="same")

        return net


def average_pool(input, pool_size=2, strides=2, padding="same", layer_name="average_pool"):

    with tf.name_scope(layer_name):
        net = tf.layers.average_pooling2d(inputs=input, pool_size=pool_size=, strides=strides, padding=padding)

        return net


def concat(input):
    return tf.concat(input, axis=3)


def relu(x):
    return tf.nn.relu(x)


def batch_instance_norm(x, scope='batch_instance_norm'):

    with tf.variable_scope(scope):
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
        x_batch = (x - batch_mean) / (tf.sqrt(batch_sigma + eps))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + eps))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0), constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

def bottleneck_layer(x, scope):

    with tf.name_scope(scope):
        x = batch_instance_norm(x, scope=scope+"batch_instance_norm1")
        x = relu(x)
        x = conv2d(input=x, num_filters=32, kernel_size=1, strides=1, layer_name=scope+"conv1")

        x = batch_instance_norm(x, scope=scope+'batch_instance_norm1')
        x = relu(x)
        x = conv2d(input=x, num_filters=32, kernel_size=3, strides=1, layer_name=scope+"conv2")

        return x

def transition_layer(x, scope):

    with tf.name_scope(scope):
        x = batch_instance_norm(x, scope+"batch_instance_norm")
        x = relu(x)
        x = conv2d(input=x, num_filters=32, kernel_size=1, strides=1, layer_name=scope+"conv")
        x = average_pool(input=x, pool_size=2, strides=2, layer_name=scope+"average_pool")

        return x


def dense_block(input, num_layers, layer_name):

    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input)

        x = bottleneck_layer(input, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(num_layers - 1):
            x = concat(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = concat(layers_concat)

        return x

def DenseNet(input):
    x = dense_block(input=input, num_layers=6, layer_name="block1")
    x = transition_layer(x, scope="transition_layer1")

    x = dense_block(x, num_layers=12, layer_name="block2")
    x = transition_layer(x, scope="transition_layer2")

    x = dense_block(x, num_layers=24, layer_name="block3")
    x = transition_layer(x, scope="transition_layer3")

    x = dense_block(x, num_layers=16, layer_name="block4")
    x = transition_layer(x, scope="transition_layer4")

    return x
