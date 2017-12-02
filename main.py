import os.path
import tensorflow as tf
import helper
import warnings
import argparse
from distutils.version import LooseVersion
from typing import List
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'
), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
    tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Set global parameters
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--logdir", type=str, default="", help="Tensorboard Summary Directory")
parser.add_argument(
    "--epoch", type=int, default=1, help="Number of epoch to train")
parser.add_argument(
    "--batch-size", type=int, default=8, help="Number of batch to train")
parser.add_argument(
    "--learning-rate", type=float, default=0.01, help="Learning Rate")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint",
    help="Checkpoint Directory")

FLAGS = parser.parse_args()


def load_vgg(sess: tf.Session, vgg_path: str) -> List[tf.Tensor]:
    """Load Pretrained VGG Model into TensorFlow.

    Args:
        sess (tf.Session): TensorFlow Session
        vgg_path (str): Path to vgg folder, containing "variables/" and "saved_model.pb"

    Returns:
        tuple: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    tensor_names = [
        vgg_input_tensor_name,
        vgg_keep_prob_tensor_name,
        vgg_layer3_out_tensor_name,
        vgg_layer4_out_tensor_name,
        vgg_layer7_out_tensor_name
    ]  # yapf: disable

    return [sess.graph.get_tensor_by_name(name) for name in tensor_names]


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """Create the layers for a fully convolutional network.

    Build skip-layers using the vgg layers.

    There will be

    - 3 Deconv(Conv Transpose) operations
    - 2 merge(add) operations


    Layer 7 -> 1x1 -> 4x4 ConvT (stride: 2)->|
    Layer 4 -> 1x1 ------------------------->| -> 4x4 Conv T (stride: 2)->|
    Layer 3 -> 1x1 ------------------------------------------------------>| -> 16x16 ConvT (stride: 8)

    Args:
        vgg_layer3_out (tf.Tensor): VGG Layer 3 output
        vgg_layer4_out (tf.Tensor): VGG Layer 4 output
        vgg_layer7_out (tf.Tensor): VGG Layer 7 output
        num_classes (int): Number of classes to classify

    Returns:
        tf.Tensor: Last layer of output
    """
    reg = 1e-7

    # Glorot (Xavier) Initialization
    initializers = {"factor": 1.0, "mode": 'FAN_AVG', "uniform": True}

    def run_conv_1x1(input_):
        """Returns 1x1 Convolution"""
        return tf.layers.conv2d(
            input_,
            num_classes,
            kernel_size=1,
            padding="same",
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                **initializers),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))

    def run_conv_transpose(input_, kernel, stride, name=None):
        """Returns Conv Transpose"""
        return tf.layers.conv2d_transpose(
            input_,
            num_classes,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                **initializers),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
            name=name)

    # layer7 -> Conv 1x1 -> ConvT
    with tf.variable_scope("layer7_deconv"):
        conv_1x1 = run_conv_1x1(vgg_layer7_out)
        output = run_conv_transpose(conv_1x1, 4, 2)

    # layer4 -> Conv 1x1 -> ConvT
    with tf.variable_scope("layer4_deconv"):
        conv_1x1 = run_conv_1x1(vgg_layer4_out)
        merged = tf.add(output, conv_1x1)
        output = run_conv_transpose(merged, 4, 2)

    # layer3 -> Conv 1x1 -> ConvT
    with tf.variable_scope("layer3_deconv"):
        conv_1x1 = run_conv_1x1(vgg_layer3_out)
        merged = tf.add(output, conv_1x1)
        output = run_conv_transpose(merged, 16, 8, name="nn_last_layer")

    return output


tests.test_layers(layers)


def optimize(nn_last_layer,
             correct_label,
             learning_rate,
             num_classes,
             write_summary=False):
    """Build the TensorFLow loss and optimizer operations.

    Args:
        nn_last_layer (tf.Tensor): TF Tensor of the last layer in the neural network
        correct_label (tf.placeholder): TF Placeholder for the correct label image
        learning_rate (tf.placeholder): TF Placeholder for the learning rate
        num_classes (int): Number of classes to classify

    Returns:
        Tuple: (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    xentroy_loss = tf.losses.softmax_cross_entropy(
        tf.reshape(correct_label, (-1, num_classes)), logits)

    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(
        learning_rate, epsilon=1.0).minimize(
            xentroy_loss, global_step=global_step)

    if write_summary:
        tf.summary.scalar("Cross Entropy Loss", xentroy_loss)

    return logits, train_op, xentroy_loss


tests.test_optimize(optimize)


def train_nn(sess,
             epochs,
             batch_size,
             get_batches_fn,
             train_op,
             cross_entropy_loss,
             input_image,
             correct_label,
             keep_prob,
             learning_rate,
             saver=None,
             write_summary=False):
    """Train neural network and print out the loss during training.

    Args:
        sess (tf.Session): TF Session
        epochs (int): Number of epochs
        batch_size (int): Batch size
        get_batches_fn (Callable[[int], any()]): Function to get batches of training data.  Call using get_batches_fn(batch_size)
        train_op (tf.Operation): TF Operation to train the neural network
        cross_entropy_loss (tf.Operation): TF Tensor for the amount of loss
        input_image (tf.placeholder): TF Placeholder for input images
        correct_label (tf.placeholder): TF Placeholder for label images
        keep_prob (tf.placeholder): TF Placeholder for dropout keep probability
        learning_rate (tf.placeholder): TF Placeholder for learning rate
    """

    def print_loss(epoch: int, loss: float) -> None:
        template = """
==========================================
[Epoch: {}] Loss: {}
==========================================
        """.format(epoch, loss)
        print(template)

    if write_summary:
        import datetime
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join(FLAGS.logdir,
                         datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")),
            graph=tf.get_default_graph())

    global_step = tf.train.get_or_create_global_step()

    # if there is a checkpoint
    last_checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
    if saver and last_checkpoint:
        print("Checkpoint exists...")
        saver.restore(sess, last_checkpoint)
        print("Loaded the checkpoint")
    else:
        # Initialize
        print("No checkpoint is found. Will create a new checkpoint")
        sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        for images, labels in get_batches_fn(batch_size):

            feed = {
                input_image: images,
                correct_label: labels,
                keep_prob: 0.5,
                learning_rate: FLAGS.learning_rate
            }

            ops = [train_op, global_step, cross_entropy_loss]

            if write_summary:
                ops.append(summary_op)

            output = sess.run(ops, feed_dict=feed)

            if write_summary:
                _, step, step_loss, step_summary = output
                summary_writer.add_summary(step_summary, global_step=step)
            else:
                _, step, step_loss = output

            print_loss(i + 1, step_loss)


tests.test_train_nn(train_nn)


def run():
    # TRAINING PARAMETERS
    epochs = FLAGS.epoch
    batch_size = 8
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)

        # DECLARE PLACEHOLDERS
        correct_label = tf.placeholder(
            tf.float32,
            shape=(None, *image_shape, num_classes),
            name="correct_label")
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # BUILD graph
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(
            sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        if FLAGS.logdir:
            image_summary = tf.expand_dims(
                tf.argmax(nn_last_layer, axis=-1), axis=-1) * 255
            image_summary = tf.cast(image_summary, tf.uint8)
            tf.summary.image("Output Image", image_summary)
            tf.summary.image("Input Image", image_input)

        logits, train_op, xent_loss = optimize(
            nn_last_layer,
            correct_label,
            learning_rate,
            num_classes,
            write_summary=FLAGS.logdir)

        saver = tf.train.Saver()
        builder = tf.saved_model.builder.SavedModelBuilder("builder")

        tf.add_to_collection("X", image_input)
        tf.add_to_collection("keep_prob", keep_prob)
        tf.add_to_collection("nn_last_layer", nn_last_layer)


        # Train network
        try:
            train_nn(
                sess,
                epochs,
                batch_size,
                get_batches_fn,
                train_op,
                xent_loss,
                image_input,
                correct_label,
                keep_prob,
                learning_rate,
                saver=saver,
                write_summary=FLAGS.logdir)

        except KeyboardInterrupt:
            print("\n\nCreating checkpoint")
            saver.save(sess, os.path.join(FLAGS.checkpoint, "model.ckpt"))
            print("Builder Saving...")
            builder.add_meta_graph_and_variables(sess, ["VGG16_SEMENTIC"])
            builder.save()
            print("Builder Saved...")
            # Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, data_dir, sess,
                                          image_shape, logits, keep_prob,
                                          image_input)


if __name__ == '__main__':
    run()
