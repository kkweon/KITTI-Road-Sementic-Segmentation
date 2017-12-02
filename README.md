- [KITTI Road Sementic Segmentation](#sec-1)
  - [Dataset](#sec-1-1)
  - [Method](#sec-1-2)
    - [VGG16](#sec-1-2-1)
    - [Skip Connection](#sec-1-2-2)
    - [Fully Convolution Network](#sec-1-2-3)

# KITTI Road Sementic Segmentation<a id="sec-1"></a>

The problem is to segment driving roads from others

## Dataset<a id="sec-1-1"></a>

-   [KITTI Road/Lane Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)

Sample input images

## Method<a id="sec-1-2"></a>

-   Pretrained VGG16
-   Skip Connection
-   Fully Convolution Network

### VGG16<a id="sec-1-2-1"></a>

First, I used the pre-trained VGG16 network

```python
tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
```

### Skip Connection<a id="sec-1-2-2"></a>

Skip connection greatly improves the performance of network.

First, run 1x1 convolution and then followed by `conv_transpose` layer

```python
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
```

### Fully Convolution Network<a id="sec-1-2-3"></a>

After all, we will need a fully convolution network because the label is the same size as the input image.

In this example, only 2 labels(road or not road) exist. Therefore, output shape is `(?, 160, 576, 2)`.
