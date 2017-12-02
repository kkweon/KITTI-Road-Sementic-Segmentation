"""
Inference model

Create a video
"""

import tensorflow as tf
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from typing import Tuple


def load_model(sess: tf.Session, builder_path: str) -> None:
    tags = ["VGG16_SEMENTIC"]
    tf.saved_model.loader.load(sess, tags, builder_path)


def get_tensors(sess: tf.Session) -> Tuple[tf.Tensor]:
    names = ["X", "keep_prob", "nn_last_layer"]

    # since get_collection returns a list
    # we want the first item of each list
    return [
        collection[0] for collection in [tf.get_collection(n) for n in names]
    ]


def build_pipeline(shape, get_prediction):
    def pipeline(image):
        assert len(image.shape) == 3, "image is not 3D: {}".format(image.shape)
        if image.shape != shape:
            H, W, _ = shape
            image = cv2.resize(image, (W, H))
        # output.shape = (1, H, W, 2)
        output = get_prediction(image)
        # output.shape = (H, W)
        output = np.squeeze(output)[..., 1] > 0.5

        mask = np.zeros_like(image)
        mask[..., 1] = output * 255

        return cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    return pipeline


def build_predict_fn(sess, X, keep_prob, nn_last_layer):
    def get_prediction(image: np.array):
        # (H, W, C)
        assert len(image.shape) == 3
        image = np.expand_dims(image, 0)

        feed = {X: image, keep_prob: 1.0}
        return sess.run(nn_last_layer, feed_dict=feed)

    return get_prediction


def main():
    image_shape = (160, 576, 3)

    with tf.Session() as sess:
        load_model(sess, "builder")
        X, keep_prob, nn_last_layer = get_tensors(sess)

        get_prediction = build_predict_fn(sess, X, keep_prob, nn_last_layer)
        pipeline = build_pipeline(image_shape, get_prediction)

        clip = VideoFileClip("./data/challenge_video.mp4")
        output_clip = clip.fl_image(pipeline)
        output_clip.write_videofile("output.mp4", audio=False)



if __name__ == '__main__':
    main()
