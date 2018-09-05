"""
From https://github.com/tsc2017/inception-score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py

Args:
    images: A numpy array with values ranging from -1 to 1 and shape in the form [N, 3, HEIGHT, WIDTH]
    where N, HEIGHT and WIDTH can be arbitrary.
    splits: The number of splits of the images, default is 10.
Returns:
    mean and standard deviation of the inception across the splits.
"""
import time
import functools
import os
import argparse
import glob
import tensorflow as tf
import numpy as np
from PIL import Image
from random import shuffle
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops

def inception_logits(images, num_splits=1):
    """Run images through Inception"""
    images=tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    logits = functional_ops.map_fn(
        fn=functools.partial(tf.contrib.gan.eval.run_inception, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

def get_inception_probs(inps, batch_size):
    preds = []
    n_batches = len(inps) // batch_size
    inception_images = tf.placeholder(tf.float32, [batch_size, 3, None, None])
    logits = inception_logits(inception_images)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size]
        pred = logits.eval({inception_images: inp})[:, :1000]
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    preds = np.exp(preds)/np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds,splits):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, batch_size=16, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.max(images[0]) <= 1)
    assert(np.min(images[0]) >= -1)

    start_time = time.time()
    session = tf.InteractiveSession()
    preds = get_inception_probs(images, batch_size)
    print('Inception Score for %i samples in %i splits'% (preds.shape[0], splits))
    mean, std = preds2score(preds,splits)
    print('Inception Score calculation time: %f s'%(time.time()-start_time))
    return mean, std
    # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits (default).


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", default='./data')
    parser.add_argument('--splits', type=int)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--ext', default='png')
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()

    if not args.splits:
        args.splits = len(os.listdir(args.data_root))

    images_path_list = glob.glob(args.data_root + '/**/*.{}'.format(args.ext), recursive=True)

    def transform(image):
        image = image / 255
        return np.asarray((image - 0.5) * 2).transpose((2, 0, 1))

    shuffle(images_path_list)
    print(images_path_list[:20])

    images = [transform(np.array(Image.open(image_path).convert('RGB'), dtype=np.float32))
              for image_path in images_path_list]

    images = np.stack(images)

    print(np.mean(images, axis=(0, 2, 3)))
    print(np.std(images, axis=(0, 2, 3)))
    print(np.max(images))
    print(np.min(images))
    print(images.shape)
    score = get_inception_score(images, args.batch_size, args.splits)
    print(score)