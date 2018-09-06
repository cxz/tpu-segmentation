import argparse
import os
import sys

import numpy as np
import pandas as pd
import cv2

from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, ElasticTransform, Compose, PadIfNeeded, RandomCrop

import tensorflow as tf

FLAGS = None


def get_split(input, fold):
    train_df = pd.read_csv(os.path.join(input, 'train.csv'))
    train_ids = train_df.id.values

    folds = pd.read_csv(os.path.join(input, 'folds.csv'))
    fold_dict = folds.set_index('id').to_dict()['fold']

    fold_ids = [train_id for train_id in train_ids if fold_dict[train_id] != fold]
    val_ids = [train_id for train_id in train_ids if fold_dict[train_id] == fold]
    return fold_ids, val_ids


def train_transform(size=128):
    return Compose([
        PadIfNeeded(min_height=size, min_width=size),
        HorizontalFlip(
            p=0.5),
        ElasticTransform(
            p=0.25,
            alpha=1,
            sigma=30,
            alpha_affine=30),
        ShiftScaleRotate(
            p=0.50,
            rotate_limit=.15,
            shift_limit=.25,
            scale_limit=.25,
            interpolation=cv2.INTER_CUBIC,
            border_mode=cv2.BORDER_REFLECT_101),
    ], p=1)


def val_transform(size=128):
    return Compose([
        PadIfNeeded(min_height=size, min_width=size)
    ], p=1)


def load_image(input, image_id, num_channels=1):
    img_mean = 0.5
    img_std = 1
    path = os.path.join(input, 'train', 'images', '%s.png' % image_id)
    img = cv2.imread(str(path))[:, :, :num_channels]
    img = img.astype(np.float32) / 255
    img -= img_mean
    img /= img_std
    return img


def load_mask(input, image_id):
    path = os.path.join(input, 'train', 'masks', '%s.png' % image_id)
    mask = cv2.imread(path, 0)
    return (mask / 255.0).astype(np.uint8)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_tfrecords(input, image_ids, mode, transform, out_prefix):
    out_fname = f"{out_prefix}.tfrecords"
    print('writing', out_fname)
    with tf.python_io.TFRecordWriter(out_fname) as writer:
        for index, image_id in enumerate(image_ids):
            image = load_image(input, image_id)
            mask = load_mask(input, image_id) if mode != 'test' else None

            if mode != 'test':
                data = {
                    'image': image,
                    'mask': mask
                }
                augmented = transform(**data)
                image = augmented['image']
                mask = augmented['mask']

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': _bytes_feature(image.tostring()),
                        'mask_raw': _bytes_feature(mask.tostring()),
            }))

            writer.write(example.SerializeToString())


def main(unused_argv):
    for fold in range(5):
        _, val_ids = get_split(FLAGS.input, fold)

        build_tfrecords(FLAGS.input, val_ids, 'val', val_transform(), f"val-fold{fold}")

        for epoch in range(FLAGS.epochs):
            build_tfrecords(FLAGS.input, val_ids, 'train', train_transform(), f"train-fold{fold}-{epoch}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      type=str,
      default='gcs://2018-tgs',
      help='Directory to read from'
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      help='Number of epochs'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)