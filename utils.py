import os
import json
import pickle

from typing import List, Tuple
from collections import namedtuple

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa

from config import cfg

np.random.seed(225)
AllParams = namedtuple("AllParams", ["e_weights", "g_weighs", "c_weights"])


def save_dict(file_path: str, mapping: dict):
    with open(file_path, "w") as file:
        json.dump(mapping, file)


def load_dict(file_path: str) -> dict:
    with open(file_path, "r") as file:
        mapping = json.load(file)
    return mapping


def save_pickle(file_path: str, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: str):
    with open(file_path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def reset_optimizers(optimizers: List[tf.optimizers.Optimizer], lr: float = 0.0001):
    """
    Resets all optimizers' states for network growing.
    """
    for opt in optimizers:
        # opt._lr = lr
        for v in opt.weights:
            v.assign(tf.zeros_like(v))


def load_optimizer_vars(opt: tf.optimizers.Optimizer, loaded_weights: list):
    """
    Resets all optimizers' states for network growing.
    """
    for v, vl in zip(opt.weights, loaded_weights):
        v.assign(vl)


def random_crop(img_batch: tf.Tensor, batch_dims: tuple) -> tf.Tensor:
    return tf.image.random_crop(img_batch, size=batch_dims)


def augment_batch(img_batch: tf.Tensor, batch_dims: tuple) -> tf.Tensor:
    """
    Applies random cropping and random flipping to a batch of images.
    """
    num_imgs, img_h, img_w, num_channels = batch_dims
    img_h_aug = int(img_h * cfg.TRAIN.AUG_RES_FC)

    # upscaling
    img_batch = tf.image.resize(img_batch, [img_h_aug, img_h_aug], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # random cropping
    img_batch = random_crop(img_batch, batch_dims)

    # random mirroring
    img_batch = tf.image.random_flip_left_right(img_batch)

    # angles = np.random.uniform(low=-np.pi / 35, high=np.pi / 35, size=num_imgs)
    # img_batch = tfa.image.rotate(img_batch, angles, interpolation="NEAREST", fill_mode="reflect")

    return img_batch


def sample_batch_img(res: int, batch_size: int) -> Tuple[tf.constant, tf.constant]:
    """
    Returns a batch of images and corresponding captions as tf.constants.
    """
    img_batch = np.zeros([batch_size, res, res, 3], dtype=np.float32)
    cap_batch = np.zeros([batch_size, cfg.ENC.MAX_SEQ_LEN], dtype=np.int32)
    batch_dims = (batch_size, res, res, 3)

    imgs_idxs = np.random.randint(low=0, high=cfg.TRAIN.NUM_SAMPLES, size=batch_size)
    caps_idxs = np.random.randint(low=0, high=8, size=batch_size)

    i = 0
    for img_idx, cap_idx in zip(imgs_idxs, caps_idxs):
        img_batch[i] = np.load(f"{cfg.DIRS.PROC_DATA_DIR}/{res}x{res}/{img_idx}.npy")
        cap_batch[i] = np.load(f"{cfg.DIRS.PROC_TXT_DIR}/{img_idx}.npy")[cap_idx]
        i += 1

    img_batch = tf.constant(img_batch, dtype=cfg.TRAIN.DTYPE)
    cap_batch = tf.constant(cap_batch, dtype=tf.int32)

    return augment_batch(img_batch, batch_dims), cap_batch


def sample_batch_cap(batch_size: int) -> tf.constant:
    """
    Returns a batch of randomly chosen image captions.
    """
    cap_batch = np.zeros([batch_size, cfg.ENC.MAX_SEQ_LEN], dtype=np.int32)
    imgs_idxs = np.random.randint(low=0, high=cfg.TRAIN.NUM_SAMPLES, size=batch_size)
    caps_idxs = np.random.randint(low=0, high=8, size=batch_size)

    i = 0
    for img_idx, cap_idx in zip(imgs_idxs, caps_idxs):
        cap = np.load(f"{cfg.DIRS.PROC_TXT_DIR}/{img_idx}.npy")
        cap_batch[i] = cap[cap_idx]
        i += 1

    #np.save("bird_txt_1.npy", cap_batch[32][np.newaxis, ...])
    cap_batch = np.repeat(cap_batch[32][np.newaxis, ...], repeats=batch_size, axis=0)
    return tf.constant(cap_batch, dtype=tf.int32)


def sample_z(batch_size: int, latent_dim: int):
    """
    Returns a batch of latent vectors.
    """
    z = tf.random.normal(shape=[batch_size * latent_dim], mean=0.0, stddev=0.01, dtype=cfg.TRAIN.DTYPE)
    return tf.reshape(z, shape=[batch_size, latent_dim])


def save_checkpoint(res: int, epoch: int, e_weights, g_weights, c_weights, g_opt, c_opt):
    """
    Saves all models' and optimizers' weights to a pickle files.
    """
    weights = AllParams(e_weights, g_weights, c_weights)
    save_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/param_dict_res_{res}_ep_{epoch}.pickle", weights)
    save_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/g_opt_w_res_{res}_ep_{epoch}.pickle", g_opt.weights)
    save_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/c_opt_w_res_{res}_ep_{epoch}.pickle", c_opt.weights)


def load_checkpoint(res: int, epoch: int):
    """
    Loads all models' weights from a pickle file.
    """
    params = load_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/param_dict_res_{res}_ep_{epoch}.pickle")
    g_opt = load_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/g_opt_w_res_{res}_ep_{epoch}.pickle")
    c_opt = load_pickle(f"{cfg.DIRS.CHECKPOINTS_GDRIVE_DIR}/c_opt_w_res_{res}_ep_{epoch}.pickle")
    return params, g_opt, c_opt


def make_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
