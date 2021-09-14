import numpy as np
import tensorflow as tf

from config import cfg

DTYPE = cfg.TRAIN.DTYPE
FORMAT = cfg.TRAIN.FORMAT


# BASIC NETWORK COMPONENTS ---
def eq_lr_init_w(shape: list, name: str = None, storage: list = None):
    fan_in = np.prod(shape[:-1])
    v_scale = np.float32(np.sqrt(2) / np.sqrt(fan_in))
    v_scale = tf.constant(v_scale, dtype=DTYPE)

    v = tf.Variable(
        initial_value=tf.random.normal(shape=shape, mean=0, stddev=1.0, dtype=DTYPE),
        shape=shape,
        name=name,
    )

    # For tf2 track variables manually
    if storage is not None:
        storage.append(v)

    return v, v_scale


def const_init_w(
        shape: list, amp: DTYPE = 0.0, name: str = "None", storage: list = None
) -> tf.Variable:
    v = tf.Variable(
        initial_value=tf.ones(shape=shape, dtype=DTYPE), shape=shape, name=name
    )

    v.assign(v * amp)

    # For tf2 track variables manually
    if storage is not None:
        storage.append(v)

    return v


def pixel_norm(x: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.math.rsqrt(
        tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon
    )


def instance_norm(x: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
    x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    x *= tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
    x = tf.cast(x, DTYPE)
    return x


def lerp(x0: tf.Tensor, x1: tf.Tensor, alpha: tf.Variable) -> tf.Tensor:
    return tf.add(x1, tf.multiply((x0 - x1), alpha))


def aug_seq(seq: tf.Tensor, z_seq: tf.Tensor) -> tf.Tensor:
    z_seq = pixel_norm(z_seq)
    return tf.concat([seq, z_seq], axis=-1)