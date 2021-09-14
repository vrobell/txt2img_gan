import numpy as np
import tensorflow as tf

from config import cfg

from operations_nodes import (
    eq_lr_init_w,
    const_init_w,
    pixel_norm,
    instance_norm,
    DTYPE,
    FORMAT,
)


def dense(
    fan_in: int,
    fan_out: int,
    add_bias: bool = True,
    storage: list = None,
    block_name: str = "",
):
    w, w_scale = eq_lr_init_w(
        shape=[fan_in, fan_out], name=f"w_dense_{block_name}", storage=storage
    )
    if add_bias:
        b = const_init_w(shape=[fan_out], name=f"b_dense_{block_name}", storage=storage)

        def process(x: tf.Tensor) -> tf.Tensor:
            return tf.add(tf.matmul(x, w*w_scale), b)

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, w*w_scale)

    return process


def conv2d(
    fan_in: int,
    f_maps: int,
    kernel_hw_dims: tuple,
    add_bias: bool = True,
    storage: list = None,
    block_name: str = "",
):
    assert len(kernel_hw_dims) == 2
    w, w_scale = eq_lr_init_w(
        shape=[kernel_hw_dims[0], kernel_hw_dims[1], fan_in, f_maps],
        name=f"w_conv_{block_name}",
        storage=storage,
    )
    if add_bias:
        b = const_init_w(
            shape=[1, 1, 1, f_maps], name=f"b_conv_{block_name}", storage=storage
        )

        def process(x: tf.Tensor) -> tf.Tensor:
            return tf.add(
                tf.nn.conv2d(
                    x, w*w_scale, strides=[1, 1, 1, 1], padding="SAME", data_format=FORMAT
                ),
                b,
            )

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
            x, w*w_scale, strides=[1, 1, 1, 1], padding="SAME", data_format=FORMAT
        )

    return process


def resize2d(x: tf.Tensor, scale: float = 2.0) -> tf.Tensor:
    assert scale > 0
    if scale == 1:
        return x

    elif scale > 1:
        # Nearest Neighbour upsampling
        scale = int(scale)
        x_shape = tf.shape(x)
        # Given that data_raw is NHWC, add dimensions after resize channels (H and W)
        x = tf.reshape(x, [-1, x_shape[1], 1, x_shape[2], 1, x_shape[3]])
        # Tile added dimensions by a scale factor to replicate H and W dims
        x = tf.tile(x, [1, 1, scale, 1, scale, 1])
        # Reshape back to original dims with H and W scaled
        x = tf.reshape(x, [-1, x_shape[1] * scale, x_shape[2] * scale, x_shape[3]])
        return x

    elif 0 < scale < 1:
        # Average pooling
        scale = int(1 / scale)
        return tf.nn.avg_pool(
            x,
            ksize=[1, scale, scale, 1],
            strides=[1, scale, scale, 1],
            padding="VALID",
            data_format=FORMAT,
        )


def up_conv2d(
    fan_in: int,
    f_maps: int,
    kernel_hw_dims: tuple,
    add_bias: bool = True,
    storage: list = None,
    block_name: str = "",
):

    assert len(kernel_hw_dims) == 2
    w, w_scale = eq_lr_init_w(
        shape=[kernel_hw_dims[0], kernel_hw_dims[1], fan_in, f_maps],
        name=f"w_up_conv_{block_name}",
        storage=None,
    )
    w_name = w.name
    w = tf.transpose(w, [0, 1, 3, 2])
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.Variable(w, trainable=True, name=w_name)

    if storage is not None:
        storage.append(w)

    if add_bias:
        b = const_init_w(
            shape=[1, 1, 1, f_maps], name=f"b_conv_{block_name}", storage=storage
        )

        def process(x: tf.Tensor) -> tf.Tensor:
            out_shape = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, f_maps]

            return tf.add(
                tf.nn.conv2d_transpose(
                    x, w*w_scale, out_shape, strides=[1, 2, 2, 1], padding="SAME", data_format=FORMAT
                ),
                b,
            )

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        out_shape = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, f_maps]

        return tf.nn.conv2d_transpose(
                    x, w*w_scale, out_shape, strides=[1, 2, 2, 1], padding="SAME", data_format=FORMAT
                )

    return process


def down_conv2d(
    fan_in: int,
    f_maps: int,
    kernel_hw_dims: tuple,
    add_bias: bool = True,
    storage: list = None,
    block_name: str = "",
):

    assert len(kernel_hw_dims) == 2
    w, w_scale = eq_lr_init_w(
        shape=[kernel_hw_dims[0], kernel_hw_dims[1], fan_in, f_maps],
        name=f"w_down_conv_{block_name}",
        storage=None,
    )
    w_name = w.name
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.Variable(w, trainable=True, name=w_name)

    if storage is not None:
        storage.append(w)

    if add_bias:
        b = const_init_w(
            shape=[1, 1, 1, f_maps], name=f"b_conv_{block_name}", storage=storage
        )

        def process(x: tf.Tensor) -> tf.Tensor:

            return tf.add(
                tf.nn.conv2d(
                    x, w*w_scale, strides=[1, 2, 2, 1], padding="SAME", data_format=FORMAT
                ),
                b,
            )

        return process

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(
                    x, w*w_scale, strides=[1, 2, 2, 1], padding="SAME", data_format=FORMAT
                )

    return process


# STD DEV LAYER TAKEN FROM STYLE GAN REPO =================================================
def std_dev_layer(x: tf.Tensor, group_size: int = 4):
    group_size = tf.minimum(group_size, tf.shape(x)[0])
    x_dim = tf.shape(x)
    # [GMHWnc] Split minibatch into M groups of size G
    y = tf.reshape(x, [group_size, -1, x_dim[1], x_dim[2], 1, x_dim[3]])
    # [GMHWnc] Subtract mean over group
    y -= tf.reduce_mean(y, axis=0, keepdims=True)
    # [MHWnc]  Calc variance over group
    y = tf.reduce_mean(tf.square(y), axis=0)
    # [MHWnc]  Calc stddev over group
    y = tf.sqrt(y + 1e-8)
    # [M11n1]  Take average over fmaps and pixels
    y = tf.reduce_mean(y, axis=[1, 2, 4], keepdims=True)
    # [M11n]
    y = tf.reduce_mean(y, axis=[4])
    # [NHW1]  Replicate over group and pixels
    y = tf.tile(y, [group_size, x_dim[1], x_dim[2], 1])
    return tf.concat([x, y], axis=3)
# =========================================================================================


# GAN'S BLOCKS ---
def to_img(
    fan_in: int, num_channels: int, block_name: str = "to_img", storage: list = None
):
    conv2d_node = conv2d(
        fan_in=fan_in,
        f_maps=num_channels,
        kernel_hw_dims=(1, 1),
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        return conv2d_node(x)

    return process


def from_img(
    fan_out: int, num_channels: int, block_name: str = "from_img", storage: list = None
):
    conv2d_node = conv2d(
        fan_in=num_channels,
        f_maps=fan_out,
        kernel_hw_dims=(1, 1),
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        return conv2d_node(x)

    return process


def affine(enc_out_dim: int, fan_out: int, block_name: str = "affine", storage: list = None):
    gamma_dense_1 = dense(fan_in=enc_out_dim, fan_out=256, add_bias=True, storage=storage, block_name=f"aff_gm_1_{block_name}")
    gamma_dense_2 = dense(fan_in=256, fan_out=fan_out, add_bias=True, storage=storage, block_name=f"aff_gm_2_{block_name}")

    beta_dense_1 = dense(fan_in=enc_out_dim, fan_out=256, add_bias=True, storage=storage, block_name=f"aff_bt_1_{block_name}")
    beta_dense_2 = dense(fan_in=256, fan_out=fan_out, add_bias=True, storage=storage, block_name=f"aff_bt_2_{block_name}")

    def process(x: tf.Tensor, enc_out: tf.Tensor) -> tf.Tensor:
        gamma = gamma_dense_2(tf.nn.relu(gamma_dense_1(enc_out)))
        gamma = tf.reshape(gamma, [tf.shape(gamma)[0], 1, 1, tf.shape(gamma)[1]])
        beta = beta_dense_2(tf.nn.relu(beta_dense_1(enc_out)))
        beta = tf.reshape(beta, [tf.shape(beta)[0], 1, 1, tf.shape(beta)[1]])

        return gamma * x + beta

    return process


def g_base_block(
    enc_out_dim: int, fan_out: int, block_name: str = "base_block", storage: list = None
):
    init_x = const_init_w(shape=[1, 4, 4, 512], amp=1.0, name="init_w", storage=storage)

    affine_node_0 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_0"
    )
    affine_node_1 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_1"
    )
    conv2d_node = conv2d(
        fan_in=512,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )

    def process(enc_out_flat: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.tile(init_x, [batch_size, 1, 1, 1])
        x = instance_norm(tf.nn.leaky_relu(x, alpha=0.2))
        x = affine_node_0(x, enc_out_flat)

        # Conv
        return affine_node_1(instance_norm(tf.nn.leaky_relu(conv2d_node(x), alpha=0.2)), enc_out_flat)

    return process


def c_base_block(
    fan_in: int,
    fan_out_conv: int,
    fan_mid: int,
    label_dim: int = 0,
    group_dim: int = 4,
    block_name: str = "base_block",
    storage: list = None,
):
    conv_node = conv2d(
        fan_in=fan_in + 1,
        f_maps=fan_out_conv,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )
    dense_node = dense(
        fan_in=16 * fan_out_conv,
        fan_out=fan_mid,
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )
    critic_node = dense(
        fan_in=fan_mid,
        fan_out=1 + label_dim,
        add_bias=True,
        storage=storage,
        block_name=block_name + "_c",
    )

    def process(x: tf.Tensor, enc_out: tf.Tensor) -> tf.Tensor:
        # Conv
        x = std_dev_layer(x, group_dim)
        x = tf.nn.leaky_relu(conv_node(tf.concat([x, enc_out], axis=-1)), alpha=0.2)

        # Critic
        x = tf.reshape(x, shape=[-1, 16 * fan_out_conv])
        x = tf.nn.leaky_relu(dense_node(x), alpha=0.2)
        return critic_node(x)

    return process


def s_base_block(
    embed_dim: int,
    fan_in: int,
    fan_out_conv: int,
    fan_mid: int,
    block_name: str = "base_block",
    storage: list = None,
):
    conv_node = conv2d(
        fan_in=fan_in,
        f_maps=fan_out_conv,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )
    dense_node = dense(
        fan_in=16 * fan_out_conv,
        fan_out=fan_mid,
        add_bias=True,
        storage=storage,
        block_name=block_name,
    )
    embed_node = dense(
        fan_in=fan_mid,
        fan_out=embed_dim,
        add_bias=True,
        storage=storage,
        block_name=block_name + "_s",
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        # Conv
        x = tf.nn.leaky_relu(conv_node(x), alpha=0.2)

        # Critic
        x = tf.reshape(x, shape=[-1, 16 * fan_out_conv])
        x = tf.nn.leaky_relu(dense_node(x), alpha=0.2)
        return embed_node(x)

    return process


def g_block(
    fan_in: int, fan_out: int, enc_out_dim: int, block_name: str = "block", storage: list = None
):
    affine_node_0 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_0"
    )
    affine_node_1 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_1"
    )
    conv2d_node_0 = up_conv2d(
        fan_in=fan_in,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name + "_0",
    )
    affine_node_2 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_2"
    )
    affine_node_3 = affine(
        enc_out_dim=enc_out_dim,
        fan_out=fan_out,
        storage=storage,
        block_name=block_name + "_3"
    )
    conv2d_node_1 = conv2d(
        fan_in=fan_out,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name + "_1",
    )

    def process(x: tf.Tensor, enc_out_flat: tf.Tensor) -> tf.Tensor:
        # Conv 0
        x = instance_norm(tf.nn.leaky_relu(conv2d_node_0(x), alpha=0.2))
        
        # Affine 0
        x = tf.nn.leaky_relu(affine_node_0(x, enc_out_flat), alpha=0.2)
        x = tf.nn.leaky_relu(affine_node_1(x, enc_out_flat), alpha=0.2)

        # Conv 1
        x = instance_norm(tf.nn.leaky_relu(conv2d_node_1(x), alpha=0.2))

        # Affine 1
        x = tf.nn.leaky_relu(affine_node_2(x, enc_out_flat), alpha=0.2)
        x = tf.nn.leaky_relu(affine_node_3(x, enc_out_flat), alpha=0.2)

        return x

    return process


def c_block(
    fan_in: int,
    fan_mid: int,
    fan_out: int,
    storage: list = None,
    block_name: str = "block",
):
    conv2d_node_0 = conv2d(
        fan_in=fan_in,
        f_maps=fan_out,   # fan_mid
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name + "_0",
    )
    conv2d_node_1 = down_conv2d(
        fan_in=fan_out,
        f_maps=fan_out,
        kernel_hw_dims=(3, 3),
        add_bias=True,
        storage=storage,
        block_name=block_name + "_1",
    )

    def process(x: tf.Tensor) -> tf.Tensor:
        # Conv 0
        x = tf.nn.leaky_relu(conv2d_node_0(x), alpha=0.2)

        # Down Conv 1
        return tf.nn.leaky_relu(conv2d_node_1(x), alpha=0.2)

    return process


# TEXT PREPROCESSING BLOCKS ---
def embedding(storage: list):
    embed_mtrx = tf.Variable(np.load(f"{cfg.DIRS.PROC_DATA_DIR}/embed_mtrx_glove.npy"), name="embed", trainable=False)
    storage.append(embed_mtrx)

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.nn.embedding_lookup(embed_mtrx, x)

    return process


def positional_encoding(seq_len: int, seq_dim: int):
    # Compute constant positional encoding
    pos = np.arange(seq_len)[:, np.newaxis]
    idxs = np.arange(seq_dim)[np.newaxis, :]
    angles = 1.0 / np.power(10000.0, ((2*(idxs//2)) / seq_dim))
    pos_enc = pos * angles

    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    pos_enc = tf.cast(pos_enc[np.newaxis, ...], dtype=DTYPE)

    def process(x: tf.Tensor) -> tf.Tensor:
        return tf.add(x, pos_enc)

    return process


def self_attention():
    def process(q: tf.Tensor, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        num = tf.matmul(q, k, transpose_b=True)
        denom = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype=DTYPE))
        attn_arg = num / denom
        return tf.matmul(tf.nn.softmax(attn_arg, axis=-1), v)

    return process


def multihead_attention(fan_in: int, num_heads: int, storage: list, block_name: str):
    assert fan_in % num_heads == 0
    split_dim = fan_in // num_heads

    q_dense = dense(fan_in=fan_in, fan_out=fan_in, add_bias=True, storage=storage, block_name=f'q_mh_attn_{block_name}')
    k_dense = dense(fan_in=fan_in, fan_out=fan_in, add_bias=True, storage=storage, block_name=f'k_mh_attn_{block_name}')
    v_dense = dense(fan_in=fan_in, fan_out=fan_in, add_bias=True, storage=storage, block_name=f'v_mh_attn_{block_name}')
    f_dense = dense(fan_in=fan_in, fan_out=fan_in, add_bias=True, storage=storage, block_name=f'f_mh_attn_{block_name}')

    attention_mechanism = self_attention()

    def _split_tensor(x: tf.Tensor, batch_size: int) -> tf.Tensor:
        shape = [batch_size, -1, num_heads, split_dim]
        x = tf.reshape(x, shape=shape)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def process(x: tf.Tensor):
        batch_size = int(tf.shape(x)[0])

        q = _split_tensor(q_dense(x), batch_size)
        k = _split_tensor(k_dense(x), batch_size)
        v = _split_tensor(v_dense(x), batch_size)

        attn = attention_mechanism(q, k, v)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, shape=[batch_size, -1, fan_in])
        return f_dense(attn)

    return process


def seq_proc_block(fan_in: int, fan_mid: int, num_attn_heads: int, storage: list, block_name: str):
    mh_attn = multihead_attention(
        fan_in=fan_in, num_heads=num_attn_heads, storage=storage, block_name=block_name
    )
    dense1 = dense(fan_in=fan_in, fan_out=fan_mid, add_bias=True, storage=storage, block_name=f"1_{block_name}")
    dense2 = dense(fan_in=fan_mid, fan_out=fan_in, add_bias=True, storage=storage, block_name=f"2_{block_name}")

    def process(x: tf.Tensor):
        x = pixel_norm(mh_attn(x) + x)

        x_proc = tf.nn.leaky_relu(dense1(x), alpha=0.2)
        x_proc = tf.nn.leaky_relu(dense2(x_proc), alpha=0.2)
        return pixel_norm(x + x_proc)

    return process


def seq2channel_attn(fan_in: int, fan_mid: int, num_channels: int, storage: list):
    reduce_dense_1 = dense(fan_in=fan_in, fan_out=fan_mid, add_bias=True, storage=storage, block_name="reduce_1")
    reduce_dense_2 = dense(fan_in=fan_mid, fan_out=16, add_bias=True, storage=storage, block_name="reduce_2")

    attn_dense_1 = dense(fan_in=16, fan_out=32, add_bias=True, storage=storage, block_name="seq2chn_1")
    attn_dense_2 = dense(fan_in=32, fan_out=num_channels, add_bias=True, storage=storage, block_name="seq2chn_2")

    def process(x: tf.Tensor) -> tf.Tensor:
        # Reduce tensor dimensionality from [NxTxD1] to [NxTxD], where D << D1
        x = pixel_norm(tf.nn.leaky_relu(reduce_dense_1(x)))
        x = pixel_norm(tf.nn.leaky_relu(reduce_dense_2(x)))

        # Compute [NxTxC] tensor of sequence attn weights per every channel c_n
        alphas = tf.nn.softmax(attn_dense_2(tf.nn.leaky_relu(attn_dense_1(x), alpha=0.2)), axis=1)

        # Weight reduced tensor to [NxDxC] as attn weighted sum of sequences per channel
        x = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), alphas)

        # Reshape tensor fo feature maps [N, C]
        return tf.reduce_mean(x, axis=1)

    return process
