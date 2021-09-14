import tensorflow as tf
from config import cfg

from operations_nodes import (
    lerp,
    aug_seq
)

from blocks import (
    embedding,
    positional_encoding,
    seq_proc_block,
    seq2channel_attn,
    g_base_block,
    g_block,
    c_base_block,
    c_block,
    from_img,
    to_img,
    resize2d,
)


class TextEncoder:
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            seq_dim: int,
            fan_mid: int,
            num_proc_blocks: int,
            num_attn_heads: int,
            num_final_channels: int
    ):
        self.weights = []
        self._stage = 1

        self._vocab_size = vocab_size
        self._seq_len = seq_len
        self._seq_dim = seq_dim
        self._fan_mid = fan_mid
        self._num_prep_blocks = num_proc_blocks
        self._num_attn_heads = num_attn_heads
        self._num_final_channels = num_final_channels

        self._prep_blocks = []
        self._proc_blocks = []
        self._init_prep_blocks()

    def _init_prep_blocks(self):
        self._prep_blocks.append(embedding(storage=self.weights))
        self._prep_blocks.append(positional_encoding(seq_len=self._seq_len, seq_dim=self._seq_dim))
        self._prep_blocks.extend([
            seq_proc_block(
                fan_in=self._seq_dim,
                fan_mid=self._fan_mid,
                num_attn_heads=self._num_attn_heads,
                storage=self.weights,
                block_name=f"proc_block_{i}"
            )
            for i in range(self._num_prep_blocks)
        ])

        self._prep_blocks.append(
            seq2channel_attn(
                fan_in=self._seq_dim,
                fan_mid=64,
                num_channels=self._num_final_channels,
                storage=self.weights
            )
        )

    def get_graph(self):

        def forward(x: tf.Tensor) -> list:
            for block in self._prep_blocks:
                x = block(x)

            return x

        return forward


class ProgGen:
    def __init__(self, latent_dim: int, num_enc_channels: int, num_channels: int):
        # Variables Storage
        self.weights = []

        # Dimensions
        self._latent_dim = latent_dim
        self._enc_out_dim_flat = 2*num_enc_channels
        self._num_channels = num_channels

        # Growing parameters
        self._curr_stage = 1
        self._alpha = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False, name="lerp_alpha_G")
        self._new_block = None
        self._prev_out = None

        # Final model blocks
        self._init_block = g_base_block(
                enc_out_dim=self._enc_out_dim_flat,
                fan_out=self.nf(1),
                block_name=f"block_{self._curr_stage}",
                storage=self.weights,
            )
            
        self._blocks = []

        self._new_out = to_img(
            fan_in=self.nf(1),
            num_channels=self._num_channels,
            block_name=f"to_img_{self._curr_stage}",
            storage=self.weights,
        )

    def nf(self, stage):
        if stage == 1:
            return 512
        elif stage == 2:
            return 512
        elif stage == 3:
            return 512
        elif stage == 4:
            return 256
        elif stage == 5:
            return 128
        elif stage == 6:
            return 64
        else:
            return 64

    def grow(self):
        """
        Adds new blocks to a model, while maintaining last ones
        for a later smooth transition using linear interpolation.
        """
        # Append last block (if exists)
        if self._curr_stage != 1:
            self._blocks.append(self._new_block)

        last_nf = self.nf(self._curr_stage)
        self._curr_stage += 1
        self.set_alpha(0.0)
        next_nf = self.nf(self._curr_stage)

        # Create new blocks and update to_img outs
        self._new_block = g_block(
            fan_in=last_nf,
            fan_out=next_nf,
            enc_out_dim=self._enc_out_dim_flat,
            block_name=f"block_{self._curr_stage}",
            storage=self.weights,
        )

        self._prev_out = self._new_out
        self._new_out = to_img(
            fan_in=next_nf,
            num_channels=self._num_channels,
            block_name=f"to_img_{self._curr_stage}",
            storage=self.weights,
        )

        # Remove redundant weights from weight update
        vars_to_remove = [
            f"w_conv_to_img_{self._curr_stage - 2}:0",
            f"b_conv_to_img_{self._curr_stage - 2}:0",
        ]
        self.weights = [
            v for v in self.weights if v.name not in vars_to_remove
        ]

    def get_graph(self):
    
        # STAGE 1: Only basic block
        if self._curr_stage == 1:

            def forward(enc_out: tf.Tensor, z_seq: tf.Tensor, batch_size: int) -> tf.Tensor:
                enc_out = aug_seq(enc_out, z_seq)
                return self._new_out(self._init_block(enc_out, batch_size))
                
            return forward
        
        # STAGE 2: Basic and new, but no blocks in list
        if self._curr_stage == 2:

            def forward(enc_out: tf.Tensor, z_seq: tf.Tensor, batch_size: int) -> tf.Tensor:
                enc_out = aug_seq(enc_out, z_seq)

                x = self._init_block(enc_out, batch_size)

                return lerp(
                    self._new_out(self._new_block(x, enc_out)),
                    resize2d(self._prev_out(x), scale=2.0),
                    self._alpha
                )
            
            return forward

        # STAGE > 2: Basic block, blocks list, new block
        def forward(enc_out: tf.Tensor, z_seq: tf.Tensor, batch_size: int) -> tf.Tensor:
            enc_out = aug_seq(enc_out, z_seq)

            x = self._init_block(enc_out, batch_size)

            for block in self._blocks:
                x = block(x, enc_out)

            return lerp(
                self._new_out(self._new_block(x, enc_out)),
                resize2d(self._prev_out(x), scale=2.0),
                self._alpha
            )
            
        return forward

    def set_alpha(self, val: float):
        """
        Sets the alpha parameter for a linear interpolation
        between new and old block.
        """
        assert 0 <= val <= 1
        self._alpha.assign(val)


class ProgCritic:
    def __init__(self, num_channels: int, label_dim: int = 0, group_dim: int = 4):
        self.weights = []

        # Dimensions
        self._group_dim = group_dim
        self._num_channels = num_channels

        # Growing parameters
        self._curr_stage = 1
        self._alpha = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False, name="lerp_alpha_C")
        self._new_block = None
        self._old_in = None
        self._new_in = from_img(
            fan_out=self.nf(1),
            num_channels=self._num_channels,
            block_name=f"from_img_{self._curr_stage}",
            storage=self.weights
        )

        # Final model blocks
        self._blocks = [
            c_base_block(
                fan_in=self.nf(1) + cfg.ENC.NUM_FINAL_CHANNELS,
                fan_out_conv=self.nf(1),
                fan_mid=self.nf(0),
                label_dim=label_dim,
                group_dim=group_dim,
                storage=self.weights
            )
        ]

    def nf(self, stage):
        if stage == 0:
            return 128
        elif stage == 1:
            return 512
        elif stage == 2:
            return 512
        elif stage == 3:
            return 512
        elif stage == 4:
            return 256
        elif stage == 5:
            return 128
        elif stage == 6:
            return 64
        else:
            return 64

    def grow(self):
        self._curr_stage += 1
        self.set_alpha(0.0)
        next_nf = self.nf(self._curr_stage)
        last_nf = self.nf(self._curr_stage - 1)

        # Insert first block (if exists)
        if self._new_block:
            self._blocks.insert(0, self._new_block)

        # Create new block and update from_rgb outs inputs
        self._old_in = self._new_in
        self._new_in = from_img(
            fan_out=next_nf,
            num_channels=self._num_channels,
            block_name=f"from_img_{self._curr_stage}",
            storage=self.weights
        )
        self._new_block = c_block(
            fan_in=next_nf,
            fan_mid=last_nf,
            fan_out=last_nf,
            block_name=f"block_{self._curr_stage}",
            storage=self.weights
        )

        # Remove redundant weights from weight update
        vars_to_remove = [
            f"w_conv_from_img_{self._curr_stage - 2}:0",
            f"b_conv_from_img_{self._curr_stage - 2}:0",
        ]
        self.weights = [
            v for v in self.weights if v.name not in vars_to_remove
        ]

    def get_graph(self):
        if self._curr_stage == 1:

            def forward(x: tf.Tensor, enc_out: tf.Tensor) -> tf.Tensor:
                enc_out = tf.tile(enc_out[:, tf.newaxis, tf.newaxis, :], [1, 4, 4, 1])
                return self._blocks[-1](self._new_in(x), enc_out)

            return forward

        def forward(x: tf.Tensor, enc_out: tf.Tensor) -> tf.Tensor:

            enc_out = tf.tile(enc_out[:, tf.newaxis, tf.newaxis, :], [1, 4, 4, 1])
            x = lerp(self._new_block(self._new_in(x)), self._old_in(resize2d(x, scale=0.5)), self._alpha)
            for block in self._blocks[:-1]:
                x = block(x)

            return self._blocks[-1](x, enc_out)

        return forward

    def set_alpha(self, val: float):
        """
        Sets the alpha parameter for a linear interpolation
        between new and old block.
        """
        assert 0 <= val <= 1
        self._alpha.assign(val)
