import os
import datetime

import tensorflow as tf

from config import cfg
from models import TextEncoder, ProgGen, ProgCritic
from losses import gen_magp_train_graph, critic_magp_train_graph
from utils import reset_optimizers, save_checkpoint, sample_z, sample_batch_img, sample_batch_cap


# SET GPU CONFIG
physical_devices = tf.config.experimental.list_physical_devices('GPU')
gpu_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# SET TENSORBOARD WRITER
log_dir = os.path.join(cfg.DIRS.TRAIN_LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
SUMMARY_WRITER = tf.summary.create_file_writer(logdir=log_dir)


# LOAD MODELS
ENC = TextEncoder(
    vocab_size=cfg.ENC.VOCAB_SIZE,
    seq_len=cfg.ENC.MAX_SEQ_LEN,
    seq_dim=cfg.ENC.SEQ_DIM,
    fan_mid=cfg.ENC.FAN_MID,
    num_proc_blocks=cfg.ENC.NUM_PROC_BLOCKS,
    num_attn_heads=cfg.ENC.NUM_ATTN_HEADS,
    num_final_channels=cfg.ENC.NUM_FINAL_CHANNELS
)

GEN = ProgGen(
    latent_dim=cfg.GEN.LATENT_DIM,
    num_enc_channels=cfg.ENC.NUM_FINAL_CHANNELS,
    num_channels=cfg.GEN.NUM_CHANNELS
)

CRITIC = ProgCritic(
    num_channels=cfg.CRITIC.NUM_CHANNELS,
    label_dim=cfg.CRITIC.LABEL_DIM,
    group_dim=cfg.CRITIC.GROUP_DIM
)


# SET OPTIMIZERS
GEN_OPT = tf.optimizers.Adam(
    learning_rate=0.001,
    beta_1=cfg.GEN.OPT_B1,
    beta_2=cfg.GEN.OPT_B2,
    epsilon=cfg.GEN.OPT_EPS
)

CRITIC_OPT = tf.optimizers.Adam(
    learning_rate=0.001,
    beta_1=cfg.CRITIC.OPT_B1,
    beta_2=cfg.CRITIC.OPT_B2,
    epsilon=cfg.CRITIC.OPT_EPS
)


def sample_training_status(res: int, step: int, g_loss: float, c_losses, g_params, c_params, e_params, g_alpha, c_alpha):
    with SUMMARY_WRITER.as_default():
        tf.summary.scalar(f'g_loss_{res}', g_loss, step=step)
        tf.summary.scalar(f'c_loss_real_{res}', c_losses[0], step=step)
        tf.summary.scalar(f'c_loss_fake_{res}', c_losses[1], step=step)
        tf.summary.scalar(f'c_loss_miss_{res}', c_losses[2], step=step)
        tf.summary.scalar(f'c_loss_total_{res}', c_losses[3], step=step)
        tf.summary.scalar(f'g_alpha_{res}', g_alpha, step=step)
        tf.summary.scalar(f'c_alpha_{res}', c_alpha, step=step)
        tf.summary.scalar(f'c_loss_magp_{res}', c_losses[4], step=step)


def sample_gen_imgs(enc, gen, epoch: int, res: int, batch_size: int = 9):
    """
    Samples images made by generator at the current training epoch.
    """
    z_seq = sample_z(batch_size=batch_size, latent_dim=cfg.GEN.LATENT_SEQ_DIM)
    cap = sample_batch_cap(batch_size=batch_size)

    fake_imgs = gen(enc(cap), z_seq, batch_size)
    fake_imgs = fake_imgs * 0.5 + 0.5

    with SUMMARY_WRITER.as_default():
        tf.summary.image(f"gen_images_res_{res}", fake_imgs, max_outputs=batch_size, step=epoch)


def train_networks(
        num_epochs: int,
        res: int,
        batch_size: int,
):
    """
    Main training loop for a single stage.
    """
    # Clear the default graph stack
    tf.compat.v1.reset_default_graph()

    # Initialize layer interpolation alpha
    alpha = 0
    alpha_step = 1 / (0.5 * 100000)
    GEN.set_alpha(alpha)
    CRITIC.set_alpha(alpha)

    # Get models' graphs
    enc = ENC.get_graph()
    gen = GEN.get_graph()
    critic = CRITIC.get_graph()

    # Get trainers
    train_gen = gen_magp_train_graph(
        enc=enc,
        gen=gen,
        critic=critic,
        enc_gen_params=GEN.weights,
        gen_opt=GEN_OPT,
        batch_size=batch_size
    )
    train_critic = critic_magp_train_graph(
        enc=enc,
        gen=gen,
        critic=critic,
        enc_critic_params=ENC.weights + CRITIC.weights,
        critic_opt=CRITIC_OPT,
        batch_size=batch_size
    )

    for epoch in range(num_epochs):
        g_loss = None

        # Set layer interpolation alpha
        if alpha < 1 < epoch:
            alpha = min(alpha + alpha_step, 1.0)
            GEN.set_alpha(alpha)
            CRITIC.set_alpha(alpha)

        # Train Generator
        if epoch > 0:
        
            z_seq = sample_z(batch_size=batch_size, latent_dim=cfg.GEN.LATENT_SEQ_DIM)
            x_cap = sample_batch_cap(batch_size=batch_size)

            g_loss = train_gen(
                x_cap=x_cap,
                z_seq=z_seq
            )

        # Train Critic (given # of iterations)
        for c_iter in range(cfg.TRAIN.CRITIC_ITERS):
        
            z_seq = sample_z(batch_size=batch_size, latent_dim=cfg.GEN.LATENT_SEQ_DIM)
            x_real, x_cap = sample_batch_img(res, batch_size)

            c_losses = train_critic(
                x_real=x_real,
                x_real_cap=x_cap,
                z_seq=z_seq
            )

        # Sample generated results
        if epoch % cfg.TRAIN.CHECKPOINT_PERIOD == 0:
            sample_gen_imgs(enc, gen, epoch, res)
            save_checkpoint(res, epoch, ENC.weights, GEN.weights, CRITIC.weights, GEN_OPT, CRITIC_OPT)  

        # Write loss logs for Tensorboard
        if g_loss and epoch % cfg.TRAIN.SAMPLE_STATUS_PERIOD == 0:
            sample_training_status(res, epoch, g_loss, c_losses, GEN.weights, CRITIC.weights, ENC.weights, GEN._alpha, CRITIC._alpha)

    sample_training_status(res, num_epochs, g_loss, c_losses, GEN.weights, CRITIC.weights, ENC.weights, GEN._alpha, CRITIC._alpha)


def main_training():
    for res in cfg.TRAIN.RES_LIST:

        batch_size = cfg.TRAIN.BATCH_SIZE_PER_RES[res]
        num_epochs = cfg.TRAIN.EPOCHS_PER_RES[res]
        res = int(res)

        # Train current stage
        train_networks(
            num_epochs=num_epochs,
            res=res,
            batch_size=batch_size,
        )

        # Grow networks
        GEN.grow()
        CRITIC.grow()

        # Reset optimizers
        reset_optimizers([GEN_OPT, CRITIC_OPT], lr=cfg.TRAIN.LR[str(res)])


if __name__ == "__main__":
    main_training()

