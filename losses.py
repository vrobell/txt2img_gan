import tensorflow as tf
from collections import namedtuple

from config import cfg


CLosses = namedtuple("CLosses", ["real", "fake",  "miss", "total", "gp"])


def gen_magp_train_graph(enc, gen, critic, enc_gen_params, gen_opt, batch_size):
    @tf.function
    def training_step(x_cap, z_seq):
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(enc_gen_params)

            enc_out = enc(x_cap)
            gen_loss = -tf.reduce_mean(critic(gen(enc_out, z_seq, batch_size), enc_out))

        gen_grads = tape.gradient(gen_loss, enc_gen_params)
        gen_opt.apply_gradients(
            zip(gen_grads, enc_gen_params), experimental_aggregate_gradients=False
        )

        return gen_loss
    return training_step


def ma_gp(critic, x_real, enc_out, batch_size):

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch([x_real, enc_out])
        y_pred = critic(x_real, enc_out)

    gp_grads = tape.gradient(y_pred, [x_real, enc_out])
    gp_grads = tf.concat([tf.reshape(gp_grads[0], [batch_size, -1]), tf.reshape(gp_grads[1], [batch_size, -1])], axis=1)
    gp_grads = tf.sqrt(tf.reduce_sum(gp_grads ** 2, axis=1))
    return tf.reduce_mean(gp_grads ** cfg.TRAIN.MA_GP_EXP)


def critic_magp_train_graph(enc, gen, critic, enc_critic_params, critic_opt, batch_size):
    
    @tf.function
    def training_step(x_real, x_real_cap, z_seq):

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(enc_critic_params)

            enc_out = enc(x_real_cap)
            x_fake = gen(enc_out, z_seq, batch_size)

            real_pred = critic(x_real, enc_out)
            miss_pred = critic(x_real[:(batch_size - 4)], enc_out[4:batch_size])
            fake_pred = critic(x_fake, enc_out)

            gp_grads = ma_gp(critic, x_real, enc_out, batch_size)

            c_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_pred))
            c_loss_miss = tf.reduce_mean(tf.nn.relu(1.0 + miss_pred))

            c_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_pred))
            c_loss = c_loss_real + 0.5*(c_loss_fake + c_loss_miss) + cfg.TRAIN.MA_GP_LAMBDA * gp_grads

        # Compute Critics gradients
        c_grads = tape.gradient(c_loss, enc_critic_params)

        # Apply Critics gradients
        critic_opt.apply_gradients(zip(c_grads, enc_critic_params), experimental_aggregate_gradients=False)

        return c_loss_real, c_loss_fake, c_loss_miss, c_loss, gp_grads

    return training_step


