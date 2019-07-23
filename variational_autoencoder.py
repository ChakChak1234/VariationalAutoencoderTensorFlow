from enum import Enum
import numpy as np
import tensorflow as tf


class Parameters(Enum):
    BATCH_SIZE = 100
    DATA_INPUT_SHAPE = 3072
    EPOCHS = 500
    EPSILON = 1.0
    LAYER1_DIM = 500
    LAYER2_DIM = 700
    LATENT_DIM = 50
    LEARNING_RATE = 1e-6
    ACTIVATION = 'relu'
    INITIALIZER = 'glorot_uniform'


class VAE(tf.keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        self.layer1 = Parameters.LAYER1_DIM.value
        self.layer2 = Parameters.LAYER2_DIM.value
        self.latent = Parameters.LATENT_DIM.value

        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(Parameters.DATA_INPUT_SHAPE.value,)),
            tf.keras.layers.Dense(self.layer1, kernel_initializer=Parameters.INITIALIZER.value),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(Parameters.ACTIVATION.value),

            tf.keras.layers.Dense(self.layer2, kernel_initializer=Parameters.INITIALIZER.value),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(Parameters.ACTIVATION.value),

            tf.keras.layers.Dense(self.latent + self.latent),
            tf.keras.layers.BatchNormalization(),
        ])

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.latent,)),

            tf.keras.layers.Dense(self.layer2, kernel_initializer=Parameters.INITIALIZER.value),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(Parameters.ACTIVATION.value),

            tf.keras.layers.Dense(self.layer1, kernel_initializer=Parameters.INITIALIZER.value),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(Parameters.ACTIVATION.value),

            tf.keras.layers.Dense(Parameters.DATA_INPUT_SHAPE.value + Parameters.DATA_INPUT_SHAPE.value,
                                  kernel_initializer=Parameters.INITIALIZER.value),
            tf.keras.layers.BatchNormalization(),
        ])

    @staticmethod
    def reparameterize(mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z):
        mean_out, logvar_out = tf.split(self.generative_net(z), num_or_size_splits=2, axis=1)
        return mean_out, logvar_out


def log_normal_pdf(x, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((x - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    mean_out, logvar_out = model.decode(z)
    clipped_logvar_out = tf.clip_by_value(logvar_out, -3, 3)

    logpx_z = log_normal_pdf(tf.cast(x, tf.float32), mean_out, clipped_logvar_out)
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)

    clipped_gradients = [
        None if gradient is None else tf.clip_by_value(gradient, -1., 1.)
        for gradient in gradients]
    return clipped_gradients, loss


def apply_gradients(gradients, variables):
    tf.keras.optimizers.Adam(Parameters.LEARNING_RATE.value).apply_gradients(zip(gradients, variables))


def main():
    data = np.loadtxt('most_variant.txt').T
    train_data = data[:500]
    test_data = data[500:520]

    model = VAE()
    file_test_elbo = open("elbo.txt", "a+")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(Parameters.BATCH_SIZE.value)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(Parameters.BATCH_SIZE.value)

    for epoch in range(1, 1 + Parameters.EPOCHS.value):
        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(gradients, model.trainable_variables)

        loss = tf.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        ELBO = -loss.result()
        file_test_elbo.write("{} \n".format(ELBO))
        print('Epoch: {}, Test set ELBO: {}'.format(epoch, ELBO))

    file_test_elbo.close()
    filepath = "weights_VAE.hdf5"
    model.save_weights(filepath)


if __name__ == "__main__":
    main()
