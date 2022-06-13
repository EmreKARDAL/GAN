import tensorflow as tf


class GAN(object):
    def __init__(self, image_wsize, image_hsize, caption_size, noise_size):
        self.image_wsize = image_wsize
        self.image_hsize = image_hsize
        self.caption_size = caption_size
        self.noise_size = noise_size
        self.criterion = tf.keras.losses.BinaryCrossentropy()
        self.discriminator = self.make_discriminator()
        self.generator = self.make_generator()
        self.g_optim = tf.keras.optimizers.SGD(learning_rate=0.0001)
        self.d_optim = tf.keras.optimizers.SGD(learning_rate=0.0001)
        self.g_loss_metrics = tf.metrics.Mean(name='g_loss')
        self.d_loss_metrics = tf.metrics.Mean(name='d_loss')

    def make_discriminator(self):
        image = tf.keras.layers.Input(shape=(self.image_wsize, self.image_hsize, 3))
        label = tf.keras.layers.Input(shape=(self.caption_size,))
        X = tf.keras.layers.GaussianNoise(0.1)(image)
        X = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.GaussianNoise(0.1)(X)
        X = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.GaussianNoise(0.1)(X)
        X = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.GaussianNoise(0.1)(X)
        X = tf.keras.layers.Conv2D(512, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.GaussianNoise(0.1)(X)
        X = tf.keras.layers.Conv2D(1024, 3, strides=1, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.GaussianNoise(0.1)(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.concat([X, label], axis=1)
        X = tf.keras.layers.Dense(256)(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Dense(1)(X)
        return tf.keras.models.Model(inputs=[image, label], outputs=X)

    def make_generator(self):
        noise = tf.keras.layers.Input(shape=(self.noise_size,))
        label = tf.keras.layers.Input(shape=(self.caption_size,))
        generator_input = tf.concat([noise, label], axis=1)
        X = tf.keras.layers.Dense(4 * 4 * 1024)(generator_input)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Reshape((4, 4, 1024))(X)
        X = tf.keras.layers.Conv2DTranspose(512, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.LeakyReLU()(X)
        X = tf.keras.layers.Conv2DTranspose(3, 1, strides=1, padding='same', activation='tanh')(X)
        return tf.keras.models.Model(inputs=[noise, label], outputs=X)

    def d_loss_fn(self, real_logits, fake_logits):
        real_loss = self.criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = self.criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(self, fake_logits):
        return self.criterion(tf.ones_like(fake_logits), fake_logits)
