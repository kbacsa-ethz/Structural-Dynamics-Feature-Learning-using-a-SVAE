import tensorflow as tf


class AE(tf.keras.Model):
    """Autoencoder."""

    def __init__(self, input_dim, latent_dim, num_layers, seq_len, dropout):
        super(AE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(seq_len, input_dim)))
        for _ in range(num_layers):
            self.encoder.add(tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, dropout=dropout))
        self.encoder.add(tf.keras.layers.Dense(latent_dim))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(seq_len, latent_dim)))
        for _ in range(num_layers):
            self.decoder.add(tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, dropout=dropout))
        self.decoder.add(tf.keras.layers.Dense(input_dim))

        self.total_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def train_step(self, x):
        data, label = x
        with tf.GradientTape() as tape:
            latent, reconstruction = self.call(data)
            reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.total_loss_tracker.result(),
        }

    def test_step(self, x):
        data, label = x
        latent, reconstruction = self.call(data)
        reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)
        self.total_loss_tracker.update_state(reconstruction_loss)
        return {
            "reconstruction_loss": self.total_loss_tracker.result(),
        }


class AnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, parameter_name, annealing_epochs):
        super(AnnealingCallback, self).__init__()
        self.parameter_name = parameter_name
        self.annealing_epochs = annealing_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_value = self.modify_parameter(epoch)
        setattr(self.model, self.parameter_name, new_value)

    def modify_parameter(self, epoch):
        if epoch < self.annealing_epochs:
            new_value = 1 / (self.annealing_epochs - epoch)
        else:
            new_value = 1.
        return new_value


class VAE(tf.keras.Model):
    """Variational autoencoder."""
    def __init__(self, input_dim, latent_dim, num_layers, seq_len, dropout):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.kl_beta = 0.

        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(seq_len, input_dim)))
        for _ in range(num_layers):
            self.encoder.add(tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, dropout=dropout))
        self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(seq_len, latent_dim)))
        for _ in range(num_layers):
            self.decoder.add(tf.keras.layers.LSTM(self.latent_dim, return_sequences=True, dropout=dropout))
        self.decoder.add(tf.keras.layers.Dense(input_dim))

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=2)
        return mean, logvar

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return mean, logvar, reconstruction

    def train_step(self, x):
        data, label = x
        with tf.GradientTape() as tape:
            mean, logvar, reconstruction = self.call(data)
            reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)
            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.kl_beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, x):
        data, label = x
        mean, logvar, reconstruction = self.call(data)
        reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)
        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class SVAE(VAE):
    """Variational autoencoder."""
    def __init__(self, input_dim, latent_dim, num_layers, seq_len, dropout, num_classes):
        super(SVAE, self).__init__(input_dim, latent_dim, num_layers, seq_len, dropout)

        self.num_classes = num_classes

        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.InputLayer(input_shape=(seq_len, latent_dim + latent_dim)))
        self.classifier.add(tf.keras.layers.LSTM(1, return_sequences=True, dropout=dropout))
        self.classifier.add(tf.keras.layers.Flatten())
        self.classifier.add(tf.keras.layers.Dense(num_classes))

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.bce = tf.keras.losses.BinaryCrossentropy()


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.class_loss_tracker,
        ]

    def classify(self, mean, logvar):
        classifier_input = tf.concat([mean, logvar], axis=2)
        logits = self.classifier(classifier_input)
        return logits

    @tf.function
    def call(self, x):
        mean, logvar = self.encode(x)
        logits = self.classify(mean, logvar)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return mean, logvar, logits, reconstruction

    def train_step(self, x):
        data, label = x
        with tf.GradientTape() as tape:
            mean, logvar, logits, reconstruction = self.call(data)
            reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)

            # binary cross-entropy
            label_one_hot = tf.one_hot(label, self.num_classes)
            class_loss = self.bce(label_one_hot, logits)

            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.kl_beta * kl_loss + class_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }

    def test_step(self, x):
        data, label = x
        mean, logvar, logits, reconstruction = self.call(data)
        reconstruction_loss = tf.reduce_mean((data - reconstruction) ** 2)

        # binary cross-entropy
        label_one_hot = tf.one_hot(label, self.num_classes)
        class_loss = self.bce(label_one_hot, logits)

        kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.kl_beta * kl_loss + class_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.class_loss_tracker.update_state(class_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.class_loss_tracker.result(),
        }

