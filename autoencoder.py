import torch
import torch.nn as nn
import itertools
from feature_extractors import RNNSingleOutputNoMask


# Define an Autoencoder (AE) class
class AE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, seq_len):
        super(AE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.encoder = encoder
        self.decoder = decoder

        fc_enc = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        fc_dec = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.fc_enc = fc_enc
        self.fc_dec = fc_dec

    def encode(self, inputs):
        encoded = self.encoder(inputs)
        postencoded = self.fc_enc(encoded)
        return postencoded

    def decode(self, z):
        predecoded = self.fc_dec(z)
        result = self.decoder(predecoded)
        return result

    def generate(self, x):
        raise self.forward(x)[0]

    def forward(self, inputs):
        encoded = self.encode(inputs)
        return self.decode(encoded)


# Define a Variational Autoencoder (VAE) class, extending AE
class VAE(AE):
    def __init__(self, encoder, decoder, latent_dim, seq_len):
        super(VAE, self).__init__(encoder, decoder, latent_dim, seq_len)

        fc_mu = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        fc_sig = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.fc_mu = fc_mu
        self.fc_sig = fc_sig

    def encode(self, inputs):
        encoded = self.encoder(inputs)
        mu = self.fc_mu(encoded)
        log_var = self.fc_sig(encoded)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return mu, log_var

    @staticmethod
    def reparameterize(mu, sig):
        std = torch.exp(0.5 * sig)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, batch_size, current_device, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def loss_function(recons, inputs, mu, log_var, criterion, kld_weight):
        kld_loss = kld_weight * torch.sum(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1))
        recons_loss = criterion(recons, inputs)
        return recons_loss, kld_loss


# Define a Conditional Variational Autoencoder (CVAE) class, extending VAE
class CVAE(VAE):
    def __init__(self, encoder, decoder, latent_dim, n_classes, seq_len):
        super(CVAE, self).__init__(encoder, decoder, latent_dim, seq_len)

    def decode(self, z, label):
        predecoded = self.fc_dec(z)
        predecoded = torch.cat([predecoded, label], dim=-1)
        result = self.decoder(predecoded)
        return result

    def forward(self, inputs, label):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, label), mu, log_var


# Define a Structured Variational Autoencoder (SVAE) class, extending VAE
class SVAE(VAE):
    def __init__(self, encoder, decoder, latent_dim, class_layers, n_classes, seq_len):
        super(SVAE, self).__init__(encoder, decoder, latent_dim, seq_len)

        self.classifier = nn.Sequential(
            RNNSingleOutputNoMask(2 * latent_dim, 1, 2, 0.2),
            nn.Flatten(),
            *list(itertools.chain.from_iterable(
                [[nn.Linear(in_features=seq_len, out_features=seq_len), nn.ReLU()] for _ in range(class_layers)])),
            nn.Dropout(0.5),
            nn.Linear(in_features=seq_len, out_features=n_classes)
        )

    def classify(self, x):
        return self.classifier(x)


class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, class_layers, n_classes, seq_len):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.classifier = nn.Sequential(
            RNNSingleOutputNoMask(latent_dim, 1, 2, 0.2),
            nn.Flatten(),
            *list(itertools.chain.from_iterable(
                [[nn.Linear(in_features=seq_len, out_features=seq_len), nn.ReLU()] for _ in range(class_layers)])),
            nn.Dropout(0.5),
            nn.Linear(in_features=seq_len, out_features=n_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
