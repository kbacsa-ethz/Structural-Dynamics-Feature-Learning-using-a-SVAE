import torch.nn as nn
import itertools
from feature_extractors import RNNSingleOutputNoMask


# Define a class for a classifier
class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, n_classes, class_layers, seq_len):
        super(Classifier, self).__init__()

        # Encoder for feature extraction
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.seq_len = seq_len

        # Classifier network
        self.classifier = nn.Sequential(
            RNNSingleOutputNoMask(latent_dim, 1, 2, 0.2),  # RNN-based feature extractor with single output and no mask
            nn.Flatten(),  # Flatten the output
            *list(itertools.chain.from_iterable(
                [[nn.Linear(in_features=seq_len, out_features=seq_len), nn.ReLU()] for _ in range(class_layers)])),
            # Multiple hidden layers with ReLU activation
            nn.Dropout(0.5),  # Dropout layer
            nn.Linear(in_features=seq_len, out_features=n_classes)  # Output layer with n_classes classes
        )

    def forward(self, x):
        # Forward pass through the encoder and then the classifier
        return self.classifier(self.encoder(x))
