# Import necessary modules and classes from other files
from feature_extractors import *
from autoencoder import *


# Define a function to create a model from its name and parameters
def model_from_name(name, seq_len, batch_size):
    # Parse parameters from the model name
    parameters = name.split('_')
    model_type = parameters[0]
    extractor_type = parameters[1]
    in_channels = int(parameters[2])
    latent_dim = int(parameters[3])
    num_layers = int(parameters[4])
    class_layers = int(parameters[5])
    dropout = float(parameters[6])
    target = parameters[7]

    if model_type == 'cvae' or model_type == 'svae' or model_type == 'classifier':
        n_classes = int(parameters[8])

    # Define the feature extractor based on the extractor_type
    if extractor_type == 'lstm':
        if model_type == 'cvae':
            encoder = LSTMSingleOutput(in_channels + n_classes, latent_dim, num_layers, dropout, seq_len, batch_size)
            decoder = LSTMSingleOutput(latent_dim + n_classes, in_channels, num_layers, dropout, seq_len, batch_size)
        else:
            encoder = LSTMSingleOutput(in_channels, latent_dim, num_layers, dropout, seq_len, batch_size)
            decoder = LSTMSingleOutput(latent_dim, in_channels, num_layers, dropout, seq_len, batch_size)
    elif extractor_type == 'gru':
        encoder = GRUSingleOutput(in_channels, latent_dim, num_layers, dropout, seq_len, batch_size)
        decoder = GRUSingleOutput(latent_dim, in_channels, num_layers, dropout, seq_len, batch_size)
    elif extractor_type == 'rnn':
        encoder = RNNSingleOutput(in_channels, latent_dim, num_layers, dropout, seq_len, batch_size)
        decoder = RNNSingleOutput(latent_dim, in_channels, num_layers, dropout, seq_len, batch_size)
    else:
        raise NotImplemented

    # Define the model based on the model_type
    if model_type == 'ae':
        from_name = AE(encoder, decoder, latent_dim, seq_len)
    elif model_type == 'vae':
        from_name = VAE(encoder, decoder, latent_dim, seq_len)
    elif model_type == 'cvae':
        from_name = CVAE(encoder, decoder, latent_dim, n_classes, seq_len)
    elif model_type == 'svae':
        from_name = SVAE(encoder, decoder, latent_dim, class_layers, n_classes, seq_len)
    elif model_type == 'classifier':
        from_name = Classifier(encoder, latent_dim, class_layers, n_classes, seq_len)
    else:
        raise NotImplemented

    # Define hyperparameters dictionary
    hyperparameters = {
        'model_type': model_type,
        'extractor_type': extractor_type,
        'in_channels': in_channels,
        'latent_dim': latent_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'target': target,
    }

    return from_name, hyperparameters


# Define a function to create a model based on input parameters
def create_model(
        model_type,
        extractor,
        in_channels,
        latent_dim,
        n_classes,
        num_layers,
        class_layers,
        dropout,
        target,
        seq_len,
        batch_size
):
    # Generate a unique model name based on the input parameters
    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        model_type,
        extractor,
        in_channels,
        latent_dim,
        num_layers,
        class_layers,
        dropout,
        target,
        n_classes
    )

    # Create the model and retrieve its hyperparameters
    model, model_hyperparameters = model_from_name(model_name, seq_len, batch_size)

    return model, model_hyperparameters
