import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tfrecords_loader import decode_fn, mask_data_along_second_dim
from tensorflow_autoencoders import AE, VAE, AnnealingCallback
from tflite_exporter import export_model


def main(cfg):

    train_dataset = (tf.data.TFRecordDataset([os.path.join('train_dataset', 'tf_dataset')]).map(decode_fn).map(lambda x: mask_data_along_second_dim(x)))
    val_dataset = (tf.data.TFRecordDataset([os.path.join('val_dataset', 'tf_dataset')]).map(decode_fn))
    test_dataset = (tf.data.TFRecordDataset([os.path.join('test_dataset', 'tf_dataset')]).map(decode_fn))

    # Create a TensorBoard callback
    log_dir = "logs"
    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    # define model
    if cfg.model_type == 'ae':
        model = AE(
            input_dim=cfg.in_channels,
            latent_dim=cfg.latent_dim,
            num_layers=cfg.num_layers,
            seq_len=cfg.seq_len,
            dropout=cfg.dropout,
        )

    elif cfg.model_type == 'vae':
        model = VAE(
            input_dim=cfg.in_channels,
            latent_dim=cfg.latent_dim,
            num_layers=cfg.num_layers,
            seq_len=cfg.seq_len,
            dropout=cfg.dropout,
        )
        callbacks.append(AnnealingCallback("kl_beta", cfg.annealing_epochs))
    else:
        raise NotImplementedError

    # build model
    model.build(input_shape=(1, cfg.seq_len, cfg.in_channels))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.learning_rate,
        decay_steps=300,
        decay_rate=cfg.learning_decay)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)

    # train model
    model.fit(
        train_dataset,
        epochs=cfg.n_epochs,
        shuffle=True,
        batch_size=cfg.batch_size,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    export_model(
        model.encoder,
        cfg.seq_len,
        cfg.in_channels,
        os.path.join(cfg.data_path, cfg.save_dir),
        test_dataset
    )


# Entry point of the script
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SVAE tensorflow training")

    parser.add_argument('--root-path', type=str, default='.')
    parser.add_argument('--data-path', type=str, default='.')
    parser.add_argument('--save-dir', type=str, default='export_model')

    # Dataset parameters
    parser.add_argument('--n-dof', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=300)

    # Model parameters
    parser.add_argument('--in-channels', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=5)
    parser.add_argument('--n-classes', type=int, default=6)  # Do not set this value to 1
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--class-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--extractor', type=str, default='lstm')
    parser.add_argument('--model-type', type=str, default='vae')
    parser.add_argument('--target', type=str, default='accelerations')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--annealing-epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--class-weight', type=float, default=1e1)
    parser.add_argument('--learning-decay', type=float, default=0.984)

    args = parser.parse_args()
    main(args)
