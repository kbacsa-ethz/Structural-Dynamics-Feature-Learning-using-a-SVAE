import os
import numpy as np
import tensorflow as tf
from structure_dataset import StructureDataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_dict(input_dict):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'displacements': _bytes_feature(input_dict['displacements']),
        'velocities': _bytes_feature(input_dict['velocities']),
        'accelerations': _bytes_feature(input_dict['accelerations']),
        'force': _bytes_feature(input_dict['force']),
        'flexibility': _float_feature(input_dict['flexibility']),
        'label': _int64_feature(input_dict['label']),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# Read the data back out.
def decode_fn(recorded_bytes):
    parsed_example = tf.io.parse_single_example(
        # Data
        recorded_bytes,

        # Schema
        {
            "displacements": tf.io.FixedLenFeature([], dtype=tf.string),
            "velocities": tf.io.FixedLenFeature([], dtype=tf.string),
            "accelerations": tf.io.FixedLenFeature([], dtype=tf.string),
            "force": tf.io.FixedLenFeature([], dtype=tf.string),
            "flexibility": tf.io.FixedLenFeature([], dtype=tf.float32),
            "label": tf.io.FixedLenFeature([], dtype=tf.int64),
        }
    )

    displacements = tf.io.decode_raw(parsed_example['displacements'], out_type=np.float32)
    velocities = tf.io.decode_raw(parsed_example['velocities'], out_type=np.float32)
    accelerations = tf.io.decode_raw(parsed_example['accelerations'], out_type=np.float32)
    force = tf.io.decode_raw(parsed_example['force'], out_type=np.float32)

    """
    return {
        'displacements': tf.reshape(displacements, [1, 300, 4]),
        'velocities': tf.reshape(velocities, [1, 300, 4]),
        'accelerations': tf.reshape(accelerations, [1, 300, 4]),
        'force': force,
        'flexibility': parsed_example['flexibility'],
        'label': parsed_example['label'],
    }
    """

    return tf.reshape(accelerations, [1, 300, 4]), parsed_example['label']


def mask_data_along_second_dim(y_true, label):
    mask_ratio = 0.5  # Adjust the probability as needed
    # Get the shape of the tensor
    tensor_shape = tf.shape(y_true)

    # Define the portion of the 2nd axis to be zeroed out
    portion_to_zero = tf.random.uniform(shape=[],
                                        minval=tf.cast(tf.cast(tensor_shape[1], tf.float32) * (1 - mask_ratio), tf.int32),
                                        maxval=tensor_shape[1],
                                        dtype=tf.int32
                                        )
    # Create a mask to zero out the portion
    mask = tf.concat([
        tf.ones([tensor_shape[0], tensor_shape[1] - portion_to_zero, tensor_shape[2]], dtype=y_true.dtype),
        tf.zeros([tensor_shape[0], portion_to_zero, tensor_shape[2]], dtype=y_true.dtype)
    ], axis=1)

    # Apply the mask to zero out the portion
    y_masked = y_true * mask
    return y_masked, label


if __name__ == '__main__':
    n_dof = 4
    for lmdb_path in ['train_dataset', 'val_dataset', 'test_dataset']:
        print(lmdb_path)
        lmdb_dataset = StructureDataset(lmdb_path, n_dof)
        print(len(lmdb_dataset))
        with tf.io.TFRecordWriter(os.path.join(lmdb_path, 'tf_dataset')) as file_writer:
            for i in range(len(lmdb_dataset)):
                output = lmdb_dataset[i]
                record_bytes = serialize_dict(output)
                file_writer.write(record_bytes)

    # test that data can be recovered
    dataset = tf.data.TFRecordDataset([os.path.join('train_dataset', 'tf_dataset')]).map(decode_fn)
    for raw_record in dataset.take(1):
        print(raw_record['flexibility'])
