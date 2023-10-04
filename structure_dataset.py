# Import necessary libraries and modules
import os
import numpy as np
import pyarrow as pa
import lmdb
from torch.utils.data import Dataset as BaseDataset

# Ignore warnings for this module
import warnings

warnings.filterwarnings("ignore")


# Define a function to serialize an object using pyarrow
def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


# Define a custom dataset class named StructureDataset
class StructureDataset(BaseDataset):
    def __init__(self, db_path, n_dof, transforms=None):
        self.db_path = db_path

        # Open the LMDB database
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # Retrieve the length of the dataset and keys from the LMDB database
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transforms = transforms
        self.n_dof = n_dof

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            # Fetch data from LMDB for the given index
            states = np.array(pa.deserialize(txn.get('states_{}'.format(self.keys[index]).encode('ascii')))).astype(
                np.float32)
            force = pa.deserialize(txn.get('force_{}'.format(self.keys[index]).encode('ascii'))).astype(np.float32)
            label = pa.deserialize(txn.get('label_{}'.format(self.keys[index]).encode('ascii')))
            flexibility = pa.deserialize(txn.get('flexibility_{}'.format(self.keys[index]).encode('ascii')))

        # Organize fetched data into a dictionary
        output = {
            "displacements": states[:, :self.n_dof],
            "velocities": states[:, self.n_dof:2 * self.n_dof],
            "accelerations": states[:, 2 * self.n_dof:3 * self.n_dof],
            "force": force,
            "flexibility": flexibility,
            "label": label
        }

        return output

    def __len__(self):
        # Return the length of the dataset
        return len(self.keys)

    def __repr__(self):
        # Return a string representation of the dataset
        return self.__class__.__name__ + '(' + self.db_path + ')'
