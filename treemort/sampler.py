import h5py
import random

from torch.utils.data import Sampler


class JointSampler(Sampler):
    def __init__(self, hdf5_file_A, hdf5_file_B, keys_A, keys_B):
        self.hdf5_file_A = hdf5_file_A
        self.hdf5_file_B = hdf5_file_B
        self.keys_A = keys_A
        self.keys_B = keys_B
        self.dead_tree_indices_A = []
        self.dead_tree_indices_B = []
        self.no_dead_tree_indices_A = []
        self.no_dead_tree_indices_B = []
        
        self.min_dead_tree_indices = []
        self.min_no_dead_tree_indices = []

        # Open the HDF5 file and separate indices based on the presence of dead trees
        with h5py.File(self.hdf5_file_A, "r") as hf:
            for idx, key in enumerate(self.keys_A):
                contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
                if contains_dead_tree:
                    self.dead_tree_indices_A.append(idx)
                else:
                    self.no_dead_tree_indices_A.append(idx)

        # Open the HDF5 file and separate indices based on the presence of dead trees
        with h5py.File(self.hdf5_file_B, "r") as hf:
            for idx, key in enumerate(self.keys_B):
                contains_dead_tree = hf[key].attrs.get("contains_dead_tree", 0)
                if contains_dead_tree:
                    self.dead_tree_indices_B.append(idx)
                else:
                    self.no_dead_tree_indices_B.append(idx)

        min_number_dead = min(len(self.dead_tree_indices_A), len(self.dead_tree_indices_B))
        min_number_no_dead = min(len(self.no_dead_tree_indices_A), len(self.no_dead_tree_indices_B))

        for i in range(0, min_number_dead):
            self.min_dead_tree_indices.append({'idx_A': self.dead_tree_indices_A[i], 'idx_B': self.dead_tree_indices_B[i]})

        for i in range(0, min_number_no_dead):
            self.min_no_dead_tree_indices.append({'idx_A': self.no_dead_tree_indices_A[i], 'idx_B': self.no_dead_tree_indices_B[i]})

    def __iter__(self):
                
        dead_tree_sample = random.sample(self.min_dead_tree_indices, len(self.min_dead_tree_indices))
        no_dead_tree_sample = random.sample(self.min_no_dead_tree_indices, len(self.min_no_dead_tree_indices))
        
        # Combine and shuffle the samples
        indices = dead_tree_sample + no_dead_tree_sample
        
        random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return len(self.min_dead_tree_indices) + len(self.min_no_dead_tree_indices)
