import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

class Dataset():
    """
    Custom Dataset class for loading and preprocessing sensor data from HDF5 files.
    """
    def __init__(self, root, activities):
        """
        Initialize the dataset and load the data.

        Args:
            root (str): Path to the HDF5 file.
            activities (dict): Mapping of activity names to integer labels.
        """
        self.root = root
        self.activities = activities
        self.dataset, self.labels = self.load_file(root, activities)
        self.length = self.dataset.shape[0]

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Get a single sample and its label by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (data sample, label)
        """
        sample = self.dataset[idx, :, :]
        label = self.labels[idx]
        return sample, label

    def minmax_normalize(self):
        """
        Apply min-max normalization to each channel in the dataset (Unused).
        """
        for i in range(self.dataset.shape[1]):
            channel = self.dataset[:, i, :]
            self.dataset[:, i, :] = (channel - channel.min()) / (channel.max() - channel.min())

    def load_file(self, path, activities):
        """
        Load data and labels from an HDF5 file, filter by activity, and preprocess.

        Args:
            path (str): Path to the HDF5 file.
            activities (dict): Mapping of activity names to integer labels.

        Returns:
            tuple: (preprocessed dataset tensor, labels tensor)
        """
        with h5py.File(path, 'r') as hf:
            data = []
            labels = []
            a_keys = list(activities.keys())

            for dataset_name in hf.keys():
                dataset = hf[dataset_name]
                label = dataset.attrs.get('activity', 'No Label')
                if label not in a_keys:
                    continue
                data.append(dataset[:])
                labels.append(activities[label])

        # Only keep samples with at least min_length rows, and trim to min_length (all should be 100)
        min_length = 100
        data = [sample[:min_length, :] for sample in data if sample.shape[0] >= min_length]

        # Replace NaNs with zeros
        data = np.nan_to_num(data)

        # Stack all samples into a single numpy array
        data = np.stack(data, axis=0)
        labels = torch.tensor(labels, dtype=torch.long)

        # Transpose to shape (samples, channels, timesteps)
        dataset = np.transpose(data, (0, 2, 1))

        # Convert to torch tensor
        dataset = torch.from_numpy(dataset).float()
        return dataset, labels

if __name__ == '__main__':
    # Example usage and visualization
    dataset = Dataset('./data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()
