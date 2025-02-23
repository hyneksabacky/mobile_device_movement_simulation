import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

class Dataset():
    def __init__(self, root, activities):
        self.root = root
        self.activities = activities
        self.dataset, self.labels = self.load_file(root, activities)
        self.length = self.dataset.shape[0]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx, :, :]
        target = self.labels[idx]
        return step, target

    def minmax_normalize(self):
        for i in range(self.dataset.shape[1]):
            self.dataset[:, i, :] = (self.dataset[:, i, :] - self.dataset[:, i, :].min()) / (
                self.dataset[:, i, :].max() - self.dataset[:, i, :].min())

    def load_file(self, path, activities):
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

        data = np.nan_to_num(data)

        data = np.stack(data, axis=0)

        dataset = np.transpose(data, (0, 2, 1))

        dataset = torch.from_numpy(dataset).float()
        labels = torch.tensor(labels, dtype=torch.long)

        return dataset, labels


if __name__ == '__main__':
    dataset = Dataset('./data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()
