import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pathlib
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, target_file: str, numerical_dims, window_size, max_replacing_rate,
                 soft_replacing_prob, uniform_replacing_prob, peak_noising_prob, length_adjusting_prob,
                 white_noising_prob, flip_replacing_interval, max_replacing_weight,
                 transform=None, n_windows=2000000):
        """
        Initialize a training dataset for data degradation and anomaly generation.

        Args:
        target_file (str): The path to the data file for training.
        numerical_dims (list): List of numerical dimensions in the data.
        window_size (int): The size of the data windows.
        max_replacing_rate (float): Maximum rate for data replacement.
        soft_replacing_prob (float): Probability of soft data replacement.
        uniform_replacing_prob (float): Probability of uniform data replacement.
        peak_noising_prob (float): Probability of adding peak noise.
        length_adjusting_prob (float): Probability of adjusting window length.
        white_noising_prob (float): Probability of adding white noise.
        flip_replacing_interval (str): Type of data flipping.

        Returns:
            None

        """
        self.n_windows = n_windows

        self.target_file = pathlib.Path(target_file)
        self.data = torch.from_numpy(np.load(self.target_file.as_posix())).to(torch.float32)
        self.n_data = self.data.shape[0]
        self.window_size = window_size

        self.numerical_dims = torch.zeros(size=(self.data.shape[-1],))
        self.numerical_dims[numerical_dims] = True
        self.numerical_dims = self.numerical_dims.to(torch.bool)
        self.categorical_dims = torch.logical_not(self.numerical_dims)
        self.n_numerical = self.numerical_dims.shape[0]
        self.n_categorical = self.categorical_dims.shape[0]

        self.max_replacing_rate = max_replacing_rate
        self.max_replacing_weight = max_replacing_weight
        self.soft_replacing_prob = soft_replacing_prob
        self.uniform_replacing_prob = uniform_replacing_prob
        self.peak_noising = peak_noising_prob
        self.length_adjusting_prob = length_adjusting_prob
        self.white_noising_prob = white_noising_prob

        self.vertical_flip = True
        self.horizontal_flip = True
        if flip_replacing_interval == 'all':
            pass
        elif flip_replacing_interval == 'vertical':
            self.horizontal_flip = False
        elif flip_replacing_interval == 'horizontal':
            self.vertical_flip = False
        elif flip_replacing_interval == 'none':
            self.vertical_flip = False
            self.horizontal_flip = False
        else:
            raise SystemError(f"Flip replacing option {flip_replacing_interval} is not supported!!!")

        self.replacing_probs = torch.Tensor([
            self.soft_replacing_prob,
            self.uniform_replacing_prob,
            self.peak_noising,
            self.length_adjusting_prob,
            self.white_noising_prob
        ])

    def __len__(self) -> int:
        """
            Get the number of windows in the dataset.

            Returns:
                int: Number of windows.

        """
        return self.n_windows

    def __getitem__(self, index: int):
        """
        Get a degraded window and corresponding labels.

        Args:
            index (int): n. of step

        Variables:
            window: data samples that will be degraded
            start_idx: start index for the degradation in the window
            length: length of the degraded sub-window
            features_to_degrade: mask to select which features will be degraded
            replacing_type: select which degradation will be applied
        Returns:
            tuple: A tuple containing a degraded window and labels (window, labels)
            window: degraded window
            labels: synthetic labels (degraded (anomaly) samples set at 1; otherwise 0)
        """

        window_idx = torch.randint(low=0, high=self.n_data - self.window_size + 1, size=(1,))
        window = self.data[window_idx: window_idx + self.window_size]
        labels = torch.zeros(size=(window.shape[0],))

        # print(f"self.data: {self.data[window_idx: window_idx + self.window_size].storage().data_ptr()}")
        # print(f"window: {window.storage().data_ptr()}")

        start_idx = torch.randint(low=0, high=self.window_size, size=(1,))
        if start_idx == self.window_size - 1:
            length = 1
        else:
            end_idx = torch.randint(low=int(start_idx + 1), high=self.window_size, size=(1,))
            length = end_idx - start_idx
        while True:
            features_to_degrade = torch.rand(size=(self.data.shape[-1],)) >= 0.5
            target_num_features = torch.logical_and(self.numerical_dims, features_to_degrade)
            target_cat_features = torch.logical_and(self.categorical_dims, features_to_degrade)

            # scelta implementativa (potrei anche imporre solo il or, ma con l'and è più semplice da scrivere)
            if torch.any(target_cat_features) and torch.any(target_num_features):
                # At least one element is True, exit the loop
                break

        replacing_type = torch.multinomial(self.replacing_probs, 1).item()
        if replacing_type == 0:
            # soft replacement
            soft_start_idx = torch.randint(low=0, high=int(self.n_data - length + 1), size=(1,))
            soft_window = self.data[soft_start_idx: soft_start_idx + length, :]

            # print(f"self.data soft: {self.data[soft_start_idx: soft_start_idx + length, :].storage().data_ptr()}")
            # print(f"soft_window: {soft_window.storage().data_ptr()}")

            # flipping
            if self.vertical_flip:
                soft_window = 1 - soft_window
            if self.horizontal_flip:
                soft_window = torch.flip(soft_window, dims=[0])

            weight = torch.rand(size=(1,)) * self.max_replacing_weight
            window[start_idx: start_idx + length, target_num_features] = (
                    weight * soft_window[:, target_num_features] +
                    (1 - weight) * soft_window[:, target_num_features]
            )

            # categorical features
            window[start_idx:start_idx + length, target_cat_features] = soft_window[:, target_cat_features]

            labels[start_idx:start_idx + length] = 1

        elif replacing_type == 1:
            # uniform replacement
            idx = torch.randint(low=0, high=self.n_data, size=(1,))
            window[start_idx:start_idx + length, features_to_degrade] = (
                self.data[idx, features_to_degrade])

            labels[start_idx:start_idx + length] = 1

        elif replacing_type == 2:
            # peak noise
            values = self.data[start_idx, target_num_features]
            positive_mask = values >= 0
            negative_mask = values < 0
            values[positive_mask] += torch.rand(size=values.shape)
            values[negative_mask] -= torch.rand(size=values.shape)

            self.data[start_idx, target_num_features] = values

            labels[start_idx] = 1

        elif replacing_type == 3:
            # length adjustment
            mask = torch.randint(low=0, high=2, size=(length,))
            for idx in range(1, length):
                if mask[idx]:
                    window[idx] = window[idx - 1]
                    labels[start_idx + idx] = 1

        elif replacing_type == 4:
            # white noise
            n_num_features = int(torch.sum(target_num_features))
            noise = torch.normal(mean=torch.zeros(size=(n_num_features,)), std=torch.eye(n_num_features))
            window[:, target_num_features] += noise

            labels[start_idx:start_idx + length] = 1

        degraded_window = window[start_idx:start_idx + length, target_num_features]
        min_value, _ = torch.min(degraded_window, dim=1, keepdim=True)
        max_value, _ = torch.max(degraded_window, dim=1, keepdim=True)
        if torch.any(min_value) < -1 or torch.any(max_value) > 1:
            # rescaling to [-1,1]
            window[start_idx:start_idx + length, target_num_features] = (
                    -1 + 2 * (degraded_window - min_value) / (max_value - min_value)
            )

        return window, labels

    def get_n_samples(self):
        """
        Get the number of data samples.

        Returns:
            int: Number of data samples.

        """
        return self.data.shape[0]

    def get_data_dim(self):
        """
        Get the data dimension.

        Returns:
            int: Data dimension.

        """
        return self.data.shape[1]


class TestDataset(Dataset):
    def __init__(self, target_data_file: str, target_labels_file: str, window_size, stride, sub_sequence_rate,
                 transform=None) -> None:
        """
        Initialize a test dataset for anomaly detection.

        Args:
            target_data_file (str): The path to the data file for testing.
            target_labels_file (str): The path to the labels file for testing.
            window_size (int): The size of the data windows.
            stride (int): The stride between windows.

        Returns:
            None

        """
        # self.target_data_file = pathlib.Path(target_data_file)
        # self.target_labels_file = pathlib.Path(target_labels_file)
        #
        # self.sub_seq_rate = sub_sequence_rate
        # self.data = torch.from_numpy(np.load(self.target_data_file.as_posix())).to(torch.float32)
        # self.labels = torch.from_numpy(np.load(self.target_labels_file.as_posix())).to(torch.int)
        #
        # self.n_data = self.data.shape[0]
        #
        # self.transform = transform
        #
        # self.window_size = window_size
        # self.stride = stride
        # self.max_overlapping_windows = int(self.window_size / self.stride)
        #
        # self.n_windows = int((self.n_data - self.window_size) / self.stride + 1)

        self.target_data_file = pathlib.Path(target_data_file)
        self.target_labels_file = pathlib.Path(target_labels_file)

        self.sub_seq_rate = sub_sequence_rate
        self.dataset = torch.from_numpy(np.load(self.target_data_file.as_posix())).to(torch.float32)
        self.dataset_labels = torch.from_numpy(np.load(self.target_labels_file.as_posix())).to(torch.int)

        self.transform = transform

        self.window_size = window_size
        self.stride = stride

        self.data = None
        self.n_data = None
        self.labels = None
        self.n_windows = None
        self.length = int(self.dataset.shape[0] * self.sub_seq_rate)
        self.set_new_test()

    def set_new_test(self):
        if self.sub_seq_rate == 1:
            start_idx = 0
        else:
            start_idx = torch.randint(low=0, high=self.dataset.shape[0] - self.length, size=(1,))
        self.data = self.dataset[start_idx:start_idx + self.length]
        self.labels = self.dataset_labels[start_idx:start_idx + self.length]

        self.n_data = self.data.shape[0]
        self.n_windows = int((self.n_data - self.window_size) / self.stride + 1)
        if self.n_windows <= 0:
            raise ValueError("ERROR: invalid sub_seq_rate")

    def __len__(self) -> int:
        """
        Get the number of windows in the dataset.

        Returns:
            int: Number of windows.

        """
        return self.n_windows

    def __getitem__(self, index: int):
        """
        Get a data sequence (of window_size length) and corresponding labels.

        Args:
            index (int): The index of the window.

        Returns:
            tuple: A tuple containing the starting index, data sequence, and labels.

        """
        # if index == 0:
        #     self.set_new_test()

        # assumo di numerare le possibili windows che si possono costruire sulla sequenza self.data;
        # quindi il dataloader, quando chiederà di ricevere una sequenza, appunto chiederà la prima, la seconda, etc...,
        # cioè chiederà la sequenza numero index
        start_idx = index * self.stride
        sequence = self.data[start_idx:start_idx + self.window_size]
        labels = self.labels[start_idx:start_idx + self.window_size]

        return start_idx, sequence, labels

    def get_n_samples(self):
        """
        Get the number of data samples.

        Returns:
            int: Number of data samples.

        """
        return self.length

    def get_data_dim(self):
        """
        Get the data dimension.

        Returns:
            int: Data dimension.

        """
        return self.dataset.shape[1]

    def get_window_size(self):
        """
        Get the window size.

        Returns:
            int: Window size.

        """
        return self.window_size

    def get_test_labels(self):
        """
        Get the test labels.

        Returns:
            Tensor: Test labels.

        """
        return self.labels


def create_train_dataloader(train_data_file: str,
                            n_windows: int,
                            numerical_dims,
                            window_size,
                            max_replacing_rate,
                            soft_replacing_prob,
                            uniform_replacing_prob,
                            peak_noising_prob,
                            length_adjusting_prob,
                            white_noising_prob,
                            flip_replacing_interval,
                            max_replacing_weight,
                            transform: transforms.Compose,
                            batch_size: int,
                            num_workers: int,
                            pin_memory: bool):
    """
    Create a data loader for the training dataset.

    Args:
        train_data_file (str): The path to the data file for training.
        n_windows (int): The number of data windows to generate.
        numerical_dims (list): List of numerical dimensions in the data.
        window_size (int): The size of the data windows.
        max_replacing_rate (float): Maximum rate for data replacement.
        soft_replacing_prob (float): Probability of soft data replacement.
        uniform_replacing_prob (float): Probability of uniform data replacement.
        peak_noising_prob (float): Probability of adding peak noise.
        length_adjusting_prob (float): Probability of adjusting window length.
        white_noising_prob (float): Probability of adding white noise.
        flip_replacing_interval (str): Type of data flipping.
        max_replacing_weight (float): Maximum weight for data replacement.
        transform (transforms.Compose): Data transformations.
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Whether to pin memory for faster transfer.

    Returns:
        DataLoader: Training data loader.

    """
    train_data = TrainDataset(target_file=train_data_file,
                              numerical_dims=numerical_dims,
                              window_size=window_size,
                              max_replacing_rate=max_replacing_rate,
                              soft_replacing_prob=soft_replacing_prob,
                              uniform_replacing_prob=uniform_replacing_prob,
                              peak_noising_prob=peak_noising_prob,
                              length_adjusting_prob=length_adjusting_prob,
                              white_noising_prob=white_noising_prob,
                              flip_replacing_interval=flip_replacing_interval,
                              max_replacing_weight=max_replacing_weight,
                              transform=transform,
                              n_windows=n_windows
                              )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=pin_memory)

    data_info = {"n_samples": train_data.get_n_samples(),
                 "data_dim": train_data.get_data_dim()}

    return train_dataloader, data_info


def create_test_dataloader(test_data_file: str,
                           test_labels_file: str,
                           window_size,
                           stride,
                           transform: transforms.Compose,
                           batch_size: int,
                           num_workers: int,
                           pin_memory: bool,
                           sub_sequence_rate: float = 1):
    """
    Create a data loader for the test dataset.

    Args:
        sub_sequence_rate (float): Subsequence rate to consider for testing of the total sequence
        test_data_file (str): The path to the data file for testing.
        test_labels_file (str): The path to the labels file for testing.
        window_size (int): The size of the data windows.
        stride (int): The stride between windows.
        transform (transforms.Compose): Data transformations.
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Whether to pin memory for faster transfer.

    Returns:
        DataLoader: Test data loader.

    """
    test_data = TestDataset(target_data_file=test_data_file,
                            target_labels_file=test_labels_file,
                            window_size=window_size,
                            stride=stride,
                            sub_sequence_rate=sub_sequence_rate,
                            transform=transform)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

    data_info = {"n_samples": test_data.get_n_samples(),
                 "data_dim": test_data.get_data_dim()}

    return test_dataloader, data_info


class TrainEpochDataset(Dataset):
    def __init__(self, target_file: str, numerical_dims, window_size, max_replacing_rate,
                 soft_replacing_prob, uniform_replacing_prob, peak_noising_prob, length_adjusting_prob,
                 white_noising_prob, flip_replacing_interval, max_replacing_weight,
                 transform=None):
        """
        Initialize a training dataset for data degradation and anomaly generation.

        Args:
        target_file (str): The path to the data file for training.
        numerical_dims (list): List of numerical dimensions in the data.
        window_size (int): The size of the data windows.
        max_replacing_rate (float): Maximum rate for data replacement.
        soft_replacing_prob (float): Probability of soft data replacement.
        uniform_replacing_prob (float): Probability of uniform data replacement.
        peak_noising_prob (float): Probability of adding peak noise.
        length_adjusting_prob (float): Probability of adjusting window length.
        white_noising_prob (float): Probability of adding white noise.
        flip_replacing_interval (str): Type of data flipping.

        Returns:
            None

        """

        self.target_file = pathlib.Path(target_file)
        self.data = torch.from_numpy(np.load(self.target_file.as_posix())).to(torch.float32)
        self.data = self.data[:2500]
        self.n_data = self.data.shape[0]
        self.window_size = window_size

        self.n_windows = self.data.shape[0] - self.window_size + 1

        self.numerical_dims = torch.zeros(size=(self.data.shape[-1],))
        self.numerical_dims[numerical_dims] = True
        self.numerical_dims = self.numerical_dims.to(torch.bool)
        self.categorical_dims = torch.logical_not(self.numerical_dims)
        self.n_numerical = self.numerical_dims.shape[0]
        self.n_categorical = self.categorical_dims.shape[0]

        self.max_replacing_rate = max_replacing_rate
        self.max_replacing_weight = max_replacing_weight
        self.soft_replacing_prob = soft_replacing_prob
        self.uniform_replacing_prob = uniform_replacing_prob
        self.peak_noising = peak_noising_prob
        self.length_adjusting_prob = length_adjusting_prob
        self.white_noising_prob = white_noising_prob

        self.vertical_flip = True
        self.horizontal_flip = True
        if flip_replacing_interval == 'all':
            pass
        elif flip_replacing_interval == 'vertical':
            self.horizontal_flip = False
        elif flip_replacing_interval == 'horizontal':
            self.vertical_flip = False
        elif flip_replacing_interval == 'none':
            self.vertical_flip = False
            self.horizontal_flip = False
        else:
            raise SystemError(f"Flip replacing option {flip_replacing_interval} is not supported!!!")

        self.replacing_probs = torch.Tensor([
            self.soft_replacing_prob,
            self.uniform_replacing_prob,
            self.peak_noising,
            self.length_adjusting_prob,
            self.white_noising_prob
        ])

    def __len__(self) -> int:
        """
            Get the number of windows in the dataset.

            Returns:
                int: Number of windows.

        """
        return self.n_windows

    def __getitem__(self, index: int):
        """
        Get a degraded window and corresponding labels.

        Args:
            index (int): window start index

        Returns:
            tuple: A tuple containing a degraded window and labels (window, labels)
            window: degraded window
            labels: synthetic labels (degraded (anomaly) samples set at 1; otherwise 0)
        """
        window = self.data[index: index + self.window_size]
        labels = torch.zeros(size=(window.shape[0],))

        # print(f"self.data: {self.data[window_idx: window_idx + self.window_size].storage().data_ptr()}")
        # print(f"window: {window.storage().data_ptr()}")

        start_idx = torch.randint(low=0, high=self.window_size, size=(1,))
        if start_idx == self.window_size - 1:
            length = 1
        else:
            end_idx = torch.randint(low=int(start_idx + 1), high=self.window_size, size=(1,))
            length = end_idx - start_idx
        while True:
            features_to_degrade = torch.rand(size=(self.data.shape[-1],)) >= 0.5
            target_num_features = torch.logical_and(self.numerical_dims, features_to_degrade)
            target_cat_features = torch.logical_and(self.categorical_dims, features_to_degrade)

            # scelta implementativa (potrei anche imporre solo il or, ma con l'and è più semplice da scrivere)
            if torch.any(target_cat_features) and torch.any(target_num_features):
                # At least one element is True, exit the loop
                break

        replacing_type = torch.multinomial(self.replacing_probs, 1).item()
        if replacing_type == 0:
            # soft replacement
            soft_start_idx = torch.randint(low=0, high=int(self.n_data - length + 1), size=(1,))
            soft_window = self.data[soft_start_idx: soft_start_idx + length, :]

            # print(f"self.data soft: {self.data[soft_start_idx: soft_start_idx + length, :].storage().data_ptr()}")
            # print(f"soft_window: {soft_window.storage().data_ptr()}")

            # flipping
            if self.vertical_flip:
                soft_window = 1 - soft_window
            if self.horizontal_flip:
                soft_window = torch.flip(soft_window, dims=[0])

            weight = torch.rand(size=(1,)) * self.max_replacing_weight
            window[start_idx: start_idx + length, target_num_features] = (
                    weight * soft_window[:, target_num_features] +
                    (1 - weight) * soft_window[:, target_num_features]
            )

            # categorical features
            window[start_idx:start_idx + length, target_cat_features] = soft_window[:, target_cat_features]

            labels[start_idx:start_idx + length] = 1

        elif replacing_type == 1:
            # uniform replacement
            idx = torch.randint(low=0, high=self.n_data, size=(1,))
            window[start_idx:start_idx + length, features_to_degrade] = (
                self.data[idx, features_to_degrade])

            labels[start_idx:start_idx + length] = 1

        elif replacing_type == 2:
            # peak noise
            values = self.data[start_idx, target_num_features]
            positive_mask = values >= 0
            negative_mask = values < 0
            values[positive_mask] += torch.rand(size=values.shape)
            values[negative_mask] -= torch.rand(size=values.shape)

            self.data[start_idx, target_num_features] = values

            labels[start_idx] = 1

        elif replacing_type == 3:
            # length adjustment
            mask = torch.randint(low=0, high=2, size=(length,))
            for idx in range(1, length):
                if mask[idx]:
                    window[idx] = window[idx - 1]
                    labels[start_idx + idx] = 1

        elif replacing_type == 4:
            # white noise
            n_num_features = int(torch.sum(target_num_features))
            noise = torch.normal(mean=torch.zeros(size=(n_num_features,)), std=torch.eye(n_num_features))
            window[:, target_num_features] += noise

            labels[start_idx:start_idx + length] = 1

        degraded_window = window[start_idx:start_idx + length, target_num_features]
        min_value, _ = torch.min(degraded_window, dim=1, keepdim=True)
        max_value, _ = torch.max(degraded_window, dim=1, keepdim=True)
        if torch.any(min_value) < -1 or torch.any(max_value) > 1:
            # rescaling to [-1,1]
            window[start_idx:start_idx + length, target_num_features] = (
                    -1 + 2 * (degraded_window - min_value) / (max_value - min_value)
            )

        return window, labels

    def get_n_samples(self):
        """
        Get the number of data samples.

        Returns:
            int: Number of data samples.

        """
        return self.data.shape[0]

    def get_data_dim(self):
        """
        Get the data dimension.

        Returns:
            int: Data dimension.

        """
        return self.data.shape[1]

    def get_window_size(self):
        """
        Get the window size.

        Returns:
            int: Window size.

        """
        return self.window_size


def create_train_epoch_dataloader(train_data_file: str,
                                  numerical_dims,
                                  window_size,
                                  max_replacing_rate,
                                  soft_replacing_prob,
                                  uniform_replacing_prob,
                                  peak_noising_prob,
                                  length_adjusting_prob,
                                  white_noising_prob,
                                  flip_replacing_interval,
                                  max_replacing_weight,
                                  transform: transforms.Compose,
                                  batch_size: int,
                                  num_workers: int,
                                  pin_memory: bool):
    """
    Create a data loader for the training dataset.

    Args:
        train_data_file (str): The path to the data file for training.
        numerical_dims (list): List of numerical dimensions in the data.
        window_size (int): The size of the data windows.
        max_replacing_rate (float): Maximum rate for data replacement.
        soft_replacing_prob (float): Probability of soft data replacement.
        uniform_replacing_prob (float): Probability of uniform data replacement.
        peak_noising_prob (float): Probability of adding peak noise.
        length_adjusting_prob (float): Probability of adjusting window length.
        white_noising_prob (float): Probability of adding white noise.
        flip_replacing_interval (str): Type of data flipping.
        max_replacing_weight (float): Maximum weight for data replacement.
        transform (transforms.Compose): Data transformations.
        batch_size (int): Batch size for the data loader.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Whether to pin memory for faster transfer.

    Returns:
        DataLoader: Training data loader.

    """
    train_data = TrainEpochDataset(target_file=train_data_file,
                                   numerical_dims=numerical_dims,
                                   window_size=window_size,
                                   max_replacing_rate=max_replacing_rate,
                                   soft_replacing_prob=soft_replacing_prob,
                                   uniform_replacing_prob=uniform_replacing_prob,
                                   peak_noising_prob=peak_noising_prob,
                                   length_adjusting_prob=length_adjusting_prob,
                                   white_noising_prob=white_noising_prob,
                                   flip_replacing_interval=flip_replacing_interval,
                                   max_replacing_weight=max_replacing_weight,
                                   transform=transform,
                                   )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)

    data_info = {"n_samples": train_data.get_n_samples(),
                 "data_dim": train_data.get_data_dim()}

    return train_dataloader, data_info
