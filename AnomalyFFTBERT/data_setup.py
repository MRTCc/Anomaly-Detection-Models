import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pathlib
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, target_file: str, numerical_dims, window_size, max_replacing_rate,
                 soft_replacing_prob, uniform_replacing_prob, peak_noising_prob, length_adjusting_prob,
                 white_noising_prob, flip_replacing_interval, max_replacing_weight,
                 transform=None, max_steps=125000):
        self.target_file = pathlib.Path(target_file)
        self.data = torch.from_numpy(np.load(self.target_file.as_posix()))
        self.n_data = self.data.shape[0]
        self.data_dim = self.data.shape[1]
        self.window_size = window_size
        self.max_replacing_rate = max_replacing_rate
        self.max_steps = max_steps

        self.numerical_dims = numerical_dims.to(torch.bool)
        self.categorical_dims = torch.logical_not(self.numerical_dims)
        self.n_numerical = self.numerical_dims.shape[0]
        self.n_categorical = self.categorical_dims.shape[0]

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

        self.window_idxs = torch.randint(low=0, high=self.n_data - self.window_size, size=(self.max_steps,))
        self.degraded_seq_idxs = torch.randint(low=0, high=self.window_size, size=(self.max_steps,))
        self.degraded_seq_lens = torch.randint(low=0, high=int(self.window_size * self.max_replacing_rate),
                                               size=(self.max_steps,))
        self.soft_replacing_weights = torch.rand(size=(self.max_steps,)) * self.max_replacing_weight
        self.degraded_features = torch.rand(size=(self.max_steps, self.data.shape[-1])) >= 0.5

        replacing_intervals = [0., self.soft_replacing_prob]

        acc = self.soft_replacing_prob + self.uniform_replacing_prob
        replacing_intervals.append(acc)
        acc += self.peak_noising
        replacing_intervals.append(acc)
        acc += self.length_adjusting_prob
        replacing_intervals.append(acc)
        acc += self.white_noising_prob
        replacing_intervals.append(acc)

        assert acc == 1.

        self.replacing_types = torch.rand(size=(self.max_steps,))

        # TODO: ottimizzare questa parte (sistemando le maschere)
        # # soft_replacement
        self.replacing_types[(replacing_intervals[0] <= self.replacing_types) &
                             (self.replacing_types < replacing_intervals[1])] = 0
        # # uniform replacement
        self.replacing_types[(replacing_intervals[1] <= self.replacing_types) &
                             (self.replacing_types < replacing_intervals[2])] = 1
        # # peak noising
        self.replacing_types[(replacing_intervals[2] <= self.replacing_types) &
                             (self.replacing_types < replacing_intervals[3])] = 2
        # # length adjusting
        self.replacing_types[(replacing_intervals[3] <= self.replacing_types) &
                             (self.replacing_types < replacing_intervals[4])] = 3
        # # white noising
        self.replacing_types[(replacing_intervals[4] <= self.replacing_types) &
                             (self.replacing_types < replacing_intervals[5])] = 4

        self.soft_replacing_seq_idxs = torch.empty(size=(self.max_steps,))
        for idx in range(self.max_steps):
            # if replacing_intervals[0] <= self.replacing_types[idx] < replacing_intervals[1]:
            #     # soft_replacement
            #     self.replacing_types[idx] = 0
            # elif replacing_intervals[1] <= self.replacing_types[idx] < replacing_intervals[2]:
            #     # uniform replacement
            #     self.replacing_types[idx] = 1
            # elif replacing_intervals[2] <= self.replacing_types[idx] < replacing_intervals[3]:
            #     # peak noising
            #     self.replacing_types[idx] = 2
            # elif replacing_intervals[3] <= self.replacing_types[idx] < replacing_intervals[4]:
            #     # length adjusting
            #     self.replacing_types[idx] = 3
            # elif replacing_intervals[4] <= self.replacing_types[idx] <= replacing_intervals[5]:
            #     self.replacing_types[idx] = 4
            # else:
            #     raise ValueError(f"Not valid replacing value: {self.replacing_types[idx]:.5f} !!!")

            self.soft_replacing_seq_idxs[idx] = torch.randint(low=0,
                                                              high=self.window_size - self.degraded_seq_lens[idx],
                                                              size=(1,))

        self.replacing_types = self.replacing_types.to(torch.int8)
        self.soft_replacing_seq_idxs = self.soft_replacing_seq_idxs.to(torch.int)
        self.uniform_replacing_idxs = torch.randint(low=0, high=self.n_data, size=(self.max_steps,))

        self.transform = transform

    def __len__(self) -> int:
        return self.max_steps

    def __getitem__(self, index: int):
        window_idx = self.window_idxs[index]
        window = self.data[window_idx:window_idx+self.window_size]
        start = self.degraded_seq_idxs[index]
        length = self.degraded_seq_lens[index]
        degraded_features = self.degraded_features[index, :]

        if self.replacing_types[index] == 0:
            # soft_replacement
            soft_start = self.soft_replacing_seq_idxs[index]

            # numerical features
            target_features = torch.logical_and(self.numerical_dims, degraded_features)
            soft_window = window[soft_start:soft_start + length, target_features]

            # flipping
            if self.vertical_flip:
                soft_window = 1 - soft_window
            if self.horizontal_flip:
                soft_window = torch.flip(soft_window, dims=[0])

            window[start:start + length, target_features] = (
                    soft_window * self.soft_replacing_weights[index]
                    + soft_window * (1 - self.soft_replacing_weights[index]))

            # categorical features
            target_features = torch.logical_and(self.categorical_dims, degraded_features)
            window[start:start + length, target_features] = window[soft_start:soft_start + length, target_features]

        elif self.replacing_types[index] == 1:
            # uniform replacement
            window[start:start + length, degraded_features] = (
                self.data[self.uniform_replacing_idxs[index], degraded_features])

        elif self.replacing_types[index] == 2:
            # peak noising
            pass
        elif self.replacing_types[index] == 3:
            # length adjusting
            pass
        elif self.replacing_types[index] == 4:
            # white noising
            pass
        else:
            raise ValueError(f"Not valid replacing value: {self.replacing_types[index]:.5f} !!!")

        # TODO: per debuggare, sul canale numerico, fare un plot della finestra originale e
        #  un plot della finestra degradata

        # return degraded_sequence, artificial_labels

    def get_n_samples(self):
        return self.data.shape[0]

    def get_data_dim(self):
        return self.data.shape[1]


class TestDataset(Dataset):
    def __init__(self, target_data_file: str, target_labels_file: str, transform=None) -> None:
        self.target_data_file = pathlib.Path(target_data_file)
        self.target_labels_file = pathlib.Path(target_labels_file)

        self.data = torch.from_numpy(np.load(self.target_data_file.as_posix()))
        self.labels = torch.from_numpy(np.load(self.target_labels_file.as_posix()))

        self.n_data = self.data.shape[0]

        self.transform = transform

    def __len__(self) -> int:
        return self.n_data

    def __getitem__(self, index: int):
        pass
        # TODO: da completare (capire come riscrivere lo script di estimation)
        # return sequence, labels

    def get_n_samples(self):
        return self.data.shape[0]

    def get_data_dim(self):
        return self.data.shape[1]


def create_train_dataloader(train_data_file: str,
                            max_steps: int,
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
    # TODO: riscrivere la creazione di TrainDataset
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
                              max_steps=max_steps)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=pin_memory)

    data_info = {"n_samples": train_data.get_n_samples(),
                 "data_dim": train_data.get_data_dim()}

    return train_dataloader, data_info


def create_test_dataloader(test_data_file: str,
                           test_labels_file: str,
                           transform: transforms.Compose,
                           batch_size: int,
                           num_workers: int,
                           pin_memory: bool):
    test_data = TestDataset(target_data_file=test_data_file,
                            target_labels_file=test_labels_file,
                            transform=transform)

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

    data_info = {"n_samples": test_data.get_n_samples(),
                 "data_dim": test_data.get_data_dim()}

    return test_dataloader, data_info
