import numpy as np
from torch.utils.data import Dataset

from numpy.random import default_rng
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

class Cifar10:
    def __init__(self, batch_size, threads, aug='none', train_count=None, num_classes=2, seed=10):
        mean, std = self._get_statistics()
        torch.manual_seed(seed)
        if aug == "cutout":
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        elif aug == "none":
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        complete_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        complete_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        labels = torch.tensor(complete_train_set.targets)
        labels_test = torch.tensor(complete_test_set.targets)
        new_labels = -torch.ones_like(labels)
        new_labels_test = -torch.ones_like(labels_test)
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []
        for i, cur_class in enumerate(torch.arange(10)[torch.randperm(10)][:num_classes]):
            indices_of_cur_class = torch.arange(50000)[labels == cur_class]
            new_labels[labels == cur_class] = i
            indices_len = len(indices_of_cur_class)
            indices_of_cur_class = indices_of_cur_class[torch.randperm(indices_len)]
            val_indices_list.append(indices_of_cur_class[:256//num_classes])
            train_indices_list.append(indices_of_cur_class[256//num_classes:256//num_classes+train_count//num_classes])

            indices_of_cur_class_test = torch.arange(10000)[labels_test == cur_class]
            new_labels_test[labels_test == cur_class] = i
            indices_len_test = len(indices_of_cur_class_test)
            indices_of_cur_class_test = indices_of_cur_class_test[torch.randperm(indices_len_test)]
            test_indices_list.append(indices_of_cur_class_test)



        complete_train_set.targets = new_labels
        complete_test_set.targets = new_labels_test
        val_indices = torch.cat(val_indices_list, dim=0)
        train_indices = torch.cat(train_indices_list, dim=0)
        test_indices = torch.cat(test_indices_list, dim=0)
        train_set = torch.utils.data.Subset(
            complete_train_set,
            train_indices
        )
        val_set = torch.utils.data.Subset(
            complete_train_set,
            val_indices
        )
        test_set = torch.utils.data.Subset(
            complete_test_set,
            test_indices
        )

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test_all_data, self.test_all_labels = zip(*[(x[None, :], y) for x, y in test_set])
        self.test_all_data = torch.cat(self.test_all_data, dim=0)
        self.test_all_labels = torch.tensor(self.test_all_labels)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class MNIST:
    def __init__(self, batch_size, threads, aug='none', train_count=None, num_classes=2, seed=10):
        mean, std = self._get_statistics()
        torch.manual_seed(seed)
        if aug == "none":
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        complete_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        complete_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        labels = complete_train_set.targets
        labels_test = complete_test_set.targets
        new_labels = -torch.ones_like(labels)
        new_labels_test = -torch.ones_like(labels_test)
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []
        for i, cur_class in enumerate(torch.arange(10)[torch.randperm(10)][:num_classes]):
            indices_of_cur_class = torch.arange(60000)[labels == cur_class]
            new_labels[labels == cur_class] = i
            indices_len = len(indices_of_cur_class)
            indices_of_cur_class = indices_of_cur_class[torch.randperm(indices_len)]
            val_indices_list.append(indices_of_cur_class[:256//num_classes])
            train_indices_list.append(indices_of_cur_class[256//num_classes:256//num_classes+train_count//num_classes])

            indices_of_cur_class_test = torch.arange(10000)[labels_test == cur_class]
            new_labels_test[labels_test == cur_class] = i
            indices_len_test = len(indices_of_cur_class_test)
            indices_of_cur_class_test = indices_of_cur_class_test[torch.randperm(indices_len_test)]
            test_indices_list.append(indices_of_cur_class_test)


        complete_train_set.targets = new_labels
        complete_test_set.targets = new_labels_test
        val_indices = torch.cat(val_indices_list, dim=0)
        train_indices = torch.cat(train_indices_list, dim=0)
        test_indices = torch.cat(test_indices_list, dim=0)
        train_set = torch.utils.data.Subset(
            complete_train_set,
            train_indices
        )
        val_set = torch.utils.data.Subset(
            complete_train_set,
            val_indices
        )
        test_set = torch.utils.data.Subset(
            complete_test_set,
            test_indices
        )

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test_all_data, self.test_all_labels = zip(*[(x[None, :], y) for x, y in test_set])
        self.test_all_data = torch.cat(self.test_all_data, dim=0)
        self.test_all_labels = torch.tensor(self.test_all_labels)

    def _get_statistics(self):
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class Cifar100:
    def __init__(self, batch_size, threads, aug):
        mean, std = self._get_statistics()

        if aug == "cutout":
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class NArmSpiral(Dataset):
    """
    `torch.utils.data.Dataset` subclass for the NArmSpiral dataset
    `NArmSpiral` can be used to provide additional control over simply loading the .csv file
    directly. This class can be pass to a `torch.utils.data.DataLoader` object to iterate
    through the dataset. It also automatically separate the dataset into two parts: the test
    dataset and the train dataset. The train dataset takes 80% of the points for itself while
    the test dataset uses the last 20%.
    Parameters
    ----------
    filename: str
        .csv file containing the dataset.
    train: bool
        Specify of the dataset should contain training data or test data.
    Attributes
    ----------
    classes : list
        List of classes inside the dataset. Classes start at 0.
    data: ndarray
        Contains the points
    """
    def __init__(self, filename, train=True):
        self._file_data = np.loadtxt(filename, delimiter=';', dtype=np.float32)

        # Empty array to store the splices for training or test
        self.data = np.empty((0, self._file_data.shape[-1]), dtype=np.float32)

        # List of classes name and count for individual classes
        self.classes, _samples = np.unique(self._file_data[:, 2], return_counts=True)

        # We assume the classes have the same amount of samples
        self._sample_count = _samples[0]

        # Split the file data into array of each classe
        split_classes = np.split(self._file_data, len(self.classes))

        # Divide the classes into 80% training samples and 20% test samples
        part = int(self._sample_count * 0.8)

        for _class in split_classes:
            if train:
                self.data = np.concatenate((self.data, _class[:part, :]))
            else:
                self.data = np.concatenate((self.data, _class[part:, :]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :2], self.data[index, 2].astype(np.int64)

class Kink(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):

        # Empty array to store the splices for training or test

        x0 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, 1, 1000)])
        y0 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, -1, 1000)])
        d0 = np.stack((x0,y0), axis=1)
        x1 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, 1, 1000)])
        y1 = np.concatenate([np.linspace(-1+margin, 0+margin, 1000), np.linspace(0+margin, -1+margin, 1000)])
        d1 = np.stack((x1, y1), axis=1)
        l0 = np.ones(2000)
        l1 = np.zeros(2000)

        self.labels = np.concatenate((l0, l1))
        self.data = np.concatenate((d0, d1))
        self.labels = default_rng(seed).permutation(self.labels)
        self.data = default_rng(seed).permutation(self.data)
        if noise:
            self.data += default_rng(seed).standard_normal(self.data.shape)
        train_sample_count = int(samples*0.7)
        test_sample_count = int(samples*0.3)
        if train:
            self.data = self.data[:train_sample_count]
            self.labels = self.labels[:train_sample_count]
        else:
            self.data = self.data[-test_sample_count:]
            self.labels = self.labels[-test_sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SemiCircle(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30):

        # Empty array to store the splices for training or test

        x0 = np.sin(np.linspace(-np.pi/2, np.pi/2, samples//2))
        y0 = np.cos(np.linspace(-np.pi/2, np.pi/2, samples//2))-margin/2
        d0 = np.stack((x0,y0), axis=1)
        x1 = np.sin(np.linspace(-np.pi/2, np.pi/2, samples//2))
        y1 = np.cos(np.linspace(-np.pi/2, np.pi/2, samples//2))+margin/2
        d1 = np.stack((x1, y1), axis=1)
        l0 = np.ones(samples//2)
        l1 = np.zeros(samples//2)

        self.labels = np.concatenate((l0, l1))
        self.data = np.concatenate((d0, d1))
        self.labels = default_rng(seed).permutation(self.labels)
        self.data = default_rng(seed).permutation(self.data)
        sample_count = int(samples*0.7)
        if train:
            self.data = self.data[:sample_count]
            self.labels = self.labels[:sample_count]
        else:
            self.data = self.data[sample_count:]
            self.labels = self.labels[sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

import numpy as np

def get_slab_data(num_slabs=3):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    slab_end = 1
    slab_begin = 1-slab_thickness
    slab_sign = -1
    slab_data = []
    slab_labels = []
    for i in range(num_slabs):
        if slab_sign == -1:
            slab_current = np.concatenate([
                np.random.uniform(-1, 0-linear_margin, size=(400, 1)), 
                np.random.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.ones(len(slab_current)))
        else:
            slab_current = np.concatenate([
                np.random.uniform(0+linear_margin, 1, size=(400, 1)), 
                np.random.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.zeros(len(slab_current)))
        slab_data.append(slab_current)
        slab_begin -= slab_thickness + complex_margin * 2
        slab_end -= slab_thickness + complex_margin * 2
        slab_sign *= -1

    slab_data = np.concatenate(slab_data, axis=0)
    slab_labels = np.concatenate(slab_labels)
    return slab_data, slab_labels
    
def get_nonlinear_data(samples, num_slabs=3):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    y_intersections = []
    y_intersection = -1+slab_thickness+complex_margin
    y_start = -1 + slab_thickness/2
    a_sign = (-1) ** num_slabs
    X = []
    for i in range(num_slabs-1):
        y_intersections.append(y_intersection)
        y_end = y_start + complex_margin * 2 + slab_thickness
        y_cur = np.linspace(y_start, y_end, 10)
        x_cur = (y_cur-y_intersection)*a_sign
        X_cur = np.concatenate([x_cur[:, None], y_cur[:, None]], axis=1)
        X.append(X_cur)
        y_intersection += complex_margin*2 + slab_thickness
        y_start = y_end
        a_sign *= -1
    X = np.concatenate(X, axis=0)
    shift = complex_margin * (2**0.5)
    X_nonlinear = np.concatenate([
        X + np.array([[shift, 0]]),
        X - np.array([[shift, 0]])], axis=0)
    Y_nonlinear = np.concatenate([
        np.zeros(len(X)),
        np.ones(len(X))
    ])

    return X_nonlinear, Y_nonlinear

def get_linear_data(samples):
    X2 = np.linspace(1, -1, samples)
    X1 = np.zeros(len(X2))

    shift = 0.1
    X_linear_c0 = np.concatenate([X1[:, None]+shift, X2[:, None]], axis=1)
    X_linear_c1 = np.concatenate([X1[:, None]-shift, X2[:, None]], axis=1)
    X_linear = np.concatenate([X_linear_c0, X_linear_c1], axis=0)
    Y_linear = np.concatenate([np.zeros(len(X1)), np.ones(len(X1))])
    return X_linear, Y_linear

class Slab(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):

        # Empty array to store the splices for training or test
        np.random.seed(seed=seed)
        self.data, self.labels = get_slab_data(num_slabs=4)
        shuffle_idx = np.random.permutation(np.arange(len(self.data)))
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        train_sample_count = int(samples*0.7)
        test_sample_count = int(samples*0.3)
        if train:
            self.data = self.data[:train_sample_count]
            self.labels = self.labels[:train_sample_count]
        else:
            self.data = self.data[-test_sample_count:]
            self.labels = self.labels[-test_sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SlabNonlinear4(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):
        np.random.seed(seed=seed)
        X, Y = get_nonlinear_data(samples=samples, num_slabs=4)
        self.data = X
        self.labels = Y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SlabLinear(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):
        np.random.seed(seed=seed)
        X, Y = get_linear_data(samples=samples)
        self.data = X
        self.labels = Y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

def majority_fn(input_seq):
    """
    Given input sequence of tokens from vocab [0, vocab_size), return the majority token.
    Returns the token that appears most frequently in the sequence.
    """
    # Count occurrences of each value
    unique_vals, counts = torch.unique(input_seq, return_counts=True)
    # Find the value with maximum count
    majority_idx = torch.argmax(counts)
    return unique_vals[majority_idx].unsqueeze(0)


def count_fn(input_seq):
    """
    Given input sequence [min_val, max_val], return counting sequence [min_val, min_val+1, ..., max_val]
    """
    min_val, max_val = input_seq[0].item(), input_seq[1].item()
    return torch.arange(min_val, max_val + 1)


def sort_fn(input_seq):
    """
    Given input sequence, return sorted sequence in ascending order
    """
    return torch.sort(input_seq)[0]


class CountSequenceDataset(Dataset):
    """
    Dataset for COUNT task: given [min_val, max_val], generate sequence from min to max.
    
    This dataset generates sequences of the form: [min_val, max_val] + [sep_token] + [min_val, min_val+1, ..., max_val]
    For next-token prediction training, where the model learns to count from min to max.
    
    Example:
    Input: [2, 5]
    Output: [2, 3, 4, 5] 
    Full sequence: [2, 5, 102, 2, 3, 4, 5]  (102 is separator '>')
    
    This dataset ensures all counting problems are unique (no duplicate [min_val, max_val] pairs).
    """
    
    def __init__(
        self,
        n_samples: int = 5000,
        min_range_size: int = 1,  # Minimum size of counting range (max - min)
        max_range_size: int = 10, # Maximum size of counting range (max - min)
        vocab_size: int = 20,
        sep_token: int = 102,
        pad_token: int = 103,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.min_range_size = min_range_size
        self.max_range_size = max_range_size
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.pad_token = pad_token
        
        # Calculate total number of unique counting problems possible
        total_unique_problems = 0
        for range_size in range(self.min_range_size, self.max_range_size + 1):
            # For each range_size, min_val can go from 0 to vocab_size - range_size - 1
            # (because max_val = min_val + range_size must be < vocab_size)
            max_possible_min = self.vocab_size - range_size
            if max_possible_min > 0:
                total_unique_problems += max_possible_min
        
        # Check if we can generate enough unique samples
        if total_unique_problems < n_samples:
            raise ValueError(
                f"Cannot generate {n_samples} unique counting problems with given constraints. "
                f"Maximum possible unique problems: {total_unique_problems}. "
                f"Consider increasing vocab_size ({vocab_size}) or reducing n_samples, "
                f"or adjusting range constraints (min_range_size={min_range_size}, max_range_size={max_range_size})."
            )
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Generate all possible unique (min_val, max_val) pairs
        all_possible_pairs = []
        for range_size in range(self.min_range_size, self.max_range_size + 1):
            max_possible_min = self.vocab_size - range_size
            for min_val in range(max_possible_min):
                max_val = min_val + range_size
                all_possible_pairs.append((min_val, max_val))
        
        # Randomly sample n_samples unique pairs without replacement
        selected_indices = torch.randperm(len(all_possible_pairs))[:n_samples]
        selected_pairs = [all_possible_pairs[i] for i in selected_indices]
        
        # Pregenerate all samples using the unique pairs
        self.sequences = []
        for min_val, max_val in selected_pairs:
            # 1) Create input sequence [min_val, max_val]
            input_seq = torch.tensor([min_val, max_val])
            
            # 2) Apply count function to get output sequence
            output_seq = count_fn(input_seq)
            
            # Create full sequence: [min_val, max_val] + [sep_token] + [output]
            full_seq = torch.cat([
                input_seq,
                torch.tensor([self.sep_token]),
                output_seq
            ])
            
            self.sequences.append(full_seq)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        return self.sequences[index]


class SortingDataset(Dataset):
    """
    Dataset for SORTING task: given random sequence, generate sorted sequence.
    
    This dataset generates sequences of the form: [input_seq] + [sep_token] + [sorted_input_seq]
    For next-token prediction training, where the model learns to sort sequences.
    
    Example:
    Input: [4, 12, 3, 7]
    Output: [3, 4, 7, 12] 
    Full sequence: [4, 12, 3, 7, 102, 3, 4, 7, 12]  (102 is separator '>')
    """
    
    def __init__(
        self,
        n_samples: int = 5000,
        min_range_size: int = 2,  # Minimum sequence length
        max_range_size: int = 10, # Maximum sequence length
        vocab_size: int = 20,
        sep_token: int = 102,
        pad_token: int = 103,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.min_range_size = min_range_size
        self.max_range_size = max_range_size
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.pad_token = pad_token
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Pregenerate all samples
        self.sequences = []
        for _ in range(n_samples):
            # 1) Sample sequence length uniformly from [min_range_size, max_range_size]
            seq_length = torch.randint(self.min_range_size, self.max_range_size + 1, (1,)).item()
            
            # 2) Create random input sequence of that length with values from [0, vocab_size)
            input_seq = torch.randint(0, self.vocab_size, (seq_length,))
            
            # 3) Apply sort function to get output sequence
            output_seq = sort_fn(input_seq)
            
            # Create full sequence: input + [sep_token] + sorted_output
            full_seq = torch.cat([
                input_seq,
                torch.tensor([self.sep_token]),
                output_seq
            ])
            
            self.sequences.append(full_seq)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        return self.sequences[index]


class CopyDataset(Dataset):
    """
    Dataset for COPY task: given random sequence, generate copy of the sequence.

    This dataset generates sequences of the form: [input_seq] + [sep_token] + [input_seq]
    For next-token prediction training, where the model learns to copy sequences.

    Example:
    Input: [4, 12, 3, 7]
    Output: [4, 12, 3, 7]
    Full sequence: [4, 12, 3, 7, 102, 4, 12, 3, 7]  (102 is separator '>')

    This dataset ensures:
    - All input sequences are unique (no duplicate input sequences across samples)
    - Each input sequence contains unique tokens (sampled without replacement from vocab)
    - Sequence lengths are sampled uniformly from [min_range_size, max_range_size]
    """

    def __init__(
        self,
        n_samples: int = 5000,
        min_range_size: int = 2,  # Minimum sequence length
        max_range_size: int = 10, # Maximum sequence length
        vocab_size: int = 20,
        sep_token: int = 102,
        pad_token: int = 103,
        unique_values: bool = True,  # Whether sequences should have unique tokens
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.min_range_size = min_range_size
        self.max_range_size = max_range_size
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unique_values = unique_values

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Create list of valid tokens (excluding special tokens)
        all_tokens = set(range(self.vocab_size))
        special_tokens = {self.sep_token, self.pad_token}
        valid_tokens = torch.tensor(sorted(all_tokens - special_tokens))

        # Check if we have enough valid tokens
        if self.unique_values and len(valid_tokens) < self.max_range_size:
            raise ValueError(
                f"Not enough valid tokens for unique sequences. "
                f"vocab_size={vocab_size}, special tokens={special_tokens}, "
                f"valid tokens={len(valid_tokens)}, but max_range_size={max_range_size}"
            )

        # Generate all samples and track uniqueness
        self.sequences = []
        seen_sequences = set()

        # Estimate maximum attempts to find unique sequences
        max_attempts = n_samples * 100  # Allow many attempts to find unique sequences
        attempts = 0

        while len(self.sequences) < n_samples and attempts < max_attempts:
            attempts += 1

            # 1) Sample sequence length uniformly from [min_range_size, max_range_size]
            seq_length = torch.randint(self.min_range_size, self.max_range_size + 1, (1,)).item()

            # 2) Create random input sequence
            if self.unique_values:
                # Ensure sequence length doesn't exceed number of valid tokens
                seq_length = min(seq_length, len(valid_tokens))
                # Sample unique values without replacement from valid tokens only
                indices = torch.randperm(len(valid_tokens))[:seq_length]
                input_seq = valid_tokens[indices]
            else:
                # Sample with replacement (can have duplicates) from valid tokens only
                indices = torch.randint(0, len(valid_tokens), (seq_length,))
                input_seq = valid_tokens[indices]

            # 3) Check for uniqueness - convert to tuple for hashing
            seq_tuple = tuple(input_seq.tolist())
            if seq_tuple in seen_sequences:
                continue

            # 4) This is a valid unique sequence
            seen_sequences.add(seq_tuple)

            # 5) Copy the input sequence (output = input)
            output_seq = input_seq.clone()

            # Create full sequence: input + [sep_token] + copied_output
            full_seq = torch.cat([
                input_seq,
                torch.tensor([self.sep_token]),
                output_seq
            ])

            self.sequences.append(full_seq)

        # Check if we successfully generated enough unique samples
        if len(self.sequences) < n_samples:
            raise ValueError(
                f"Cannot generate {n_samples} unique copy problems with given constraints. "
                f"Only generated {len(self.sequences)} unique sequences after {max_attempts} attempts. "
                f"Consider increasing vocab_size ({vocab_size}), increasing max_range_size ({max_range_size}), "
                f"or reducing n_samples."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        return self.sequences[index]


class MajorityDataset(Dataset):
    """
    Dataset for MAJORITY task: given random sequence, identify the most frequent token.

    This dataset generates sequences of the form: [input_seq] + [sep_token] + [majority_token]
    For next-token prediction training, where the model learns to identify the majority element.

    Example (with vocab_size=2, binary):
    Input: [1, 1, 0, 0, 1]
    Output: [1] (since 1 appears 3 times vs 0 appears 2 times)
    Full sequence: [1, 1, 0, 0, 1, 102, 1]  (102 is separator '>')

    This dataset ensures:
    - All sequences are unique (no duplicate sequences)
    - No ties (one token always has strict majority)
    - Sequence lengths are sampled uniformly from [1, max_seq_len]
    """

    def __init__(
        self,
        n_samples: int = 5000,
        max_seq_len: int = 10,
        vocab_size: int = 2,
        sep_token: int = 102,
        pad_token: int = 103,
        seed: int = 42,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.sep_token = sep_token
        self.pad_token = pad_token

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Generate all samples and track uniqueness
        self.sequences = []
        seen_sequences = set()

        # Estimate maximum possible unique sequences
        # This is a rough upper bound - actual count depends on no-tie constraint
        max_attempts = n_samples * 100  # Allow many attempts to find unique sequences
        attempts = 0

        while len(self.sequences) < n_samples and attempts < max_attempts:
            attempts += 1

            # 1) Sample sequence length uniformly from [1, max_seq_len]
            seq_length = torch.randint(1, self.max_seq_len + 1, (1,)).item()

            # 2) Create random input sequence with values from [0, vocab_size)
            # IMPORTANT: Exclude sep_token and pad_token from input sequences to avoid ambiguity
            input_seq = torch.randint(0, self.vocab_size, (seq_length,))

            # Replace any occurrences of sep_token or pad_token with valid vocab tokens
            # This ensures input sequences don't contain special tokens
            while (input_seq == self.sep_token).any() or (input_seq == self.pad_token).any():
                mask = (input_seq == self.sep_token) | (input_seq == self.pad_token)
                input_seq[mask] = torch.randint(0, self.vocab_size, (mask.sum().item(),))

            # 3) Check for ties - reject sequences where no clear majority exists
            unique_vals, counts = torch.unique(input_seq, return_counts=True)
            max_count = torch.max(counts)
            num_with_max_count = torch.sum(counts == max_count).item()

            # Skip if there's a tie (multiple values have the same max count)
            if num_with_max_count > 1:
                continue

            # 4) Check for uniqueness - convert to tuple for hashing
            seq_tuple = tuple(input_seq.tolist())
            if seq_tuple in seen_sequences:
                continue

            # 5) This is a valid unique sequence with no ties
            seen_sequences.add(seq_tuple)

            # 6) Apply majority function to get output
            output_seq = majority_fn(input_seq)

            # Create full sequence: input + [sep_token] + majority_token
            full_seq = torch.cat([
                input_seq,
                torch.tensor([self.sep_token]),
                output_seq
            ])

            self.sequences.append(full_seq)

        # Check if we successfully generated enough unique samples
        if len(self.sequences) < n_samples:
            raise ValueError(
                f"Cannot generate {n_samples} unique majority problems with given constraints. "
                f"Only generated {len(self.sequences)} unique sequences after {max_attempts} attempts. "
                f"Consider increasing max_seq_len ({max_seq_len}), increasing vocab_size ({vocab_size}), "
                f"or reducing n_samples."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        return self.sequences[index]


if __name__ == "__main__":
    copy_dataset = CopyDataset(
        n_samples=1000,
        min_range_size=1,
        max_range_size=5,
        vocab_size=10,
        sep_token=102,
        pad_token=103,
        unique_values=True,
        seed=42,
    )
    for i in range(5):
        seq = copy_dataset[i]
        print(f"Sample {i}: {seq.tolist()}")
