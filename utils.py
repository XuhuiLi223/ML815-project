import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import numpy as np
import time
import warnings
from misc import utils
import pandas as pd
import pickle
from model.resnet import ResNet
from model.resnet_ap import ResNetAP
from model.convnet import ConvNet
warnings.filterwarnings("ignore")

# Values borrowed from https://github.com/VICO-UoE/DatasetCondensation/blob/master/utils.py
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
MEANS = {'cifar': [0.4914, 0.4822, 0.4465], 'imagenet': [0.485, 0.456, 0.406]}
STDS = {'cifar': [0.2023, 0.1994, 0.2010], 'imagenet': [0.229, 0.224, 0.225]}
MEANS['cifar10'] = MEANS['cifar']
STDS['cifar10'] = STDS['cifar']
MEANS['cifar100'] = MEANS['cifar']
STDS['cifar100'] = STDS['cifar']
MEANS['svhn'] = [0.4377, 0.4438, 0.4728]
STDS['svhn'] = [0.1980, 0.2010, 0.1970]
MEANS['mnist'] = [0.1307]
STDS['mnist'] = [0.3081]
MEANS['fashion'] = [0.2861]
STDS['fashion'] = [0.3530]

def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet18':
        model = ResNet(args.dataset,
                       depth=18,
                       num_classes=nclass,
                       norm_type=args.norm_type,
                       size=size,
                       nch=args.nch)
    elif args.net_type == 'resnet32':
        model = ResNet(args.dataset,
                       depth=32,
                       num_classes=nclass,
                       norm_type=args.norm_type,
                       size=size,
                       nch=args.nch)
    elif args.net_type == 'convnet':
        model = ConvNet(num_classes=nclass)
    elif args.net_type == 'resnet-ap' or args.net_type == 'resnet_ap':
        model = ResNetAP(args.dataset,
                         depth=10,
                         num_classes=nclass,
                         width=args.width,
                         norm_type=args.norm_type,
                         size=size,
                         nch=args.nch)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}, norm: {args.norm_type}")

    return model


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]


class ImageFolder(datasets.DatasetFolder):
    """Dataset class for loading subsets with specified IPC.
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=datasets.folder.default_loader,
                 is_valid_file=None,
                 load_memory=False,
                 load_transform=None,
                 nclass=100,
                 phase=0,
                 slct_type='random',
                 ipc=-1,
                 seed=-1,
                 spec='none',
                 return_origin=False,
                 return_path=False,
                 mode_id_file=None):
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None
        super(ImageFolder, self).__init__(root,
                                          loader,
                                          self.extensions,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        # Override
        self.spec = spec
        self.return_origin = return_origin
        if nclass < 1000:
            self.classes, self.class_to_idx = self.find_subclasses(nclass=nclass,
                                                                   phase=phase,
                                                                   seed=seed)
        else:
            self.classes, self.class_to_idx = self.find_classes(self.root)
        self.original_labels = self.find_original_classes()
        self.nclass = nclass
        self.samples = datasets.folder.make_dataset(self.root, self.class_to_idx, self.extensions,
                                                    is_valid_file)

        if ipc > 0:
            self.samples = self._subset(slct_type=slct_type, ipc=ipc)

        self.targets = [s[1] for s in self.samples]
        self.original_targets = [self.original_labels[s[1]] for s in self.samples]
        
        self.load_memory = load_memory
        self.load_transform = load_transform
        if self.load_memory:
            self.imgs = self._load_images(load_transform)
        else:
            self.imgs = self.samples
        self.return_path = return_path
        self.mode_id_file = mode_id_file
        if self.mode_id_file is not None:
            self.mode_id_df = pd.read_csv(self.mode_id_file)
            self.mode_id_df = self.mode_id_df.set_index("image_id")
            self.mode_ids = [self.mode_id_df.loc[s[0].split("/")[-1]]["mode_id"] for s in self.samples]

    def find_subclasses(self, nclass=100, phase=0, seed=0):
        """Finds the class folders in a dataset.
        """
        classes = []
        phase = max(0, phase)
        cls_from = nclass * phase
        cls_to = nclass * (phase + 1)
        if seed == 0:
            if self.spec == 'woof':
                file_list = './misc/class_woof.txt'
            elif self.spec == 'nette':
                file_list = './misc/class_nette.txt'
            elif self.spec == 'imagenet100':
                file_list = './misc/class100.txt'
            elif self.spec == 'imagenet1k':
                file_list = './misc/class_indices.txt'
            else:
                raise AssertionError(f'spec does not exist!')
            with open(file_list, 'r') as f:
                class_name = f.readlines()
            for c in class_name:
                c = c.split('\n')[0]
                classes.append(c)
            classes = classes[cls_from:cls_to]
        else:
            np.random.seed(seed)
            class_indices = np.random.permutation(len(self.classes))[cls_from:cls_to]
            for i in class_indices:
                classes.append(self.classes[i])

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        assert len(classes) == nclass

        return classes, class_to_idx

    def find_original_classes(self):
        all_classes = sorted(os.listdir(self.root))
        original_labels = []
        for class_name in self.classes:
            original_labels.append(all_classes.index(class_name))
        return original_labels

    def _subset(self, slct_type='random', ipc=10):
        n = len(self.samples)
        idx_class = [[] for _ in range(self.nclass)]
        for i in range(n):
            label = self.samples[i][1]
            idx_class[label].append(i)

        min_class = np.array([len(idx_class[c]) for c in range(self.nclass)]).min()
        print("# examples in the smallest class: ", min_class)
        assert ipc <= min_class

        if slct_type == 'random':
            indices = np.arange(n)
        else:
            raise AssertionError(f'selection type does not exist!')

        samples_subset = []
        idx_class_slct = [[] for _ in range(self.nclass)]
        for i in indices:
            label = self.samples[i][1]
            if len(idx_class_slct[label]) < ipc:
                idx_class_slct[label].append(i)
                samples_subset.append(self.samples[i])

            if len(samples_subset) == ipc * self.nclass:
                break

        return samples_subset

    def _load_images(self, transform=None):
        """Load images on memory
        """
        imgs = []
        for i, (path, _) in enumerate(self.samples):
            sample = self.loader(path)
            if transform != None:
                sample = transform(sample)
            imgs.append(sample)
            if i % 100 == 0:
                print(f"Image loading.. {i}/{len(self.samples)}", end='\r')

        print(" " * 50, end='\r')
        return imgs


    def __getitem__(self, index):
        if not self.load_memory:
            path = self.samples[index][0]
            sample = self.loader(path)
            image_id = path.split("/")[-1]
        else:
            sample = self.imgs[index]
            

        target = self.targets[index]
        original_target = self.original_targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            original_target = self.target_transform(original_target)
            
        if self.mode_id_file is not None:
            if not self.load_memory:
                mode_id = self.mode_id_df.loc[image_id]['mode_id']
            else:
                mode_id = self.mode_ids[index]
            # Return original labels for DiT generation
            if self.return_origin:
                if self.return_path:
                    return sample, target, original_target, mode_id, path
                return sample, target, original_target, mode_id
            else:
                return sample, target, mode_id

        # Return original labels for DiT generation
        if self.return_origin:
            if self.return_path:
                return sample, target, original_target, path
            return sample, target, original_target
        else:
            return sample, target


def transform_cifar(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        print("Dataset with basic Cifar augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['cifar'], std=STDS['cifar'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_svhn(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(32, padding=4)]
        print("Dataset with basic SVHN augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_mnist(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic MNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_fashion(augment=False, from_tensor=False, normalize=True):
    if not augment:
        aug = []
    else:
        aug = [transforms.RandomCrop(28, padding=4)]
        print("Dataset with basic FashionMNIST augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(cast + aug + normal_fn)
    test_transform = transforms.Compose(cast + normal_fn)

    return train_transform, test_transform


def transform_imagenet(size=-1,
                       augment=False,
                       from_tensor=False,
                       normalize=True,
                       rrc=True,
                       rrc_size=-1):
    if size > 0:
        resize_train = [transforms.Resize(size), transforms.CenterCrop(size)]
        resize_test = [transforms.Resize(size), transforms.CenterCrop(size)]
        # print(f"Resize and crop training images to {size}")
    elif size == 0:
        resize_train = []
        resize_test = []
        assert rrc_size > 0, "Set RRC size!"
    else:
        resize_train = [transforms.RandomResizedCrop(224)]
        resize_test = [transforms.Resize(256), transforms.CenterCrop(224)]

    if not augment:
        aug = []
        # print("Loader with DSA augmentation")
    else:
        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[
                                      [-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203],
                                  ])
        aug = [transforms.RandomHorizontalFlip(), jittering, lighting]

        if rrc and size >= 0:
            if rrc_size == -1:
                rrc_size = size
            rrc_fn = transforms.RandomResizedCrop(rrc_size, scale=(0.5, 1.0))
            aug = [rrc_fn] + aug
            print("Dataset with basic imagenet augmentation and RRC")
        else:
            print("Dataset with basic imagenet augmentation")

    if from_tensor:
        cast = []
    else:
        cast = [transforms.ToTensor()]

    if normalize:
        normal_fn = [transforms.Normalize(mean=MEANS['imagenet'], std=STDS['imagenet'])]
    else:
        normal_fn = []

    train_transform = transforms.Compose(resize_train + cast + aug + normal_fn)
    test_transform = transforms.Compose(resize_test + cast + normal_fn)

    return train_transform, test_transform


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)


class ClassBatchSampler(object):
    """Intra-class batch sampler 
    """
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size=min(n_ex, batch_size),
                                                          drop_last=drop_last)
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target


class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)

        self.cls_targets = torch.tensor([np.ones(self.batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()


class ClassMemDataLoader():
    """Class loader with data on GPUs
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.targets, dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                           batch_size=batch_size,
                                                           drop_last=drop_last)
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)
        self.cls_targets = torch.tensor([np.ones(batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def class_sample(self, c, ipc=-1):
        if ipc > 0:
            indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)

        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target



def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test, download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test, download=True)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor(),
                                      download=True)
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test,
                                    download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor(), download=True)

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test, download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test, download=True)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader


def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = MEANS[dataname]
    std = STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean


def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)


# ==================== Metrics Functions ====================

class MetricsTracker:
    """Track training metrics including throughput, memory, and MFU"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.samples_processed = 0
        self.total_time = 0.0
        self.peak_memory = 0.0
        self.start_time = None
        self.flops_count = 0
        self.step_count = 0
    
    def start_step(self):
        """Call at the start of each training step"""
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
    
    def end_step(self, batch_size, num_gpus=1):
        """Call at the end of each training step"""
        if self.start_time is not None:
            step_time = time.time() - self.start_time
            self.total_time += step_time
            self.samples_processed += batch_size * num_gpus
            self.step_count += 1
        
        # Record peak memory (in GB)
        current_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.peak_memory = max(self.peak_memory, current_peak)
    
    def get_throughput(self):
        """Get throughput in samples/second"""
        if self.total_time > 0:
            return self.samples_processed / self.total_time
        return 0.0
    
    def get_peak_memory(self):
        """Get peak memory usage in GB"""
        return self.peak_memory
    
    def get_avg_step_time(self):
        """Get average step time in seconds"""
        if self.step_count > 0:
            return self.total_time / self.step_count
        return 0.0


def get_peak_memory_gb():
    """Get current peak memory usage in GB"""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory():
    """Reset peak memory statistics"""
    torch.cuda.reset_peak_memory_stats()


def calculate_throughput(samples_processed, time_elapsed):
    """
    Calculate throughput in samples/second
    
    Args:
        samples_processed: Total number of samples processed
        time_elapsed: Total time elapsed in seconds
    
    Returns:
        Throughput in samples/second
    """
    if time_elapsed > 0:
        return samples_processed / time_elapsed
    return 0.0


def get_model_flops(model, input_shape=(1, 3, 224, 224), device='cuda'):
    """
    Calculate model FLOPS using fvcore or manual calculation
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
        device: Device to run on
    
    Returns:
        Total FLOPS count
    """
    try:
        from fvcore.nn import FlopCountMode, flop_count
        model.eval()
        input_tensor = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            flops_dict, _ = flop_count(model, (input_tensor,), mode=FlopCountMode.OPERATION_COUNT)
        
        total_flops = sum(flops_dict.values())
        return total_flops
    except ImportError:
        # Fallback: estimate FLOPS manually
        return estimate_flops_manual(model, input_shape)


def estimate_flops_manual(model, input_shape):
    """
    Manual FLOPS estimation (simplified version)
    This is a basic estimation and may not be 100% accurate
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
    
    Returns:
        Estimated FLOPS count
    """
    total_flops = 0
    batch_size, channels, height, width = input_shape
    
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Conv2d FLOPS: kernel_size^2 * in_channels * out_channels * output_h * output_w * batch_size
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size[0] * module.kernel_size[1]
            stride = module.stride[0] * module.stride[1]
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            
            output_h = (height + 2 * padding - module.kernel_size[0]) // module.stride[0] + 1
            output_w = (width + 2 * padding - module.kernel_size[1]) // module.stride[1] + 1
            
            flops = kernel_size * in_channels * out_channels * output_h * output_w * batch_size
            total_flops += flops
            
            # Update dimensions for next layer
            height, width = output_h, output_w
            channels = out_channels
            
        elif isinstance(module, torch.nn.Linear):
            # Linear FLOPS: in_features * out_features * batch_size
            flops = module.in_features * module.out_features * batch_size
            total_flops += flops
    
    return total_flops


def calculate_mfu(flops_per_sample, throughput, num_gpus=1, gpu_flops_per_second=None):
    """
    Calculate Model FLOPS Utilization (MFU)
    
    MFU = (Actual FLOPS) / (Theoretical Peak FLOPS)
    
    Args:
        flops_per_sample: FLOPS per sample (forward + backward)
        throughput: Throughput in samples/second
        num_gpus: Number of GPUs
        gpu_flops_per_second: Theoretical peak FLOPS per GPU (e.g., A100: 312e12, V100: 125e12)
                             If None, will try to detect or use default
    
    Returns:
        MFU as a percentage (0-100)
    """
    if gpu_flops_per_second is None:
        # Try to detect GPU type and use default FLOPS
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        if "A100" in gpu_name or "A10" in gpu_name:
            gpu_flops_per_second = 312e12  # A100 FP32 peak
        elif "V100" in gpu_name:
            gpu_flops_per_second = 125e12  # V100 FP32 peak
        elif "RTX 3090" in gpu_name or "RTX 4090" in gpu_name:
            gpu_flops_per_second = 142e12  # RTX 3090 FP32 peak (approximate)
        else:
            # Default to a reasonable value
            gpu_flops_per_second = 100e12
    
    # Actual FLOPS = FLOPS per sample * throughput
    actual_flops = flops_per_sample * throughput
    
    # Theoretical peak FLOPS = GPU FLOPS * number of GPUs
    theoretical_flops = gpu_flops_per_second * num_gpus
    
    if theoretical_flops > 0:
        mfu = (actual_flops / theoretical_flops) * 100
        return mfu
    return 0.0


def get_training_metrics(model, samples_processed, time_elapsed, input_shape, 
                        num_gpus=1, device='cuda'):
    """
    Get comprehensive training metrics
    
    Args:
        model: PyTorch model
        samples_processed: Total samples processed
        time_elapsed: Total time elapsed in seconds
        input_shape: Input tensor shape for FLOPS calculation
        num_gpus: Number of GPUs used
        device: Device to run on
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Throughput
    metrics['throughput'] = calculate_throughput(samples_processed, time_elapsed)
    
    # Peak Memory
    metrics['peak_memory_gb'] = get_peak_memory_gb()
    
    # FLOPS
    forward_flops = get_model_flops(model, input_shape, device)
    # Backward pass typically requires ~2x FLOPS of forward
    total_flops_per_sample = forward_flops * 3  # forward + backward (2x)
    metrics['flops_per_sample'] = total_flops_per_sample
    metrics['flops_per_second'] = total_flops_per_sample * metrics['throughput']
    
    # MFU
    metrics['mfu_percent'] = calculate_mfu(
        total_flops_per_sample, 
        metrics['throughput'], 
        num_gpus
    )
    
    return metrics


def print_metrics(metrics, prefix=""):
    """Print formatted metrics"""
    print(f"{prefix}Metrics:")
    print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
    print(f"  Peak Memory: {metrics['peak_memory_gb']:.2f} GB")
    if 'flops_per_sample' in metrics:
        print(f"  FLOPS per sample: {metrics['flops_per_sample']/1e9:.2f} GFLOPs")
        print(f"  FLOPS per second: {metrics['flops_per_second']/1e12:.2f} TFLOPS")
    if 'mfu_percent' in metrics:
        print(f"  MFU: {metrics['mfu_percent']:.2f}%")






