import os
import json
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class task1_Dataset_MNIST(Dataset):
    """
    Dataset class for task 1 : sample and test images, with sequence of noise images in between
    """
    def __init__(self, is_train_data, data_processing_fun, data_args, delay_length=5, p=0.5, transform=None, empty_image_value=0, 
                 empty_image_class = -1):
        """
        Initialize the dataset class
        """
        # self.data_processing_fun = data_processing_fun
        # self.data_args = data_args
        # self.data = data
        # self.labels = labels
                
        self.transform = transform or transform or self.default_transform()
        # datasets.MNIST('./data/MNIST/raw', train=is_train_data, download=True, transform=self.transform)
        
        self.data, self.labels = self.load_data_and_labels(data_processing_fun, data_args)

        self.delay_length = delay_length
        self.p = p
        self.empty_image = np.full((1, 28, 28), fill_value=empty_image_value, dtype=np.uint8)
        self.num_classes = len(set(self.labels))
        self.empty_image_class = empty_image_class

    def load_data_and_labels(self, data_processing_fun, data_args):
        """
        Load data and labels using the provided loader functions and arguments
        This function aims to produce a flexible scheme where the dataset added can vary. However for the moment it includes MNIST only
        """
        data, labels = data_processing_fun(*data_args)
        return data, labels
    
    def default_transform(self):
        """
        Define default transformation: Convert image to PyTorch tensor and normalize
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization to range (-1,1)
        ])


    def __len__(self):
        """
        Length of the dataset is the number of images
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Generate one item of the dataset, with two types of labels (classes and repeats)
        """
        # MNIST images are grayscale and need to be reshaped appropriately
        sample_image = self.data[idx].reshape(1, 28, 28)  
        sample_label = self.labels[idx]

        # Randomly choose whether the test image matches the sample
        match = random.random() < self.p
        if match:
            same_class_images = [(img.reshape(1, 28, 28), lbl) for i, (img, lbl) in enumerate(zip(self.data, self.labels)) if lbl == sample_label and i != idx]
            test_image, _ = random.choice(same_class_images)
        else:
            different_class_images = [(img.reshape(1, 28, 28), lbl) for img, lbl in zip(self.data, self.labels) if lbl != sample_label]
            test_image, test_label = random.choice(different_class_images)

        # 3-channel inputs as the model will expect this shape
        sample_image_transformed = self.transform(sample_image)
        test_image_transformed = self.transform(test_image)
        delay_images_transformed = [self.transform(self.empty_image) for _ in range(self.delay_length)]
            
        sequence = torch.stack([sample_image_transformed] + delay_images_transformed + [test_image_transformed], dim=0)

        # First label type: model class of every image
        model_classes = [sample_label] + [self.empty_image_class for _ in range(self.delay_length)] + [test_label if not match else sample_label]

        # Second label type: if the image is repeated or not
        repeats = [0] + [0] + [1 for _ in range(self.delay_length - 1)] + [1 if match else 0]

        # to tensors
        model_classes = torch.tensor(model_classes, dtype=torch.long)
        repeats = torch.tensor(repeats, dtype=torch.long)
        sample_test_match = torch.tensor([float(match)], dtype=torch.long)

        return sequence, model_classes, repeats, sample_test_match
    





class tasks2and3_Dataset_MNIST(Dataset):
    """
    Dataset class for tasks 2 and 3 : sample and test images, with sequence of noise images in between which can be passive distractors or not.
    Task 2 only needs to mention if there has been a repetition between sample and test images, whereas task 3 needs to also correctly indicate the class of every distractor image.
    """
    def __init__(self, is_train_data, data_processing_fun, data_args, delay_length=5, p=0.50, p2 = 0.75, transform=None, empty_image_value=0, 
                 empty_image_class = -1):
        """
        Initialize the dataset class
        """
        self.transform = transform or transform or self.default_transform()
        # datasets.MNIST('./data/MNIST/raw', train=is_train_data, download=True, transform=self.transform)
        
        self.data, self.labels = self.load_data_and_labels(data_processing_fun, data_args)

        self.delay_length = delay_length
        self.p = p
        self.empty_image = np.full((1, 28, 28), fill_value=empty_image_value, dtype=np.uint8)
        self.num_classes = len(set(self.labels))
        self.empty_image_class = empty_image_class

        self.p2 = p2 # p2 addresses the probability of the noise image being a distractor

    def load_data_and_labels(self, data_processing_fun, data_args):
        """
        Load data and labels using the provided loader functions and arguments
        This function aims to produce a flexible scheme where the dataset added can vary. However for the moment it includes MNIST only
        """
        data, labels = data_processing_fun(*data_args)
        return data, labels
    
    def default_transform(self):
        """
        Define default transformation: Convert image to PyTorch tensor and normalize
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization to range (-1,1)
        ])


    def __len__(self):
        """
        Length of the dataset is the number of images
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Generate one item of the dataset, with two types of labels (classes and repeats)
        """
        # MNIST images are grayscale and need to be reshaped appropriately
        sample_image = self.data[idx].reshape(1, 28, 28)  
        sample_label = self.labels[idx]

        # Randomly choose whether the test image matches the sample
        match = random.random() < self.p
        if match:
            same_class_images = [(img.reshape(1, 28, 28), lbl) for i, (img, lbl) in enumerate(zip(self.data, self.labels)) if lbl == sample_label and i != idx]
            test_image, _ = random.choice(same_class_images)
        else:
            different_class_images = [(img.reshape(1, 28, 28), lbl) for img, lbl in zip(self.data, self.labels) if lbl != sample_label]
            test_image, test_label = random.choice(different_class_images)

        # Create the delay images in between with the possibility of being distractors or black noise
        delay_images = []
        delay_labels = []
        for _ in range(self.delay_length):
            if random.random() < self.p2:  # With probability p2, add a distractor
                # Ensure distractor is not the same as the sample
                while True:
                    distractor_idx = random.randint(0, len(self.data) - 1)
                    distractor_image, distractor_label = self.data[distractor_idx].reshape(1, 28, 28), self.labels[distractor_idx]
                    if distractor_label != sample_label:
                        break
                delay_images.append(distractor_image)
                delay_labels.append(distractor_label)
            else:
                delay_images.append(self.empty_image)
                delay_labels.append(self.empty_image_class)  
                
        # 3-channel inputs as the model will expect this shape
        sample_image_transformed = self.transform(sample_image)
        test_image_transformed = self.transform(test_image)
        delay_images_transformed = [self.transform(image) for image in delay_images]
            
        sequence = torch.stack([sample_image_transformed] + delay_images_transformed + [test_image_transformed], dim=0)

        # First label type: model class of every image
        model_classes = [sample_label] + delay_labels + [test_label if not match else sample_label]

        # Second label type: if the image is repeated or not
        seen_classes = set()
        repeats = [0] * len(model_classes)
        for i, cls in enumerate(model_classes):
            if cls in seen_classes:
                repeats[i] = 1
            else:
                seen_classes.add(cls)

        # to tensors
        model_classes = torch.tensor(model_classes, dtype=torch.long)
        repeats = torch.tensor(repeats, dtype=torch.long)
        sample_test_match = torch.tensor([float(match)], dtype=torch.long)

        return sequence, model_classes, repeats, sample_test_match
    



class tasks4and5_Dataset_MNIST(Dataset):
    """
    Dataset class for tasks 4 and 5 : no sample or test images, but just a sequence of images where there is not predefined order.
    Task 4 only needs to mention if there has been a repetition in the whole sequence whereas task 5 needs to mention if every individual image si repetated or not.
    """
    def __init__(self, is_train_data, data_processing_fun, data_args, sequence_length=5, p=0.50, p2 = 0.75, transform=None, empty_image_value=0, 
                 empty_image_class = -1):
        """
        Initialize the dataset class
        """
        self.transform = transform or transform or self.default_transform()
        # datasets.MNIST('./data/MNIST/raw', train=is_train_data, download=True, transform=self.transform)
        
        self.data, self.labels = self.load_data_and_labels(data_processing_fun, data_args)

        self.sequence_length = sequence_length
        self.p = p
        self.empty_image = np.full((1, 28, 28), fill_value=empty_image_value, dtype=np.uint8)
        self.num_classes = len(set(self.labels))
        self.empty_image_class = empty_image_class

        self.p2 = p2 # p2 addresses the probability of the noise image being a distractor

    def load_data_and_labels(self, data_processing_fun, data_args):
        """
        Load data and labels using the provided loader functions and arguments
        This function aims to produce a flexible scheme where the dataset added can vary. However for the moment it includes MNIST only
        """
        data, labels = data_processing_fun(*data_args)
        return data, labels
    
    def default_transform(self):
        """
        Define default transformation: Convert image to PyTorch tensor and normalize
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization to range (-1,1)
        ])


    def __len__(self):
        """
        Length of the dataset is the number of images
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Generate one item of the dataset, with two types of labels (classes and repeats)
        """
        # Create the sequence of images where every individual image can be either an image from the dataset or a black noise image
        sequence_images = []
        sequence_labels = []
        for _ in range(self.sequence_length):
            if random.random() < self.p2:  # With probability p2, add a distractor
                distractor_idx = random.randint(0, len(self.data) - 1)
                distractor_image, distractor_label = self.data[distractor_idx].reshape(1, 28, 28), self.labels[distractor_idx]
                sequence_images.append(distractor_image)
                sequence_labels.append(distractor_label)
            else:
                sequence_images.append(self.empty_image)
                sequence_labels.append(self.empty_image_class)  
                
        # 3-channel inputs as the model will expect this shape
        images_transformed = [self.transform(image) for image in sequence_images]
        sequence = torch.stack(images_transformed, dim=0)

        # First label type: model class of every image
        model_classes = sequence_labels
        
        # Second label type: if the image is repeated or not
        seen_classes = set()
        repeats = [0] * len(model_classes)
        for i, cls in enumerate(model_classes):
            if cls in seen_classes:
                repeats[i] = 1
            else:
                seen_classes.add(cls)

        # to tensors
        model_classes = torch.tensor(model_classes, dtype=torch.long)
        repeats = torch.tensor(repeats, dtype=torch.long)
        
        return sequence, model_classes, repeats