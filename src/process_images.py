import pickle
import matplotlib.pyplot as plt
import struct
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']
def show_image(one_image):
    # Reshape the array to 32x32x3
    image = one_image.reshape(3, 32, 32).transpose(1, 2, 0)
    plt.imshow(image)
    plt.show()
def show_sequence(sequence, label):
    # sequence shape: [sequence_length, channels, height, width]
    # label shape: [1]

    fig, axs = plt.subplots(1, len(sequence), figsize=(15, 3))  # Adjust figsize as needed
    for i, img in enumerate(sequence):
        img = img.transpose(0, 2).transpose(0, 1)  # Convert to [height, width, channels]
        axs[i].imshow(img)
        axs[i].axis('off')  # Turn off axis

    plt.suptitle(f'Label: {label.item()}')
    plt.show()


##
## MNIST Images
##
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, _, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, _ = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def show_image_MNIST(one_image, title = 'MNIST image'):
    plt.imshow(one_image, cmap='gray')
    plt.title(title)
    plt.show()

def show_sequence_MNIST(sequence, labels):
    fig, axs = plt.subplots(1, len(sequence), figsize=(15, 3))
    for i, img in enumerate(sequence):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.suptitle(f'Labels: {" ".join(str(label) for label in labels)}')
    plt.show()



def create_pairs_MNIST(images, labels, num_pairs=3, p=0.5):
    pairs = []
    pair_labels = []
    unique_classes = np.unique(labels)

    # Determine the indeces for each class
    class_indices = [np.where(labels == i)[0] for i in unique_classes]

    # Every image will have asssociated 3 different pairs of images
    for idx, image in enumerate(images):
        current_class = labels[idx]
        
        # Generate pairs, similar or dissimilar according to probability p
        for _ in range(num_pairs):
            if np.random.rand() < p:
                same_class_indices = class_indices[current_class]
                pair_idx = np.random.choice(same_class_indices)
                pair_label = 0  # Similar
            else:
                different_class = np.random.choice(list(set(unique_classes) - {current_class}))
                different_class_indices = class_indices[different_class]
                pair_idx = np.random.choice(different_class_indices)
                pair_label = 1  # Dissimilar
            
            pairs.append((image, images[pair_idx]))
            pair_labels.append(pair_label)
            
    return pairs, pair_labels


def fromtuple2loader(image_pairs, labels, batch_size = 64) : 
    tensor_pairs = torch.stack([torch.stack(list(map(torch.Tensor, z))) for z in image_pairs])
    tensor_labels = torch.Tensor(labels).long()

    dataset = TensorDataset(tensor_pairs, tensor_labels)
    loader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, loader


def visualize_loader_pairs_MNIST(dataloader, num_to_visualize=3):
    fig, axes = plt.subplots(num_to_visualize, 2, figsize=(5, num_to_visualize*2))
    for i, (pair_batch, label_batch) in enumerate(dataloader):
        if i >= num_to_visualize: break
        image1 = pair_batch[i][0].squeeze() 
        image2 = pair_batch[i][1].squeeze()

        axes[i, 0].imshow(image1, cmap='gray')
        axes[i, 0].set_title('Image 1')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image2, cmap='gray')
        axes[i, 1].set_title('Image 2 - Label: {}'.format('Same' if label_batch[i] == 0 else 'Different'))
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
