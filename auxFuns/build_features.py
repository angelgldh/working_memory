import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LeNet(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=15, out_channels=30,
                 kernel_1=9, kernel_pooling=3, kernel_2=5,
                 out_dim=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=kernel_pooling, stride=3)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_2, padding=0)
        # Calculate the size of the flattened features after conv2 + pooling layers
        self._to_linear = None
        self._get_conv_output_size([in_channels, 28, 28])
        self.fc = nn.Linear(self._to_linear, out_dim)                             

    def _get_conv_output_size(self, shape):
        # Helper function to calculate the size of the flattened features
        input = torch.rand(shape).unsqueeze(0)  # Add a batch dimension
        output = self.pool(F.relu(self.conv1(input)))
        output = F.relu(self.conv2(output))
        self._to_linear = output.numel() // output.shape[0]  # Calculate total features excluding batch dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))      # Convolution + ReLU activation
        x = self.pool(x)               # Subsampling / Max pooling
        x = F.relu(self.conv2(x))      # Convolution + ReLU activation
        x = x.view(-1, self._to_linear)  # Flatten the tensor for the fully connected layer
        x = self.fc(x)                # Fully connected layer
        return x


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between two latent representations
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Loss for similar samples
        loss_similar = (1 - label) * torch.pow(euclidean_distance, 2)
        
        # Loss for dissimilar samples
        loss_dissimilar = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        # Combine the losses
        loss_contrastive = torch.mean(loss_similar + loss_dissimilar)

        return loss_contrastive
    

def compute_pairwise_distances(model, valid_loader):
    model.eval()
    distances = []
    true_labels = []
    with torch.no_grad():
        for data, labels in valid_loader:
            image1, image2 = data[:,0], data[:,1]
            
            # Forward pass
            output1 = model(image1)
            output2 = model(image2)
            
            # Compute Euclidean distance
            distance = F.pairwise_distance(output1, output2)
            distances.extend(distance.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(distances), np.array(true_labels)