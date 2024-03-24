import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(ConvLSTMClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # LSTM layer
        # Assuming the features from the conv layers are flattened and have size 64*7*7
        self.lstm = nn.LSTM(64*7*7, hidden_dim, num_layers, batch_first=True)
        
        # Classifier layer for the entire sequence
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Apply conv layers to each image in the sequence individually
        batch_size, seq_len, H, C, W = x.size()
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.conv1(c_in)
        c_out = F.relu(c_out)
        c_out = self.pool(c_out)
        c_out = self.conv2(c_out)
        c_out = F.relu(c_out)
        c_out = self.pool(c_out)
        
        # Prepare LSTM input
        r_in = c_out.view(batch_size, seq_len, -1)
        
        # LSTM output
        r_out, (h_n, h_c) = self.lstm(r_in)
        
        # Classifier output for the entire sequence
        output = self.classifier(r_out.contiguous().view(-1, r_out.size(2)))
        return output.view(batch_size, seq_len, -1)