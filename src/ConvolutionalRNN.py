import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAttention(nn.Module):
    """Applies attention mechanism on the output features from the CNN."""
    def __init__(self, channels):
        super(CNNAttention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.conv(x)  # Apply convolution
        x = self.avg_pool(x)  # Reduce frequency dimension to 1
        x = x.squeeze(2)  # Remove the frequency dimension
        attention = F.softmax(x, dim=2)  # Apply softmax across the width (time dimension)
        return attention

class RNNAttention(nn.Module):
    """Applies attention mechanism on the output features from the RNN."""
    def __init__(self, features):
        super(RNNAttention, self).__init__()
        self.fc = nn.Linear(features, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        scores = self.fc(x).squeeze(2)  # Linear transformation and remove last dim
        weights = F.softmax(scores, dim=1)  # Softmax over sequences
        context = torch.sum(x * weights.unsqueeze(2), dim=1)  # Weighted sum of RNN outputs
        return context, weights

class ConvolutionalRNN(nn.Module):
    """Combines CNN, RNN and attention mechanisms for feature extraction and classification."""
    def __init__(self, input_channels=2, num_classes=10):
        super(ConvolutionalRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 3), stride=(4, 3)),  # Output: (batch, 32, 31, 42)
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # Output: (batch, 64, 7, 42)
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 6)),  # Output: (batch, 128, 7, 7)
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 7))  # Output: (batch, 256, 1, 7)
        )
        
        self.cnn_attention = CNNAttention(256)
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        self.rnn_attention = RNNAttention(512)  # 256 * 2 because of bidirectional GRU
        self.fc = nn.Linear(512, num_classes)  # Output layer after attention

    def forward(self, x):
        # Convolution layers
        x = self.conv_layers(x)
        
        # CNN Attention
        cnn_attention = self.cnn_attention(x)

        x = x * cnn_attention.unsqueeze(2)  # Apply attention over channels
        
        # Prepare for RNN
        x = x.squeeze(2)  # Remove the frequency dimension
        x = x.permute(0, 2, 1).contiguous()  # Reshape to (batch, seq_len, features)
        
        # RNN
        x, _ = self.gru(x)
        
        # Apply tanh activation
        x = torch.tanh(x)
        
        # RNN Attention
        context, rnn_attention_weights = self.rnn_attention(x)
        
        # Fully connected layer for classification
        x = self.fc(context)

        return x#, cnn_attention.squeeze(2), rnn_attention_weights





class BasicCNNModel(nn.Module):
    """A simple CNN model for classification of audio sequences."""
    def __init__(self, input_channels=2, num_classes=10):
        super(BasicCNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch, 64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: (batch, 128, 16, 16)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_size: int = 2, output_size: int = 10):
        """Standard Convolutional Network layers for the MNIST dataset.

        Args:
            input_size (int): input size of the model.
            output_size (int): The number of output classes.

        """
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(65536, 1024)
        self.fc2 = torch.nn.Linear(1024, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Example usage
if __name__ == '__main__':
    # Dummy input for testing
    dummy_input = torch.randn(10, 2, 128, 128)
    model = ConvolutionalRNN(input_channels=2, num_classes=10)
    output, cnn_attn, rnn_attn = model(dummy_input)
    print("Output shape:", output.shape)
    print("CNN Attention shape:", cnn_attn.shape)
    print("RNN Attention weights shape:", rnn_attn.shape)
