import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    MLP model for Rank-N-Contrast
    input_size: int, size of the input, i.e., number of features
    n_classes: int, number of classes, i.e., number of output neurons
    """
    def __init__(self, input_size, n_classes, verbose=False):
        super(MLP, self).__init__()
        self.verbose = verbose
        self.n_classes = n_classes
        self.flatten = nn.Flatten()  # Flatten the input from (6, 1) to (6,)
        self.fc1 = nn.Linear(input_size, 64)  # First fully connected layer
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(64, 128)  # Second fully connected layer
        self.relu2 = nn.ReLU()
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(128, n_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        if self.verbose:
            print("After flattening: ", x.shape)

        x = self.fc1(x)  # First layer
        if self.verbose:
            print("After first layer: ", x.shape)

        x = self.tanh1(x)  # TanH activation

        x = self.fc2(x)  # Second layer
        if self.verbose:
            print("After second layer: ", x.shape)

        x = self.tanh2(x)  # ReLU activation
        x = self.fc3(x)  # Output layer
        x = torch.flatten(x, 1)
        if self.verbose:
            print("After last layer: ", x.shape)

        return x
        #return x.view(-1, self.n_classes, 1)  # Reshape to (batch_size, 128, 1)


class MLP_NACA(nn.Module):
    def __init__(self, input_size, n_classes, verbose=False):
        super(MLP_NACA, self).__init__()
        self.verbose = verbose
        self.n_classes = n_classes
        self.flatten = nn.Flatten()  # Flatten the input from (6, 1) to (6,)
        self.fc1 = nn.Linear(input_size, 512)  # First fully connected layer
        self.relu1 = nn.ReLU()
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(512, 1024)  # Second fully connected layer
        self.relu2 = nn.ReLU()
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(1024, n_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        if self.verbose:
            print("After flattening: ", x.shape)

        x = self.fc1(x)  # First layer
        if self.verbose:
            print("After first layer: ", x.shape)

        x = self.tanh1(x)  # TanH activation

        x = self.fc2(x)  # Second layer
        if self.verbose:
            print("After second layer: ", x.shape)

        x = self.tanh2(x)  # ReLU activation
        x = self.fc3(x)  # Output layer
        x = torch.flatten(x, 1)
        if self.verbose:
            print("After last layer: ", x.shape)

        return x
        #return x.view(-1, self.n_classes, 1)  # Reshape to (batch_size, 128, 1)