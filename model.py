import torch.nn as nn


class StocksModel(nn.Module):
    NN_WIDTH = 32

    def __init__(self, input_layer_size):
        super().__init__()
        self.fc1 = nn.Linear(input_layer_size, self.NN_WIDTH)
        self.av1 = nn.ReLU()
        self.fc2 = nn.Linear(self.NN_WIDTH, self.NN_WIDTH)
        self.av2 = nn.ReLU()
        self.fc3 = nn.Linear(self.NN_WIDTH, self.NN_WIDTH)
        self.av3 = nn.ReLU()
        self.fc4 = nn.Linear(self.NN_WIDTH, self.NN_WIDTH)
        self.av4 = nn.ReLU()
        self.fc5 = nn.Linear(self.NN_WIDTH, self.NN_WIDTH)
        self.av5 = nn.ReLU()
        self.fc6 = nn.Linear(self.NN_WIDTH, 8)
        self.avout = nn.LogSoftmax(dim=-1)
        return None

    def forward(self, x):
        x = self.fc1(x)
        x = self.av1(x)
        x = self.fc2(x)
        x = self.av2(x)
        x = self.fc3(x)
        x = self.av3(x)
        x = self.fc4(x)
        x = self.av4(x)
        x = self.fc5(x)
        x = self.av5(x)
        x = self.fc6(x)
        x = self.avout(x)  # Alternate way: nn.functional.log_softmax(x)
        return x