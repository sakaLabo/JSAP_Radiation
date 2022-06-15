import torch

INPUT_NUM = 2
NEURON0_NUM = 20
NEURON1_NUM = 10
OUTPUT_NUM = 2
BATCH_SIZE = 25

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(INPUT_NUM, NEURON0_NUM)
        self.layer1 = torch.nn.Linear(NEURON0_NUM, NEURON1_NUM)
        self.layer2 = torch.nn.Linear(NEURON1_NUM, OUTPUT_NUM)

    def forward(self, x):
        x = torch.relu(self.layer0(x))
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x