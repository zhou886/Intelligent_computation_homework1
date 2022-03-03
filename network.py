from numpy import argmax
from torch.nn import *

class Network(Module):
    def __init__(self, root) -> None:
        super().__init__()
        self.module = Sequential()
        
        channels = 1
        features = 0
        size = 28
        for i in range(len(root)):
            if root[i][0] == 'C':
                self.module.add_module('{0}Conv'.format(i), Conv2d(in_channels=channels, out_channels=root[i][2], kernel_size=root[i][1], padding=(root[i][1]-1)//2))
                channels = root[i][2]
            elif root[i][0] == 'P':
                self.module.add_module('{0}MaxPool'.format(i), MaxPool2d(kernel_size=root[i][1]))
                size //= root[i][1]
            else:
                if root[i-1][0] != 'F':
                    self.module.add_module('{0}Flatten'.format(i), Flatten())
                    self.module.add_module('{0}Linear'.format(i), Linear(in_features=channels*size*size, out_features=root[i][1]))
                    self.module.add_module('{0}Sigmoid'.format(i), Sigmoid())
                else:
                    self.module.add_module('{0}Linear'.format(i), Linear(in_features=features, out_features=root[i][1]))
                    self.module.add_module('{0}Sigmoid'.format(i), Sigmoid())
                features = root[i][1]
    
    def forward(self, input):
        output = self.module(input)
        return output