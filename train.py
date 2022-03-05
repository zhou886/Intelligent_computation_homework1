from numpy import float32
from torch.utils.data import DataLoader
from torch.nn import *
from torch.optim import *
from torch import cuda, no_grad, tensor, zeros
from torchvision import datasets, transforms
from network import Network

def train(network_module:Network) -> float:
    '''
    接受一个CNN网络模型,返回它在测试集上的总损失
    '''
    epoch = 50
    batch_size = 512
    learning_rate = 0.001

    train_set = datasets.MNIST(r'./MINST', train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(r'./MINST', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    loss_function_MSE = MSELoss()
    optimizer = SGD(network_module.parameters(), lr=learning_rate)

    if cuda.is_available():
        network_module = network_module.cuda()
        loss_function_MSE = loss_function_MSE.cuda()

    test_set_size = len(test_set)
    total_test_loss = 0
    total_test_accuracy = 0

    for i in range(epoch):

        network_module.train()
        for data in train_loader:
            imgs, targets = data
            if cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            tmp = zeros(len(targets), 10)
            for i in range(len(targets)):
                tmp[i][targets[i]] = 1
            if cuda.is_available():
                tmp = tmp.cuda()
            targets = tmp

            output = network_module(imgs)
            loss = loss_function_MSE(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network_module.eval()

        total_test_loss = 0
        total_test_accuracy = 0

        with no_grad():
            for data in test_loader:
                imgs, targets = data
                if cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()

                tmp = zeros(len(targets), 10)
                for i in range(len(targets)):
                    tmp[i][targets[i]] = 1
                if cuda.is_available():
                    tmp = tmp.cuda()
            
                output = network_module(imgs)
                loss = loss_function_MSE(output, tmp)
                total_test_loss += loss.item()
                accuracy = (output.argmax(1) == targets).sum()
                total_test_accuracy += accuracy
        
        total_test_accuracy = 1.0*total_test_accuracy/test_set_size
    
    return total_test_loss