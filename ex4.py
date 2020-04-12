import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gcommand_loader import GCommandLoader

# Load all the data
dataTrain = GCommandLoader("./data/train")
dataValid = GCommandLoader("./data/valid")
dataTest= GCommandLoader("./data/test")


valid_loader = torch.utils.data.DataLoader(
        dataValid, batch_size=100, shuffle=None,
        num_workers=0, pin_memory=True, sampler=None)

train_loader = torch.utils.data.DataLoader(
        dataTrain, batch_size=100, shuffle=True, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        dataTest, shuffle=False, pin_memory=True)

# initialize learning rate
lr=0.001

# CNN algorithm
class CnnAlgo(nn.Module):
    # initialize parameters
    def __init__(self):
        numOfClass = 30
        super(CnnAlgo, self).__init__()
        # create two conv layer
        self.layer1 = nn.Sequential(nn.Conv2d(1, 10,kernel_size=5,stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(10, 15,kernel_size=5,stride=1,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))
        self.dropout = nn.Dropout()
        # create three hidden layer
        self.fc1 = nn.Linear(15000, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, numOfClass)

    # The forward function
    def forward(self, input):
        # send input to the first layer and fed into the following layer
        result = self.layer1(input)
        result = self.layer2(result)
        result = result.reshape(result.size(0),-1)
        # make dropout
        result = self.dropout(result)
        # send result to the hidden layer
        result = self.fc1(result)
        result = self.fc2(result)
        result = self.fc3(result)
        return result

    # The test function
    def test(self):
        # create file in name test_y
        file = open("test_y", "w")
        model.eval()
        with torch.no_grad():
            total = 0
            # run over all the data
            for images, labels in test_loader:
                output = model(images)
                # predict the label
                pred = output.data.max(1, keepdim=True)[1]
                num = pred.sum().item()
                # get the data file name
                dataFileName = dataTest.spects[total][0].split('\\')
                # write to the file the file name and the prediction
                file.write(str(dataFileName[2]) + ", " + str(num) + "\n")
                total += 1


    # The validation function calculate the accuracy
    def validate(self):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                resultOut = model(images)
                _, predicted = torch.max(resultOut.data, 1)
                total += labels.size(0)
                # calculate the accuracy
                correct += (predicted == labels).sum().item()
            # print the Accuracy of the model
            print('Test Accuracy is : {} %'.format((correct / total) * 100))

    # The function train_ model
    def train_model(self):
        model.train()
        # run over all the data and train the model
        for i,(image,label) in enumerate(train_loader):
            output = model(image)
            # calculate loss
            loss = criterion(output, label)
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# create model
model = CnnAlgo()
criterion = nn.CrossEntropyLoss()
# create adam optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# main function
def main():
    # run over the algorithm 8 times
    for epoch in range(1, 8):
        # train the algorithm
        model.train_model()
        # validate the model
        model.validate()
    # test the model
    model.test()

main()




