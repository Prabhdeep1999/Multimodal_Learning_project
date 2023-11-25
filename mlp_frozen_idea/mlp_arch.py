import torch
import torch.nn as nn
from torch.utils.data import DataLoader

max_iters = 50
batch_size = 8
learning_rate = 0.008
hidden_size = 64

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.droput_layer = nn.Dropout(0.4)
        self.final_layer = nn.Sequential(
            nn.Linear(512, 4),
        )

    def forward(self, x):
        
        # flattening the input
        flatten_inp = self.flatten(x)
        
        # linear layer with droput to ensure utilization of each modality
        logits_fc1 = self.droput_layer(self.fc1(flatten_inp))
        
        # linear layers and dropout for ensuring regularization
        logits_fc2 = self.fc2(logits_fc1)
        logits_fc3 = self.droput_layer(self.fc3(logits_fc2))
        logits_fc4 = self.fc4(logits_fc3)
        logits_fc5 = self.droput_layer(self.fc5(logits_fc4))
        
        # skip connection
        logits_fc5 = logits_fc5 + logits_fc3
        logits_fc6 = self.fc6(logits_fc5)
        
        # skip connection
        logits_fc6 = logits_fc6 + logits_fc2
        logits = self.droput_layer(logits)
        logits = self.final_layer(logits)

        return logits

net = MLP()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

tot_acc = []
tot_loss = []
for epoch in range(max_iters):  # loop over the dataset multiple times

    running_loss = 0.0
    acc = 0
    valid_loss = 0
    v_acc = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        if type(net).__name__ == "MLP":
            inputs, labels = data
        elif type(net).__name__ == "ConvNet":
            inputs, labels = data
            try:
                inputs = torch.reshape(inputs, (batch_size, 1, 32, 32))
            except Exception as e:
                print(e)
                continue

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, pred = torch.max(outputs.data, 1)
        acc += ((labels == pred).sum().item())

        # print statistics
        running_loss += loss.item()
    
    train_acc = acc / len(train_data)
    if epoch % 2 == 0:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}, accuracy: {train_acc:.3f}')
        tot_acc.append(train_acc)
        tot_loss.append(running_loss)
        running_loss = 0.0
        train_acc = 0.0

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        try:
            inputs = torch.reshape(inputs, (batch_size, 1, 32, 32))
        except Exception as e:
            print(e)
            continue
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

# plot loss curves
import matplotlib.pyplot as plt
plt.plot(range(len(tot_loss)), tot_loss, label="Loss")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(tot_loss) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()

# plot accuracy curves
plt.plot(range(len(tot_acc)), tot_acc, label="Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(tot_acc) - 1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.show()