from cProfile import label
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import random

#A
df = pd.read_csv('houses.csv')
print(df.info())

#B
print('\nNum of NaN:')
print(df.isna().sum(axis=0))

#E
df['date'] = df['date'].str.replace("T000000","")
year = df['date'].astype(int)
df.drop(['date'], axis=1, inplace=True)
month = year // 100
year = month // 100
month = month % 100
df.insert(1, "year", year)
print(df['year'])
df.insert(2, "month", month)
print(df['month'])

#C
corrMatrix = df.corr()
print('\nCorrelation Matrix:')
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()
print('\nSquare feet living has the most')

#D
df.hist(column="price")
df.hist(column="sqft_living")
plt.show()

sn.jointplot(df["price"], df["sqft_living"])
plt.show()

#F
msk = np.random.rand(len(df)) <= 0.8

train_df = df[msk]
test_df = df[~msk]

#G
scaler = MinMaxScaler()

train = scaler.fit_transform(train_df)
test = scaler.transform(test_df)

#H
class data_delivery():
    def __init__(self,data):
        self.dataset = data
    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self,idx):
        temp = [*self.dataset[idx][0:3], *self.dataset[idx][4:22]]
        return torch.tensor(temp, dtype=torch.float32), self.dataset[idx][3]

train_loader = data_delivery(train)
test_loader = data_delivery(test)

model = nn.Sequential(
                    nn.Linear(21,42),
                    nn.ReLU(),
                    nn.Linear(42, 42),
                    nn.ReLU(),
                    nn.Linear(42, 21),
                    nn.ReLU(),
                    nn.Linear(21, 10),
                    nn.ReLU(),
                    nn.Linear(10, 1)
)

device = torch.device("cuda")
model = model.to(device)

criterion = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

batch_size = 200
train = torch.utils.data.DataLoader(train_loader, batch_size, True)
test_loader = torch.utils.data.DataLoader(test_loader, batch_size, False)

n = 50
los_test_all = []
los_train_all = []
los_min = 0.3

for i in range(n+1):
    los_test = []
    los_train = []
    model.train()
    for inputs, targets in train:
        inputs,targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outs = model(inputs)
        targets = targets.view(outs.shape[0],1)
        loss = criterion(outs,targets)
        los_train.append(loss.item())
        loss.backward()
        optimizer.step()
    los_train_all.append(sum(los_train)/len(los_train))
    print(f'epoch',i)
    model.eval()

    with torch.no_grad():
        for c,j in test_loader:
            c,j = c.to(device), j.to(device)
            outs_test = model(c)
            j = j.view(outs_test.shape[0], 1)
            los_test.append(criterion(outs_test, j).item())
        los_now = sum(los_test)/len(los_test)
        los_test_all.append(sum(los_test)/len(los_test))
        print(los_now)
        if(los_now < los_min):
            los_min = los_now
            torch.save(model,"./model.pth")

plt.plot(los_test_all, label='test loss')
plt.plot(los_train_all, label='validation loss')
plt.legend()
plt.show()

tar = test[:,3]
x_pred = torch.tensor(np.concatenate((test[:, 0:3], test[:, 4:22]), axis=1), dtype=torch.float32).to(device)
pred = model(x_pred)
l = [0, 1]
plt.plot(l, l)
plt.plot(pred.cpu().detach().numpy(), tar, '.')
plt.show()

test_pred_df = pd.DataFrame(data=test, columns=df.columns)

test_pred_df["price"] = pred.cpu().detach().numpy()

price_pred_real = scaler.inverse_transform(test_pred_df)[:, 3]

target = np.array(test_df['price'])

listOfNumbers = []

for x in range (0, 5):
    listOfNumbers.append(random.randint(1, len(target)))

l = [0, 2000000]
plt.plot(l, l)

for i in listOfNumbers:
    plt.plot(price_pred_real[i], target[i], '.')
plt.show()

print()