import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize

movies_df = pd.read_csv('./movies.csv')
print(movies_df.head())
print(movies_df.tail())

ratings_df = pd.read_csv('./ratings.csv')
print(ratings_df.head())
print(ratings_df.tail())

print('shape of movies:',movies_df.shape)
print('shape of ratings:',ratings_df.shape)

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# print(movies_df.head())
# print(ratings_df.head())
# print('The Number of Movies in Dataset', len(movies_df))

movies_df['List Index'] = movies_df.index
print(movies_df.head())

movieslist = movies_df['Title']
merged_df = movies_df.merge(ratings_df, on='MovieID')
print(merged_df.head())

merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)
print(merged_df.head())

user_Group = merged_df.groupby('UserID')
print(user_Group.head())

train_x = list()
for userID, curUser in user_Group:
    temp = [0]*len(movies_df)
    for num, movie in curUser.iterrows():
        temp[int(movie['List Index'])] = movie['Rating']/5.0
    train_x.append(temp)

#train_x = normalize(train_x, axis=1, norm='l1')
train_x=torch.tensor(train_x, dtype=torch.float32)

#mean, std = torch.mean(train_x), torch.std(train_x)
#train_x  = (train_x-mean)/std
#import torch.nn.functional as f
#train_x = f.normalize(train_x, p=2, dim=0)

class data_delivery():
    def __init__(self,data):
        self.dataset = data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return torch.tensor(self.dataset[idx], dtype=torch.float32)

train_loader = data_delivery(train_x)

class RBM():
    def __init__(self, nv, nh):
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        self.W = torch.randn(nh, nv)

    def forward(self, x):
        wx = torch.mm(x, self.W.t())
        return torch.sigmoid(wx + self.a.expand_as(wx))
        
    def backward(self, y):
        wy = torch.mm(y, self.W)
        return torch.sigmoid(wy + self.b.expand_as(wy))

    def train(self, v0, vk, ph0, phk, eps):
        self.a += eps*(torch.sum((ph0 - phk), 0))
        self.b += eps*(torch.sum((v0 - vk), 0))
        self.W += eps*((torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t())

eps = 0.01
batch_size = 100
nh = 20
nv = len(train_x[0])
rbm = RBM(nv, nh)

num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train_loss = 0
    s = 0
    for id in range(0, len(train_x) - batch_size, batch_size):
        vk = train_x[id : id + batch_size]
        v0 = train_x[id : id + batch_size]
        ph0 = rbm.forward(v0)
        for k in range(num_epochs):
            hk = rbm.forward(vk)
            vk = rbm.backward(hk)
            vk[v0<0] = v0[v0<0]
        phk = rbm.forward(vk)
        rbm.train(v0, vk, ph0, phk, eps)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1
    print('Epoch: '+str(epoch))
    print('Loss: ' +str((train_loss/s).item()))

test_loss = 0

v = torch.tensor(train_x[75], dtype=torch.float32).unsqueeze(0)
vt = torch.tensor(train_x[75], dtype=torch.float32).unsqueeze(0)

h = rbm.forward(v)
v = rbm.backward(h)

test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
print('Loss: '+str((test_loss/s).item()))

#Print recommendations for user 75
v = v.squeeze(0)
sorted, idx = torch.sort(v)
print('Recommended Movies for User 75:\n')
for i in range(1, 16):
    print(movieslist[np.array(idx[-i])])