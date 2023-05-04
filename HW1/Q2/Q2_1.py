import numpy as np
import matplotlib.pyplot as plt

def activation_function(net):
    return 1 if net >= 0 else -1

def checker():
    response = False
    sum_errors = 0
    for p in train_set:
        x1 , x2 , t = p["x1"] , p["x2"] , p["t"]
        net = w1*x1 + w2*x2 + b
        h = activation_function(net)
        error_p = 0.5*(t - h)**2
        if error_p != 0:
            response = True
        sum_errors += error_p 
    sum_errors_epochs.append(sum_errors)
    return response

np.random.seed(seed=3)
w1 = np.random.rand()
w2 = np.random.rand()
b = np.random.rand()
lr = 0.001
sum_errors_epochs = []
train_set = list()

n1=100
#x , y = np.random.normal(loc=-1,scale=0.3,size=n1) , np.random.normal(loc=-1,scale=0.3,size=n1)
x , y = np.random.normal(loc=0,scale=0.6,size=n1) , np.random.normal(loc=0,scale=0.6,size=n1)
plt.scatter(x,y)
for i in range(n1):
    train_set.append({"x1" : x[i], "x2" :y[i], "t" : +1})

#n2=100
#x , y = np.random.normal(loc=1,scale=0.3,size=n2) , np.random.normal(loc=1,scale=0.3,size=n2)
n2=20
x , y = np.random.normal(loc=2,scale=0.8,size=n2) , np.random.normal(loc=2,scale=0.8,size=n2)
plt.scatter(x,y)
for i in range(n2):
    train_set.append({"x1" : x[i], "x2" :y[i], "t" : -1})

epoch = 0

while checker():
    epoch+=1
    for p in train_set:
        x1 , x2 , t = p["x1"] , p["x2"] , p["t"] 
        net = w1*x1 + w2*x2 + b
        h = activation_function(net)
        w1 = w1 + lr*(t-h)* x1
        w2 = w2 + lr*(t-h)* x2
        b  = b  + lr*(t-h)

plt.title("Adaline")
x = np.linspace(-1,4)
label = 'Y = '+str(round(-w1/w2,3))+' * X + '+str(round(-b/w2,3))+'\nw1 = '+str(round(w1,3))+' ,w2 = '+str(round(w2,3))+' ,b = '+str(round(b,3))
plt.text(0.5, 4, label, fontsize = 9)
plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.plot(x,(-w1/w2)*x-b/w2,'-r')
plt.xlabel("X")
plt.ylabel("Y",rotation=0)
plt.show()
plt.title("Adaline Loss")
plt.plot(sum_errors_epochs)
plt.xlabel("Epoch")
plt.ylabel("Loss Function")
plt.show()