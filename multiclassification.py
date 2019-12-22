#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


def sigmoid(Z, w):   
    d = np.dot(Z, w)
    return 1 / (1 + np.exp(-d))


# In[12]:


def gradient_descent(Z, yp, y,lmda,w):
    return np.dot(Z.T, (yp - y)) / len(y) + lmda*w


# In[13]:


def update_weight_loss(w, alpha, G):
    return w - alpha * G


# In[14]:


def extendOne(X):
    extender = np.ones((X.shape[0], 1)) 
    return np.hstack((extender, X))


# In[15]:


df = pd.read_csv(r'C:\Users\Isra Shafique\Downloads\iris.csv', header=None, names=[
    "Sepal length (cm)", 
    "Sepal width (cm)", 
    "Petal length (cm)",
    "Petal width (cm)",
    "Species"
])
df.head()


# In[16]:


df['Species'] = df['Species'].astype('category').cat.codes
data = np.array(df)
np.random.shuffle(data)
num_train = int(.8 * len(data))  # 80/20 train/test split
x_train, tr_y = data[:num_train, :-1], data[:num_train, -1]
x_test, y_test = data[num_train:, :-1], data[num_train:, -1]


# In[ ]:


num_iter = 100000
tr_XHat = extendOne(x_train)
tr_y=tr_y.reshape(-1,1)
# In[ ]:
w = np.zeros((tr_XHat.shape[1], 1))

lmda = 0
alpha = 0.1
ws=[]

classes=np.unique(tr_y)
for c in classes:
    bin_y=np.where(c==tr_y,1,0)
    for i in range(num_iter):
        yp = sigmoid(tr_XHat, w)
        G = gradient_descent(tr_XHat, yp, bin_y,lmda,w)
        w = update_weight_loss(w, alpha, G)
    ws.append(w)


p=np.zeros(3)
x2=np.array([5.4,3.7,1.5,0.2])
x2=x2.reshape(-1,1)
x2=x2.T
i=0;
for w1 in ws:
    p[i]=sigmoid(extendOne(x2),w1)
    i=i+1
result=np.max(p);
print(result)
ans = np.where(p == result)
print(ans)

# In[ ]:



    

