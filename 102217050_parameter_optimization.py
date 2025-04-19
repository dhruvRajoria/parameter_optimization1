#!/usr/bin/env python
# coding: utf-8

# ###Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')


# ###Uploading the Dataset from UCI Library

# In[ ]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
df.head()


# In[ ]:


dataset = df
dataset.head()


# ###Pre-Processing of Dataset

# In[ ]:


dataset.shape


# In[ ]:


dataset.isnull().sum()


# In[ ]:


# dataset = dataset.drop(['Date','Time'],axis=1)


# In[ ]:


sns.countplot(x = 'quality', data=dataset)


# In[ ]:





# In[ ]:


X = dataset.iloc[:,0:-1]
y = dataset['quality']


# In[ ]:


ss = StandardScaler()
ss.fit_transform(X)


# ###Creation of 10 samples with 70-30 ratio of Training and Testing Set

# In[ ]:


samples = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    samples.append((X_train, X_test, y_train, y_test))


# In[ ]:


samples


# ### Creation of a Fitness Function

# In[ ]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
result = pd.DataFrame(columns=['Sample', 'Best Accuracy', 'Best Kernel', 'Best Nu', 'Best Epsilon'])


# In[ ]:


def fitnessFunction(kernel, C, gamma):
  svm = SVC(kernel=kernel, C=C, gamma=gamma, max_iter=1000)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)
  return accuracy_score(y_pred, y_test)


# ### Calling the Function for Every Sample

# In[ ]:


for i in range(len(samples)):
  best_accuracy = 0
  best_C = 0
  best_gamma = 0
  for kernel in kernels:
    X_train, X_test, y_train, y_test = samples[i]
    C = np.random.uniform(0, 10)
    gamma = np.random.uniform(0, 10)
    score = fitnessFunction(kernel, C, gamma)
    if score>best_accuracy:
      best_accuracy = round(score, 2)
      best_C = round(C, 2)
      best_gamma = round(gamma, 2)
      best_kernel = kernel
  print('Best Accuracy = ', best_accuracy, 'Best Kernel = ', best_kernel, 'Best Nu = ', best_C, 'Best Epsilon = ', best_gamma)
  result.loc[i] = [i+1, best_accuracy, best_kernel, best_C, best_gamma]


# ###Creation of a Result Table

# In[ ]:


result


# ### Plotting of the Convergence Graph/Linear Curve

# In[ ]:


X_train, X_test, y_train, y_test = samples[result['Best Accuracy'].idxmax()]


# In[ ]:


train_sizes, train_scores, test_scores = learning_curve(SVC(kernel=result['Best Kernel'].iloc[result['Best Accuracy'].idxmax()],
                                                        C=result['Best Nu'].iloc[result['Best Accuracy'].idxmax()],
                                                        gamma=result['Best Epsilon'].iloc[result['Best Accuracy'].idxmax()],
                                                        max_iter = 1000), X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1,
                                                        train_sizes = np.linspace(0.01, 1.0, 50))


# In[ ]:


train_sizes


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)


# In[ ]:


plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, test_mean, label='Cross-Validation Score')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Convergence Graph')
plt.legend(loc="best")
plt.show()


# In[ ]:




