#!/usr/bin/env python
# coding: utf-8

# In[126]:

# Importing all necessary Modules
import pandas as pd
import csv
import sys
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import itertools
from scipy.stats import describe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'darkgrid')


# ### Part 1: Modeling Challenge

# * Step 1: Read in the dataframe
# * Step 2: Add the columns to the dataframe

# In[74]:


data_columns = pd.read_csv('./data/field_names.txt', sep='\n', header=None)

col_val = []
for i in range(len(data_col_val)):
    col = data_col_val.iloc[[i]].values[0][0]
    col_val.append(col)


# In[75]:


df = pd.read_csv('./data/breast-cancer.csv')
df.col_val = col_val


# In[76]:


df.head(5)


# In[77]:


df.info()


# In[78]:


feature_mean_value= list(df.col_val[1:11])
feature_standard_err= list(df.col_val[11:20])
features_worst=list(df.col_val[21:31])
print(feature_mean_value)
print("....................................")
print(feature_standard_err)
print("....................................")
print(features_worst)


# In[79]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
df.describe()


# In[80]:


sns.countplot(df['diagnosis'],label="Count")


# In[81]:


correlation=df[feature_mean_value].correlation()
plt.figure(figsize=(14,14))
sns.heatmap(correlation,cbar=True,square=True,annot=True,fmt='.2f',annot_kws={'size': 15},xticklabels=feature_mean_value,yticklabels=feature_mean_value,cmap='coolwarm')


# In[82]:


pred_var=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
train,test=train_test_split(df,test_size=0.3)
print(train.shape)
print(test.shape)


# In[83]:


train_X=train[pred_var]
train_y=train.diagnosis
test_X=test[pred_var]
test_y=test.diagnosis


# In[84]:


model=RandomForestClassifier(n_estimators=100)


# In[85]:


model.fit(train_X,train_y)


# In[86]:


prediction=model.predict(test_X)


# In[87]:


prediction=model.predict(test_X)


# In[88]:


metrics.accuracy_score(prediction,test_y)


# In[89]:


## Now lets try the same problem with Neural Network Approach 


# In[90]:


df.head()


# In[91]:


df.shape


# In[92]:


""""
The data has 568 rows, each representing a single patient, with 28 measurements each (note: there are 32 col_val, but column 1 is an ID, column 2 is a diagnosis, and the 32nd is not a measurement)

We are now going to proceed by massaging the data. The following steps will be taken to prepare the data for our neural network:

Replace the diagnosis of "M" for malignant with 1, and "B" for benign with 0
Remove the 32nd column (not sure why this is included, it is a column of 'NaN' values. Perhaps this was erroneously included in the dataset)
Remove the 'id' column. These values will not be needed to create our neural network.
Split our dataset into two: a calssification vector (diagnosis) and our feature matrix (the 28 measurements)
Normalize our feature matrix by forcing the mean of each measurement to 0, and dividing each measurement by the maximum value of that measurement in the dataset.
"""


# In[96]:


data=df


# In[98]:


X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[99]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[100]:


# Initialising the ANN
classifier = Sequential()


# In[101]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# In[102]:


# Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))


# In[103]:


# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# In[104]:


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[105]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=150)
# Long scroll ahead but worth
# The batch size and number of epochs have been set using trial and error. Still looking for more efficient ways. Open to suggestions. 


# In[ ]:


Batch size defines number of samples that going to be propagated through the network.

An Epoch is a complete pass through all the training data.


# In[106]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[107]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[108]:


print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))


# In[109]:


sns.heatmap(cm,annot=True)
plt.savefig('h.png')


# In[ ]:


# Try with Support Vector Machine


# In[110]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[111]:


bc=df


# In[115]:


bcs = pd.DataFrame(preprocessing.scale(bc.ix[:,2:32]))
bcs.col_val = list(bc.ix[:,2:32].col_val)
bcs['diagnosis'] = bc['diagnosis']


# In[116]:


from pandas.tools.plotting import scatter_matrix
p = sns.PairGrid(bcs.ix[:,20:32], hue = 'diagnosis', palette = 'Reds')
p.map_upper(plt.scatter, s = 20, edgecolor = 'w')
p.map_diag(plt.hist)
p.map_lower(sns.kdeplot, cmap = 'GnBu_d')
p.add_legend()

p.figsize = (30,30)


# In[117]:


mbc = pd.melt(bcs, "diagnosis", var_name="measurement")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set2');
p.set_xticklabels(rotation = 90, labels = list(bcs.col_val));


# In[120]:


X = bcs.ix[:,0:30]

y = bcs['diagnosis']
class_names = list(y.unique())


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[124]:


# Model score 
svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(X,y)
y_pred = svc.fit(X_train, y_train).predict(X_test)
t = pd.DataFrame(svc.predict_proba(X_test))
svc.score(X_train,y_train), svc.score(X_test, y_test)


# In[127]:


mtrx = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(mtrx,classes=class_names,title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(mtrx, classes=class_names, normalize = True, title='Normalized confusion matrix')

plt.show()


# In[ ]:




