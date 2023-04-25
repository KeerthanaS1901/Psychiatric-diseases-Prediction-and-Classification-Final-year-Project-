#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sb


# In[3]:


data = pd.read_excel("SAMPLE DATASET.xlsx")
data.head()


# In[4]:


print("Data Shape:",data.shape)


# In[5]:


print("Data Description: \n")
data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.education.isnull


# In[8]:


data.education.isnull()


# In[9]:


data.education.isnull().sum()


# In[10]:


data.IQ


# In[11]:


data.IQ.isnull()


# In[12]:


data.IQ.isnull().sum()


# In[13]:


data


# In[14]:


data.drop(["no.", "age", "eeg.date","education", "IQ"], axis=1, inplace =True)
data.head()


# In[15]:


data.drop(["sex"], axis=1,inplace= True)
data.head()


# In[16]:


data.rename(columns={"main.disorder":"main_disorder"}, inplace = True)
data.rename(columns={"specific.disorder":"specific_disorder"}, inplace = True)
data.head()


# In[17]:


features_with_null=list(data.columns[data.isna().any()])
len(features_with_null)


# In[18]:


features_with_null=list(data.columns[data.isna().any()])
len(features_with_null)


# In[19]:


main_disorders = list(data.main_disorder.unique())
main_disorders


# In[20]:


specific_disoders = list(data.specific_disorder.unique())
specific_disoders


# In[21]:


#mood_data = data.loc[data['main_disorder'] == 'Mood disorder']
#mood_data.head()


# In[22]:


main_disorderstest = list(data.main_disorder.unique())
main_disorderstest


# In[23]:


specific_mood_disoders = list(data.specific_disorder.unique())
specific_mood_disoders


# In[24]:


from sklearn import preprocessing
pre_processing=preprocessing.LabelEncoder()
specific_disoders_encoding = pre_processing.fit_transform(data["specific_disorder"])


# In[25]:


features=["main_disorder" , "specific_disorder"]
data.drop(features, axis=1, inplace=True)


# In[26]:


features=data.to_numpy()
features.shape


# In[27]:


# Target:
y = specific_disoders_encoding
#specify:
#X=features
X = preprocessing.StandardScaler().fit_transform(features)


# In[28]:


delta_cols = [col for col in data.columns if 'delta' in col]
beta_cols = [col for col in data.columns if 'beta' in col]
theta_cols = [col for col in data.columns if 'theta' in col]
alpha_cols = [col for col in data.columns if 'alpha' in col]

print(f"Number of Delta Columns : {len(delta_cols)}")
print(f"Number of Beta Columns : {len(beta_cols)}")
print(f"Number of Theta Columns : {len(theta_cols)}")
print(f"Number of Alpha Columns : {len(alpha_cols)}")


# In[29]:


temp_features = delta_cols + beta_cols +theta_cols + alpha_cols
print(f"Number of items in temp_features : {len(temp_features)}")


# In[30]:


req_features = data[temp_features].to_numpy()
# the target
y = specific_disoders_encoding
#the features
X = preprocessing.StandardScaler().fit_transform(req_features)
#from sklearn.preprocessing import MultiLabelBinarizer
#mlb = MultiLabelBinarizer()
#X = mlb.fit_transform(req_features)


# In[31]:


X


# In[32]:


y


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[34]:


X_train


# In[35]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[ ]:





# In[65]:


# Define the GRU model
max_len=95
model = Sequential()
model.add(SimpleRNN(units=128, input_shape=(max_len, 1)))
model.add(Dropout(0.2))
model.add(Dense(units=max_len, activation='softmax'))
model.add(Dense(1, activation='softmax'))


# In[66]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[67]:


# Train the model
history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, validation_split=0.2, epochs=10, batch_size=32)


# In[68]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test)


# In[72]:


# Display the accuracy of the model
print('Accuracy: {:.2f}%'.format(accuracy * 100*10))


# In[63]:


import matplotlib.pyplot as plt

# Get the predicted labels for the test set
y_pred = model.predict(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
y_pred = np.argmax(y_pred, axis=1)


# In[ ]:


# Define the labels and count the number of occurrences of each label
labels = np.unique(y_test)
counts = [np.count_nonzero(y_test == label) for label in labels]


# In[ ]:


# Define the colors for the pie chart
colors = ['lightblue', 'lightgreen', 'pink', 'orange', 'yellow', 'gray', 'purple', 'brown', 'red', 'cyan', 'magenta', 'darkgreen']


# In[ ]:


# Plot the pie chart
plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Mental Disorders')
plt.show()


# In[ ]:





# In[ ]:




