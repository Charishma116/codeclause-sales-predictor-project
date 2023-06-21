#!/usr/bin/env python
# coding: utf-8

# In[53]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot


# In[54]:


files =[file for file in os.listdir(r"C:\Users\karri\Downloads\archive")]
for file in files:
    print(file)


# In[55]:


path = r"C:\Users\karri\Downloads\archive"
files = os.listdir(path)

all_data = pd.DataFrame()

for file in files:
    if file.endswith(".csv"):
        current_df = pd.read_csv(os.path.join(path, file))
        all_data = pd.concat([all_data, current_df])

print(all_data.shape)


# In[56]:


all_data.to_csv(r'C:\Users\karri\Downloads\archive\all_data.csv', index=False)


# In[57]:


all_data.dtypes


# In[58]:


all_data.head()


# In[59]:


all_data.isnull().sum()


# In[60]:


all_data = all_data.dropna(how='all')
all_data.shape


# In[61]:


'male'.split('/')[0]


# In[62]:


def month(x):
    return x.split('/')[0]


# In[63]:


import pandas as pd

# Assuming you have a DataFrame called all_data that contains supermarket sales data
# Replace the data below with your own supermarket sales data
data = {
    'Invoice ID': ['INV001', 'INV002', 'INV003'],
    'Branch': ['A', 'B', 'C'],
    'City': ['City A', 'City B', 'City C'],
    'Customer type': ['Regular', 'VIP', 'Regular'],
    'Gender': ['Male', 'Female', 'Male'],
    'Product line': ['Electronics', 'Clothing', 'Home and Garden'],
    'Unit price': [100, 50, 80],
    'Quantity': [2, 3, 1],
    'Tax 5%': [10, 7.5, 4],
    'Total': [110, 57.5, 84],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'Time': ['08:30:00', '13:15:00', '18:45:00'],
    'Payment': ['Card', 'Cash', 'Card'],
    'cogs': [100, 50, 80],
    'gross margin percentage': [0.05, 0.1, 0.08],
    'gross income': [10, 7.5, 4],
    'Rating': [4, 5, 3]
}

all_data = pd.DataFrame(data)

# Convert the "Date" column to datetime format
all_data['Date'] = pd.to_datetime(all_data['Date'])

# Extract month information and assign it to the new "Month" column
all_data['Month'] = all_data['Date'].dt.month

# Print the updated DataFrame
print(all_data)


# In[64]:


# Read the CSV file and assign it to the df variable
df = pd.read_csv(r'C:/Users/karri/Downloads/archive/all_data.csv')



# Create the figure with subplots and boxplots
fig, axs = plt.subplots(3, figsize=(5, 5))
plt1 = sns.boxplot(df['Total'], ax=axs[0])
plt2 = sns.boxplot(df['Quantity'], ax=axs[1])
plt3 = sns.boxplot(df['Rating'], ax=axs[2])
plt.tight_layout()
plt.show()


# In[65]:


sns.countplot(df['gross margin percentage'])
plt.show()


# In[66]:


sns.countplot(df['Unit price'])
plt.show()


# In[67]:


sns.pairplot(df, x_vars=['Total', 'Quantity', 'Rating'], y_vars='Product line', height=4, aspect=1, kind='scatter')
plt.show()


# In[68]:


correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()


# In[69]:


X = df['Unit price']
y = df['Total']



# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[72]:


X_train.head()


# In[73]:


y_train.head()


# In[74]:


import statsmodels.api as sm


# In[75]:


X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()


# In[76]:


print(lr.params)
print('summary:')
print(lr.summary())


# In[77]:


C = 10.2
M = 0.08


# In[78]:


plt.scatter(X_train, y_train)
plt.plot(X_train, C + M*X_train, 'r')
plt.show()


# In[79]:


X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)


# In[80]:


y_pred.head()


# In[81]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[82]:


import numpy as np
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


# In[83]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[84]:


plt.scatter(X_test, y_test)
plt.plot(X_test, C + M * X_test, 'r')
plt.show()
print('Hence,Increase in sales is predicted')


# In[ ]:





# In[ ]:





# In[ ]:




