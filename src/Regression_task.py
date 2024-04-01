#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ----------------  Read and load data ----------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
get_ipython().run_line_magic('matplotlib', 'inline')

from google.colab import drive

drive.mount('/content/gdrive')

# Load the training and test data
#Load the training data from a CSV file located at '/content/gdrive/My Drive/semestre 6/DATA MINING/train.csv' and store it in the 'train_data' variable.
train_data = pd.read_csv('filepath') #in my case it was on google drive
#Load the test data from a CSV file located at '/content/gdrive/My Drive/semestre 6/DATA MINING/test.csv' and store it in the 'test_data' variable.
test_data = pd.read_csv('filepath') #in my case it was on google drive
display(train_data.head(3))
display(test_data.head(3))
train_data.isnull().sum()
combined_data = pd.concat([train_data, test_data], ignore_index=True)
numerical_columns = combined_data.select_dtypes(include=[np.number]).columns
correlation_matrix = combined_data[numerical_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


# In[ ]:


# ---------------- Data Preprocessing -------------------
train_data_imputed = train_data.fillna(train_data.mean())

vendor=preprocessing.LabelEncoder()
vendor.fit(train_data_imputed.Vendor_Name.unique())
train_data_imputed['Vendor_Name']=vendor.transform(train_data_imputed['Vendor_Name'])
model=preprocessing.LabelEncoder()
model.fit(train_data_imputed.Model_Name.unique())
train_data_imputed['Model_Name']=model.transform(train_data_imputed['Model_Name'])
train_data_imputed.corr() #Checking the correlation of the Model_Name and Vendor_Name columns

columns_to_drop = ['Y_ERP', 'Vendor_Name', 'Model_Name'] #Drop these columns since they don't have strong correlation with ERP
X_train = train_data_imputed.drop(columns_to_drop, axis=1)
y_train = train_data_imputed['Y_ERP']

display(y_train)
display(X_train)


# In[ ]:


# ---------------- Build training model -------------------
alpha = 1.0
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
print(f'Training R-squared: {train_score}')


# In[ ]:


# ---------------- Predict on testing data -------------------

columns_to_drop = ['Y_ERP', 'Vendor_Name', 'Model_Name'] #Drop these columns since they don't have strong correlation with ERP
X_test = test_data.drop(columns_to_drop, axis=1)
y_test = test_data['Y_ERP']

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.grid(True)
plt.show()

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')


# In[ ]:


print('Testing MSE: ', mse)

