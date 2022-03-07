# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:10:24 2020

@author: admin
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix



df = pd.read_csv('bank1.csv')
df.head()

df.info()
# age , salary, balance in respective datatypes

df.housing.value_counts()
# 3 types found 'YES', 'yes', 'no'

df.housing.replace('YES', 'yes', inplace = True)
# df.loc[df.housing == 'YES'] = 'yes'  
# Dtypes of other feature gets changed to obj. why?

df.isna().any()
# salary has nan values

df.salary.isna().sum()  #12 Nan(not a number) values

df.loc[df.salary.isna()]

# NaN value treatment
# bfill, ffill, replace with 0

# fill it with median value of balance where salary is nan
#  mean and median salary of costumers where -330<balance<3240
temp = df.loc[(df['balance'] < 3240) & (df.balance >= -333)]
temp.salary.median() # 60,000
mean_for_sal__nan = temp.salary.mean() # 56,475


df.salary.fillna(mean_for_sal__nan, inplace = True)
# NaN value Treated successfuly!!!!!

df.salary.max() # 120000
df.salary.min() # 0 .... Not possible!!

# replacement of 0 with mean or median 
zero_sal = df.loc[df.salary == 0] # 288 values
zero_sal.balance.mean() # 1772.3
median_for_zero = zero_sal.balance.median()# 677
zero_sal.balance.mode()[0] # 0.... obv!!!

zero_sal.targeted.value_counts() # equally targeted
zero_sal.housing.value_counts() # 262<- no, 26<- yes
# housing is a factor
zero_sal.default.value_counts() # only 2 defaulters
df.salary.replace(0,median_for_zero , inplace = True)

# now minimum salary is 677.0!!!!!!!!

df.describe()

sns.boxplot(x = df.salary ,data = df, orient = 'vertical')
df = df.drop('contact', axis = 1)
sns.heatmap(df.corr() , annot = True)
# does not show relation of categorical values

# create dummies(one-hot encoding) as knn takes float values not object
df.job.unique()
df1 = pd.get_dummies(df.job)
##########################################################
# label encoding
df.head()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
category_x = df[['job', 'marital', 'education', 'targeted', 'housing']]
df = df.drop(['job', 'marital', 'education', 'targeted', 'housing'], axis =  1)
df = df.drop('contact', axis = 1)
le = LabelEncoder
for i in range(len(category_x.iloc[0])):
    category_x.iloc[:, i] = le.fit_transform(category_x.iloc[:, i])
len(category_x.iloc[0])
category_x = category_x.apply(LabelEncoder().fit_transform)
df = pd.concat([df, category_x], axis = 1)


###########################################################3

# KNN 
X = df.drop('default', axis = 1)
y = df['default']
y.value_counts()
print(X, '\n', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 200 )


knn = KNeighborsClassifier(n_neighbors=5, weights= 'uniform', metric= 'euclidean')

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Classification report \n', classification_report(pred, y_test))
acc = accuracy_score(pred, y_test)

 #             precision    recall    f1-score   support

 #        no       1.00      0.98      0.99     13550
 #       yes       0.01      0.21      0.02        14
#

































