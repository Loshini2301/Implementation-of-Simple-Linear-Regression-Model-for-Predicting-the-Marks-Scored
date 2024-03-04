# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## NAME:LOSHINI.G
## DEPARTMENT:IT
## REFERENCE NUMBER:212223220051
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries
2. Set variables for assigning dataset values
3. Import linear regression from sklearn
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: LOSHINI.G
RegisterNumber: 212223220051 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
print(df)
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
print(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,regressor.predict(X_train),color="green")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/Loshini2301/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150007305/7aa2f1ef-a4e9-4c36-8099-8fc3c31f7b4c)

![Screenshot 2024-03-04 202626](https://github.com/Loshini2301/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150007305/2d430058-4ed1-4c89-be56-5b27f98090de)
![Screenshot 2024-03-04 202641](https://github.com/Loshini2301/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/150007305/e178b1c8-271b-4a52-8dd0-a3bc5c0d716d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
