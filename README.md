# Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload the csv file and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeRegressor.
5. Import metrics and calculate the Mean squared error.
6. Apply metrics to the dataset, and predict the output.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by : Nithilan S
RegisterNumber : 212223240108
```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2 = metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data Head

![image](https://github.com/user-attachments/assets/6bcf15d7-f9f4-4863-9b9c-321195ed5cc3)

## Data Info

![image](https://github.com/user-attachments/assets/49df6c3e-bfd3-49dc-94f1-98e095907017)

## Data Head after applying LabelEncoder():

![image](https://github.com/user-attachments/assets/b7a16346-712b-48de-a17a-33792d339bcd)

## MSE:

![image](https://github.com/user-attachments/assets/f262b6c0-0c16-46e0-b376-89be11acbde0)

## r2:

![image](https://github.com/user-attachments/assets/d9a2c1a3-4a1c-4418-8bf6-9230e4722e9a)

## Data Prediction:

![image](https://github.com/user-attachments/assets/869357a9-85a4-4056-92cb-0b38653663df)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
