# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function 
respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: G.Jayanth

RegisterNumber: 212221230030
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
1.Placement data

![image](https://user-images.githubusercontent.com/94836154/233605503-fe00bd81-665b-4867-8fe2-188859962237.png)

2.Salary data

![image](https://user-images.githubusercontent.com/94836154/233605616-bfb8d944-7fcf-4e6c-9b09-e0a1a4fc61c6.png)

3.Checking the null() function

![image](https://user-images.githubusercontent.com/94836154/233605756-468eb263-cde8-4a48-88d0-d0b379a87986.png)

4.Data Duplicate

![image](https://user-images.githubusercontent.com/94836154/233605927-a4abe1f2-da30-4699-b7e5-c4574bc5b68d.png)

5.Print data

![image](https://user-images.githubusercontent.com/94836154/233606083-c2ee4e66-fa96-487b-ae4e-1f5069e05431.png)

6.Data-status

![image](https://user-images.githubusercontent.com/94836154/233606247-37b0304f-4f05-4f06-893a-7ee6a199ce00.png)

![image](https://user-images.githubusercontent.com/94836154/233606323-6242dd31-4ea1-475f-b000-42c88eebd965.png)

7.y_prediction array

![image](https://user-images.githubusercontent.com/94836154/233606462-bbdc0513-dda5-475c-9096-537175122ee6.png)

8.Accuracy value

![image](https://user-images.githubusercontent.com/94836154/233606573-d87608ae-2a3d-45ca-9d6c-aaf7aed4b5c7.png)

9.Confusion array

![image](https://user-images.githubusercontent.com/94836154/233606886-1014e083-bdbe-4ed4-a582-d304193dc06c.png)

10.Classification report

![image](https://user-images.githubusercontent.com/94836154/233606972-298f4004-29dc-4d67-bb98-ac47042bdf51.png)

11.Prediction of LR

![image](https://user-images.githubusercontent.com/94836154/233607081-6c8dab89-ae11-4c5f-b401-22a534153dc3.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
