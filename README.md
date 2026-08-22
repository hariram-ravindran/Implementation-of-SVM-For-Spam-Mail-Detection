# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the SMS/email dataset containing messages and their labels (spam or ham).

2.Convert the text messages into numerical features using TF-IDF Vectorization.

3.Split the dataset into training and testing data.

4.Train an SVM (Support Vector Machine) classifier using the training data.

5.Predict whether new messages are Spam or Ham and evaluate the model using accuracy.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:HARIRAM R
RegisterNumber:212224240050 
*/
# Exp 11 – SVM for Spam Mail Detection with Visualization

# 1. Import Required Libraries
import chardet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Detect File Encoding
file_path = "spam.csv"   # use "/content/spam.csv" for Colab

with open(file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print("Detected Encoding:", result)

# 3. Load Dataset
data = pd.read_csv(file_path, encoding=result['encoding'])

# 4. Basic Data Exploration
print(data.head())
print(data.info())
print(data.isnull().sum())

# 5. Visualization: Spam vs Ham Distribution
plt.figure(figsize=(5,4))
sns.countplot(x=data['v1'])
plt.title("Distribution of Spam and Ham Messages")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()

# 6. Message Length Visualization
data['msg_length'] = data['v2'].apply(len)

plt.figure(figsize=(6,4))
sns.histplot(data=data, x='msg_length', hue='v1', bins=50, kde=True)
plt.title("Message Length Distribution (Spam vs Ham)")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()

# 7. Feature and Target Selection
x = data['v2'].values     # messages
y = data['v1'].values     # labels

# 8. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# 9. Text Vectorization (Bag of Words)
cv = CountVectorizer()
x_train_vec = cv.fit_transform(x_train)
x_test_vec = cv.transform(x_test)

# 10. Initialize and Train SVM
svc = SVC(kernel='linear')
svc.fit(x_train_vec, y_train)

# 11. Prediction
y_pred = svc.predict(x_test_vec)

# 12. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 13. Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix – SVM Spam Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
```

## Output:
<img width="613" height="493" alt="image" src="https://github.com/user-attachments/assets/c2bdea98-581b-4389-a9c6-e258c3c37c94" />
<img width="660" height="475" alt="Screenshot 2026-08-22 153953" src="https://github.com/user-attachments/assets/db7347bb-ff74-42d3-9c04-105bc168af0b" />
<img width="718" height="741" alt="Screenshot 2026-08-22 154002" src="https://github.com/user-attachments/assets/d5f8d88c-9797-4309-b08b-ce896de47fc3" />
<img width="613" height="493" alt="Screenshot 2026-08-22 154007" src="https://github.com/user-attachments/assets/989cc438-244c-4a6e-b800-2e30a827dae7" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
