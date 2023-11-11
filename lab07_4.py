import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read data from the Excel file
df = pd.read_excel(r"C:\Users\anish\Downloads\embeddingsdata.xlsx")

# Extract data from the columns 'embed_0' and 'embed_1' for training
X = df[['embed_0', 'embed_1']]  # Features
y = df['Label']  
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create an SVM classifier with the current kernel function
    clf = svm.SVC(kernel=kernel)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    # Use the classifier to make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate and print accuracy for the current kernel
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with '{kernel}' kernel: {accuracy:.2f}")
