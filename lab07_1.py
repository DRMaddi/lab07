import numpy as np
import pandas as pd
from sklearn import svm

# Read data from the Excel file
df = pd.read_excel(r"C:\Users\anish\Downloads\embeddingsdata.xlsx")

# Extract data from the columns 'embed_0' and 'embed_1'
X_train = df[['embed_0', 'embed_1']]  # Features
y_train = df['Label']  # Labels, replace 'class_column' with the actual column name in your dataset

# Create an SVM classifier
clf = svm.SVC()

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Get the support vectors
support_vectors = clf.support_vectors_

# Now you can study the support vectors
print("Support Vectors:")
print(support_vectors)
