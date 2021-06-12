# Model that classifies handwritten Alphabet letters 
# (only upper case SO FAR!)
# Transported from https://colab.research.google.com/drive/1btTjfee2dB9Vs3Elb4CwO3wg9UilfiUk#scrollTo=UzWEBmOd8X7I

# TensorFlow and tf.keras
import tensorflow as tf

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Pandas dataframes
import pandas as pd

print(pd.__version__)

# Create dataframe using dataset
df = pd.read_csv('A_Z Handwritten Data.csv')
print(df)

#0-25 = labels, so 0 = A, 25 = Z
#First column = label column
#Rest of columns, 28 x 28 = 784, pixel image value

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


#Dividing dataframe into 80% training, 20% testing
train = df.sample(frac=0.8, random_state = 200)
test = df.drop(train.index).sample(frac=1.0)

#df.iloc[row_start:row_end , col_start: col_end]
#Splitting data into labels, images
train_labels = train.iloc[:, :1]
train_images = train.iloc[:, 1:]

test_labels = test.iloc[:, :1]
test_images = test.iloc[:, 1:]

#dataset dimensions: (372450, 785)
#train_labels: (297960, 1)
#train_images: (297960, 784)