# Model that classifies handwritten Alphabet letters 
# (only upper case SO FAR!)
# Transported from https://colab.research.google.com/drive/1btTjfee2dB9Vs3Elb4CwO3wg9UilfiUk#scrollTo=UzWEBmOd8X7I

# TensorFlow and tf.keras
import tensorflow as tf

# Tensorflow JS for web apps
import tensorflowjs as tfjs

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# Pandas dataframess
import pandas as pd

# Create dataframe using dataset
df = pd.read_csv('A_Z Handwritten Data.csv')

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

#PREPROCESS THE DATA
# plt.figure()
# #reshape Series back into 28 x 28
# reshaped = train_images.iloc[0].values.reshape(28, 28)
# plt.imshow(reshaped)
# plt.colorbar()
# plt.grid(False)
# plt.show()

#Scale data all to range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

newTrain = train_images.astype('float32')
newTest = test_images.astype('float32')


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images.iloc[i].values.reshape(28,28), cmap=plt.cm.binary)
#     #iloc returns either series/dataframe, item() returns value
#     plt.xlabel(class_names[train_labels.iloc[i].item()]) 
# plt.show()

#MODEL TIME
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26) #26 output nodes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(newTrain, train_labels, epochs=10)

# Converting to TF.js : https://www.tensorflow.org/js/tutorials/conversion/import_keras
tfjs.converters.save_keras_model(model, pathlib.Path().absolute())

#EVALUATING ACCURACY
test_loss, test_acc = model.evaluate(newTest,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#MAKING PREDICTIONS
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(newTest)

print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels.iloc[0].item())

def plot_image(i, predictions_array, true_label, img):
  #true_label, img = true_label.iloc[i].item(), img.iloc[i].values.reshape(28,28)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  # true_label = true_label.iloc[i].item()
  plt.grid(False)
  plt.xticks(range(26))
  plt.yticks([])
  thisplot = plt.bar(range(26), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  # Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels.iloc[i].item(),
#              test_images.iloc[i].values.reshape(28,28))
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels.iloc[i].item())
# plt.tight_layout()
# plt.show()


#USE THE TRAINED MODEL!!
# i = 1200

# #Get image from test datset
# img = test_images.iloc[i]
# img = (np.expand_dims(img,0))
# predictions_single = probability_model.predict(img)
# # print(predictions_single)

# print(test_labels.iloc[i].item())
# print("Expected ", class_names[test_labels.iloc[i].item()])
# print(np.argmax(predictions_single[0]))
# print("Got ", class_names[np.argmax(predictions_single[0])])

# plot_value_array(i, predictions_single[0], test_labels.iloc[i].item())
# _ = plt.xticks(range(26), class_names, rotation=45)

# hist = model.fit(newTrain, train_labels, epochs=10)
# plt.figure()
# plt.ylabel("Loss (training and validation)")
# plt.xlabel("Training Steps")
# plt.ylim([0,2])
# plt.plot(hist["loss"])
# plt.plot(hist["val_loss"])

# plt.figure()
# plt.ylabel("Accuracy (training and validation)")
# plt.xlabel("Training Steps")
# plt.ylim([0,1])
# plt.plot(hist["accuracy"])
# plt.plot(hist["val_accuracy"])