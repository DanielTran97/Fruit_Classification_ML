import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os, os.path
import random
import cv2

from PIL import Image
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

#suppress verbose Tensorflow logging
tf.get_logger().setLevel('ERROR')

#Get image from folders 1-4, then read through each images from each folder.
def get_data(path):
    for file in os.listdir(path):
        if file[0] == '.': #Ignore .VSCODE file 
            continue
        #Read images from file into memory
        img = cv2.imread("{}/{}".format(path, file))
        #Convert cv2 BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Resize images to 64 x 64
        img_resized = cv2.resize(img, dsize=(64,64))
        try:
            #Perform concatenate on the image
            x_train = np.concatenate((x_train, img_resized))
        except:
            x_train = img_resized
    #Allow numpy to computes the number of rows         
    return np.reshape(x_train, (-1, 64, 64, 3))

#preparing image by folders 
'''
Folder #
1   ->  apple
2   ->  banana
3   ->  orange
4   ->  mixed
'''
def image_prep(filename):
    for i in range(len(filename)):
        data = get_data(filename[i])
        #encode the folder class with onehot-encoding
        y_onehots = encode_onehot(i, data.shape[0])
        if i == 0:
            x = data
            y = y_onehots
        else:
            x = np.concatenate((x, data))
            y = np.concatenate((y, y_onehots))
    #Randomize and permutate the data so that when fitting the model, validation_split will not take the same classified data for validation.
    randomizer = np.random.permutation(len(x))
    x = x[randomizer]
    y = y[randomizer]
    return x, y

#Performs onehot-encoding for each output class
def encode_onehot (pos, n_rows):
    y_onehot = [0] * 4
    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows
    return np.array(y_onehots)

#Save images from train folder into filename, then prepare the images by folder 1-4 with image_prep()
def train_image():
    filename = []
    for i in range(4):
        filename.append('./train/{}/'.format(i+1))
    return image_prep(filename)

#Save images from test folder into filename, then prepare the images by folder 1-4 with image_prep()
def test_image():
    filename = []
    for i in range(4):
        filename.append('./test/{}/'.format(i+1))
    return image_prep(filename)

def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RandomFlip(mode = "horizontal_and_vertical", seed = 5))
    model.add(tf.keras.layers.RandomRotation(0.2, fill_mode= "reflect", interpolation="bilinear", seed= 5, fill_value=0.0))
    model.add(tf.keras.layers.Conv2D(filters =32, kernel_size=(7,7), padding="same", activation="relu", input_shape= (64,64,3)))
    model.add(tf.keras.layers.BatchNormalization(axis =1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.7)) #Dropout layers prevent neurons from relying on one input because it might be dropped out at random
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size=(3,3), padding="same", activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(tf.keras.layers.Dropout(0.5)) #Adding a third Dropout Layers dramatically decreases test accuracy
    #model.add(Flatten())
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), padding="same", activation = "relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.6))
    model.add(tf.keras.layers.Dense(28,activation = "relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #model.summary()
    return model

'''
High epoch value does contribute to overfitting but it correspond to higher accuracy. However, improvement rates do decrease overtime 
with diminishing return hence using Early Stopping to cut the training short when no improvement is detected.
''' 
def train_model(model, x_train, y_train):
    earlyStop = EarlyStopping(monitor = 'loss', patience = 17) #adjust based on numbers of Epoch (lower Epoch requires less patience, and vice versa)
    #batch_size set to 10 to increase training time leading to better accuracy 
    return model.fit(x=x_train, y=y_train, epochs = 158, validation_split = 0.20, batch_size = 10, callbacks = [earlyStop])

#Auto Evaluation of the model against test set
def auto_eval(model, x_test, y_test):
    loss, accuracy = model.evaluate(x=x_test, y=y_test)

    print('loss = ', loss)
    print('accuracy = ', accuracy)

#Manual Evaluation of model against test set
def manual_eval(model, x_test, y_test):
    predictions = model.predict(x=x_test) #Predict the values from model
    n_preds = len(predictions) #Calculate accuracy
    correct = 0
    wrong = 0
    for i in np.arange(n_preds):
        predict_max = np.argmax(predictions[i])
        actual_max = np.argmax(y_test[i])

        if predict_max == actual_max:
            correct += 1
        else:
            wrong += 1
        print('correct: {0}, wrong: {1}'.format(correct,wrong))
        print('accuracy =', correct/n_preds)

#Plot image samples
def plot_random_inputs_train(x_train, y_train):
    class_names = ['apple', 'banana', 'orange', 'mixed']
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    fig.subplots_adjust(hspace=0.4)
    
    for i, ax in enumerate(axes.flat):
        # Select a random index from the train set
        index = random.randint(0, len(x_train) - 1)
        
        # Get the image and true label
        image = x_train[index]
        true_label = np.argmax(y_train[index])
        
        # Plot the image and label
        ax.imshow(image)
        ax.set_title(class_names[true_label])  

    plt.show()

def plot(histogram):
    _, ax = plt.subplots(nrows =1, ncols =2, figsize =(12,5))

    #Plot Loss/Validation curve
    ax[0].plot(histogram.history['loss'])
    ax[0].plot(histogram.history['val_loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend(['train', 'validation'], loc='upper left')

    #Plot Accuracy/Validation curve
    ax[1].plot(histogram.history['accuracy'])
    ax[1].plot(histogram.history['val_accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend(['train', 'validation'], loc='upper left')

    plt.show()

def main():
    np.random.seed(5)
    model = create_model() #Create CNN
    x_train, y_train = train_image()
    histogram = train_model(model, x_train/255, y_train) #Normalize x_train
    plot(histogram)

    x_test, y_test = test_image()

    #Test the model against new data
    auto_eval(model, x_test/255, y_test)
    manual_eval(model, x_test/255, y_test)

    #Plot sample images
    plot_random_inputs_train(x_train, y_train)

if __name__ == '__main__':
    main()