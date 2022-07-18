import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import requests
import os
import tensorflow as tf

homedir = os.path.expanduser('~')
datadir = f"{homedir}/snowx/"

exampledataurl = "https://raw.githubusercontent.com/snowex-hackweek/website/main/book/tutorials/machine-learning/data/snow_depth_data.csv"

r = requests.get(exampledataurl, allow_redirects=True)

open(f"{datadir}snow_depth_data.csv", 'wb').write(r.content)

# retrieve the example training data first
dataset = pd.read_csv(f"{datadir}snow_depth_data.csv")

print(dataset.info())

print(dataset.head())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_dataset.describe().transpose()

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_dataset)
scaled_test = scaler.transform(test_dataset) ## fit_transform != transform. 
## transform uses the parameters of fit_transform

  
# Sepatare Features from Labels
train_X, train_y = scaled_train[:, :-1], scaled_train[:, -1]
test_X, test_y = scaled_test[:, :-1], scaled_test[:, -1]

print("TensorFlow version ==>", tf.__version__) 
print("Keras version ==>",tf.keras.__version__)

tf.random.set_seed(0) ## For reproducible results
linear_regression = tf.keras.models.Sequential() # Specify layers in their sequential order
# inputs are 4 dimensions (4 dimensions = 4 features)
# Dense = Fully Connected.  
linear_regression.add(tf.keras.layers.Dense(1, activation=None ,input_shape=(train_X.shape[1],)))
# Output layer has no activation with just 1 node

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
linear_regression.compile(optimizer = opt, loss='mean_squared_error')

print(linear_regression.summary())

# NOTE: can changed from epochs=150 to run faster, change to verbose=1 for per-epoch output
history =linear_regression.fit(train_X, train_y, epochs=100, validation_split = 0.2, verbose=0)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# linear regression

linear_regression.get_weights()

def Sigmoid(z):
    """
    A function that performs the sigmoid transformation
    
    Arguments:
    ---------
        -* z: array/list of numbers to activate
    
    Returns:
    --------
        -* logistic: the transformed/activated version of the array
    """
    
    logistic = 1/(1+ np.exp(-z))
    return logistic
    

def Tanh(z):
    """
    A function that performs the hyperbolic tangent transformation
    
    Arguments:
    ---------
        -* z: array/list of numbers to activate
    
    Returns:
    --------
        -* hyp: the transformed/activated version of the array
    """
    
    hyp = np.tanh(z)
    return hyp


def ReLu(z):
    """
    A function that performs the hyperbolic tangent transformation
    
    Arguments:
    ---------
        -* z: array/list of numbers to activate
    
    Returns:
    --------
        -* points: the transformed/activated version of the array
    """
    
    points = np.where(z < 0, 0, z)
    return points

z = np.linspace(-10,10)
fa = plt.figure(figsize=(16,5))

plt.subplot(1,3,1)
plt.plot(z,Sigmoid(z),color="red", label=r'$\frac{1}{1 + e^{-z}}$')
plt.grid(True, which='both')
plt.xlabel('z')
plt.ylabel('g(z)', fontsize=15)
plt.title("Sigmoid Activation Function")
plt.legend(loc='best',fontsize = 22)


plt.subplot(1,3,2)
plt.plot(z,Tanh(z),color="red", label=r'$\tanh (z)$')
plt.grid(True, which='both')
plt.xlabel('z')
plt.ylabel('g(z)', fontsize=15)
plt.title("Hyperbolic Tangent Activation Function")
plt.legend(loc='best',fontsize = 18)

plt.subplot(1,3,3)
plt.plot(z,ReLu(z),color="red", label=r'$\max(0,z)$')
plt.grid(True, which='both')
plt.xlabel('z')
plt.ylabel('g(z)', fontsize=15)
plt.title("Rectified Linear Unit Activation Function")
plt.legend(loc='best', fontsize = 18)

tf.random.set_seed(1000)  ## For reproducible results
network = tf.keras.models.Sequential() # Specify layers in their sequential order
# inputs are 4 dimensions (4 dimensions = 4 features)
# Dense = Fully Connected.   
# First hidden layer has 1000 neurons with relu activations.
# Second hidden layer has 512 neurons with relu activations
# Third hidden layer has 256 neurons with Sigmoid activations
network.add(tf.keras.layers.Dense(1000, activation='relu' ,input_shape=(train_X.shape[1],)))
network.add(tf.keras.layers.Dense(512, activation='relu')) # sigmoid, tanh
network.add(tf.keras.layers.Dense(256, activation='sigmoid'))
# Output layer uses no activation with 1 output neurons
network.add(tf.keras.layers.Dense(1)) # Output layer

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
network.compile(optimizer = opt, loss='mean_squared_error')

network.summary()

# NOTE: if you have time, consider upping epochs -> 150
history =network.fit(train_X, train_y, epochs=20, validation_split = 0.2, verbose=0)


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

# Prediction

## Linear Regression

yhat_linReg = linear_regression.predict(test_X)
inv_yhat_linReg = np.concatenate((test_X, yhat_linReg), axis=1)
inv_yhat_linReg = scaler.inverse_transform(inv_yhat_linReg)
inv_yhat_linReg = inv_yhat_linReg[:,-1]

## DNN
yhat_dnn = network.predict(test_X) 
inv_yhat_dnn = np.concatenate((test_X, yhat_dnn), axis=1)
inv_yhat_dnn = scaler.inverse_transform(inv_yhat_dnn)
inv_yhat_dnn = inv_yhat_dnn[:,-1]

## True Snow Depth (Test Set)
inv_y = test_dataset["snow_depth"]


## Put Observed and Predicted (Linear Regression and DNN) in a Dataframe
prediction_df = pd.DataFrame({"Observed": inv_y,
                    "LR":inv_yhat_linReg, "DNN":inv_yhat_dnn})

# check performance

def metrics_print(test_data,test_predict):
    print('Test RMSE: ', round(np.sqrt(mean_squared_error(test_data, test_predict)), 2))
    print('Test R^2 : ', round((r2_score(test_data, test_predict)*100), 2) ,"%")
    print('Test MAPE: ', round(mean_absolute_percentage_error(test_data, test_predict)*100,2), '%')
    
print("##************** Linear Regression Results **************##")
metrics_print(prediction_df['Observed'], prediction_df['LR'])
print(" ")
print(" ")

print("##************** Deep Learning Results **************##")
metrics_print(prediction_df['Observed'], prediction_df['DNN'])
print(" ")
print(" ")

# visualize performance

fa = plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
plt.scatter(prediction_df['Observed'],prediction_df['LR'])
plt.xlabel('True Values [snow_depth]', fontsize=15)
plt.ylabel('Predictions [snow_depth]', fontsize=15)
plt.title("Linear Regression")


plt.subplot(1,2,2)
plt.scatter(prediction_df['Observed'],prediction_df['DNN'])
plt.xlabel('True Values [snow_depth]', fontsize=15)
plt.ylabel('Predictions [snow_depth]', fontsize=15)
plt.title("Deep Neural Network")

# visualize error

LR_error = prediction_df['Observed'] - prediction_df['LR']
DNN_error = prediction_df['Observed'] - prediction_df['DNN']

fa = plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
LR_error.hist()
plt.xlabel('Error', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title("Linear Regression")

plt.subplot(1,2,2)
DNN_error.hist()
plt.xlabel('Error', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title("Deep Neural Network")

# save best model

network.save('DNN')

## To load model, use;
model = tf.keras.models.load_model('DNN')





