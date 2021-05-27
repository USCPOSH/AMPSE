# ANN for Comparator

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math


sns.set()
homeaddress = '/home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/'
# Importing the dataset
#  /home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/Datasets/PY_COMPPin6502_TT.csv'
dataset = pd.read_csv(homeaddress + 'Datasets/PY_COMPPin6501_TT.csv',header=None)
#dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/Datasets/PY_COMPPin6502_TT.csv',header=None)
dataset = dataset.dropna()



X = np.array(dataset.iloc[1:, 0:15].values,dtype='float64')
y = np.array(dataset.iloc[1:, 23:24].values,dtype='float64')

# ['power','readyp','delayr','delayf','kickn','cin','scin','irn']

#plt.hist(y[:,6])

parname = ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload','frdyinv','fcompnand','vos','VCM','VDD']





np.random.seed(1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

#
#plt.scatter(X_train[:,4]*1e0,y_train[:,3]*1e3)
#plt.scatter(X_test [:,4]*1e0,y_test [:,3]*1e3)


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler(feature_range=(-1,1))
sc_y = StandardScaler()

sX_train = sc_X.fit_transform(X_train)
sy_train = sc_y.fit_transform(y_train)
sX_test  = sc_X.transform    (X_test )
sy_test  = sc_y.transform    (y_test )





#==================================================================
#********************  Learning  **********************************
#==================================================================

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import tensorflow.keras.initializers as init
import tensorflow as tf


# Initialising the ANN
reg = Sequential()
reg.add(Dense(units = 64, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 15))
reg.add(Dense(units = 128, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 64, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 1, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.0001),loss = losses.binary_crossentropy,metrics=[tf.keras.metrics.Accuracy()])



# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.1, batch_size = 500, epochs = 200)

score = reg.evaluate(sX_test, sy_test, batch_size = 500)

#==================================================================
#**************************  Prediction  **************************
#==================================================================
sy_pred=reg.predict(sX_test)
y_pred=sc_y.inverse_transform(sy_pred)
z=(y_pred-y_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
scores = [mean_absolute_error(sy_pred[:,i],sy_test[:,i]) for i in range(len(y_test[0,:]))]


#print('%s, ' % scores*sc_y.scale_)
#==================================================================
#********************  Saving the regressor  **********************
#==================================================================
"""
import pickle
name  = 'compp65fff'
addr = homeaddress+'Reg_files/PY_COMPPin6502_TT/'

reg_json=reg.to_json()
with open(addr+'model_'+name+'.json', "w") as json_file:
    json_file.write(reg_json)
reg.save_weights(addr+'reg_'+name+'.h5')  

from sklearn.externals import joblib
joblib.dump(sc_X, addr+'scX_'+name+'.pkl') 
joblib.dump(sc_y, addr+'scY_'+name+'.pkl')
pickle.dump( reg.get_weights(), open( addr+'w8_'+name+'.p', "wb" ) )
pickle.dump( scores, open( addr+'err_'+name+'.p', "wb" ) )


"""

#==================================================================
#********************  Loading the regressor  *********************
#==================================================================
"""
from sklearn.externals import joblib
from keras.models import model_from_json 
json_file = open('TH/model_th65.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
reg.load_weights('TH/reg_th65.h5')
   """ 
#Sc_X = joblib.load('scX_th65.pkl') 
#Sc_y = joblib.load('scY_th65.pkl')




#==================================================================
#************************  Visualization  *************************
#==================================================================
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

