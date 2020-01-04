# ANN for VCO type1 with 8 output
# We want to get the linearity in this script.
# To estimate the non-linearity we get 8 different points along the transfer curve of the VCO. Later we decide 
# This one gives much better results
#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Importing the dataset
#dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/TransferLearning/VCO/MakeReg/VCO/PYDATA1_65.csv',header=None)
dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/CADtoolForRegression/VCO_ADC/Datasets/PY_VCO01_TT.csv',header=None)

dataset=dataset.dropna()
#dataset = dataset[(dataset.T != 0).all()]

#dataset = pd.read_csv('Sweep.csv',header=None)

#dataset=dataset.drop([0],axis=0)


#X = np.array(dataset.iloc[:, 1:6].values,dtype='float64')
#yupdated=np.array(dataset.iloc[:, 6:9].values,dtype='float64')

X = np.array(dataset.iloc[1:, 1:5].values,dtype='float64')
yupdated=np.array(dataset.iloc[1:, 8:].values,dtype='float64')

#yupdated[:,2:4]=1/yupdated[:,2:4]

c=np.arange(0,len(yupdated))
c=c[yupdated[:,2]<0.00]
yupdated=np.delete(yupdated,c,axis=0)
X=np.delete(X,c,axis=0)

y=yupdated


np.random.seed(1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


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
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
import keras.initializers as init


# Initialising the ANN
reg = Sequential()
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu', input_dim = 4))
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 12, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.001),loss = losses.mse)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.05, batch_size = 500, epochs = 2500)

score = reg.evaluate(sX_test, sy_test, batch_size = 500)



#==================================================================
#********************  Saving the regressor  **********************
#==================================================================
"""
name  = 'vco65'
addr = '/home/mohsen/PYTHON_PHD/CADtoolForRegression/VCO_ADC/Reg_files/VCO/'

reg_json=reg.to_json()
with open(addr+'model_'+name+'.json', "w") as json_file:
    json_file.write(reg_json)
reg.save_weights(addr+'reg_'+name+'.h5')  

from sklearn.externals import joblib
joblib.dump(sc_X, addr+'scX_'+name+'.pkl') 
joblib.dump(sc_y, addr+'scY_'+name+'.pkl')
pickle.dump( reg.get_weights(), open( addr+'w8_'+name+'.p', "wb" ) )
"""

#==================================================================
#********************  Loading the regressor  *********************
#==================================================================
"""
from sklearn.externals import joblib
from keras.models import model_from_json 
json_file = open('model_vco65.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
reg.load_weights('reg_vco65.h5')

Sc_X = joblib.load('scX_vco65.pkl') 
Sc_y = joblib.load('scY_vco65.pkl')

"""

#==================================================================
#**************************  Prediction  **************************
#==================================================================

sy_pred=reg.predict(sX_test)

from sklearn.metrics import mean_squared_error

y_pred=sc_y.inverse_transform(sy_pred)
z=(y_pred-y_test)


z_pred=y_pred/np.std(y_pred,axis=0)
z_test=y_test/np.std(y_pred,axis=0)

scores = [mean_squared_error(sy_pred[:,i],sy_test[:,i]) for i in range(len(y_test[0,:]))]


#==================================================================
#************************  Visualization  *************************
#==================================================================


#i=11;plt.grid();plt.scatter(y_predi[:,i]*1e-12,y_testi[:,i]*1e-12);plt.xlabel('SPICE Simulated Fmin (GHz)');plt.ylabel('Fmin Error (GHz)')
lst_metric_names=['Power (mW)', 'Input Voltage VCM (V)','Input differential Voltage (V)','Noise (Hz^2)','freq1 (GHz)','freq2 (GHz)',
                  'freq3 (GHz)','freq4 (GHz)','freq5 (GHz)','freq6 (GHz)','freq7 (GHz)','freq8 (GHz)']
lst_metric_coef=[1000,1,1,1e-6,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9,1e-9]

i=2
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])

