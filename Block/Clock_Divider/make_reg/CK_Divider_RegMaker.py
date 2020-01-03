#==================================================================
#*******************  Initialization  *****************************
#==================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('/home/Baishakhi/workarea_POSH_soumya/Clock_Divider/datasets/ck_dataset.csv',header=None)

dataset=dataset.dropna()

X = np.array(dataset.iloc[1:, 0:5].values,dtype='float64') # Selecting the parameters
y = np.array(dataset.iloc[1:, 5:].values,dtype='float64')  # Selecting the metrics

c = np.arange(0,len(y))
c = c[y[:,2]<0.00]
y = np.delete(y,c,axis=0)
X = np.delete(X,c,axis=0)

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import tensorflow.keras.initializers as init



# Initialising the ANN
reg = Sequential()
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu', input_dim = 5))
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 200, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 4  , kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.001),loss = losses.mse)

# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.05, batch_size = 500, epochs = 10000)

score = reg.evaluate(sX_test, sy_test, batch_size = 500)

#==================================================================
#********************  Saving the regressor  **********************
#==================================================================

name  = 'ckdiv65'
addr = '/home/Baishakhi/workarea_POSH_soumya/Clock_Divider/reg_files/'

reg_json=reg.to_json()
with open(addr+'model_'+name+'.json', "w") as json_file:
    json_file.write(reg_json)
reg.save_weights(addr+'reg_'+name+'.h5')  

from sklearn.externals import joblib
joblib.dump(sc_X, addr+'scX_'+name+'.pkl') 
joblib.dump(sc_y, addr+'scY_'+name+'.pkl')
pickle.dump( reg.get_weights(), open( addr+'w8_'+name+'.p', "wb" ) )


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

lst_metric_names = ['Output Frequency (GHz)','Power (mW)','Output_High (mV)','Output_Low (mV)' ]
lst_metric_coef  = [1e-9,1e3,1e3,1e3]

i=0
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])

i=1
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])

i=2
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])

i=3
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])
