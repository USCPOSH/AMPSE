# ANN for Comparator

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# Importing the dataset
dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/Datasets/PY_COMPPin6501_TT.csv',header=None)



X = np.array(dataset.iloc[1:, 0:15].values,dtype='float64')
y = np.array(dataset.iloc[1:, [20,22,25,28,29,30,31]].values,dtype='float64')




parname = ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload','frdyinv','fcompnand','vos','VCM','VDD']


remfilt = [not math.isnan(d) for d in y[:,-1]]
X = X[remfilt]
y = y[remfilt]
y[:,1]=y[:,1]+(1.0-(y[:,1]+1e-9)//(1e-9))*1e-9





np.random.seed(1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

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
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
import keras.initializers as init


# Initialising the ANN
reg = Sequential()
reg.add(Dense(units = 80, kernel_initializer = init.glorot_uniform(), activation = 'elu', input_dim = 15))
reg.add(Dense(units = 80, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 80, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 7, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.01),loss = losses.mse)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.05, batch_size = 500, epochs = 500)

score = reg.evaluate(sX_test, sy_test, batch_size = 500)
print(score)

#==================================================================
#********************  Saving the regressor  **********************
#==================================================================
"""
import pickle
name  = 'compp65'
addr = '/home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/Reg_files/PY_COMPPin6501_TT/'

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
json_file = open('TH/model_th65.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
reg.load_weights('TH/reg_th65.h5')
   """ 
#Sc_X = joblib.load('scX_th65.pkl') 
#Sc_y = joblib.load('scY_th65.pkl')

#==================================================================
#**************************  Prediction  **************************
#==================================================================
sy_pred=reg.predict(sX_test)
y_pred=sc_y.inverse_transform(sy_pred)
z=(y_pred-y_test)
from sklearn.metrics import mean_squared_error
scores = [mean_squared_error(sy_pred[:,i],sy_test[:,i]) for i in range(len(y_test[0,:]))]


#==================================================================
#************************  Visualization  *************************
#==================================================================
lst_metric_names=['Power (mW)', 'ready (ps)','delay (ps)','kickn (uV)','Cin (fF)','Sigma Cin (fF)','IRN (mV)']
lst_metric_coef=[1000,1e12,1e12,1e6,1e15,1e15,1e3]

i=1
plt.figure()
plt.grid();plt.scatter(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])
