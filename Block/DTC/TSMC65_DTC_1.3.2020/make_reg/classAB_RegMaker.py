# NN for Folded Cascode

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# Importing the dataset
cf = os.getcwd()
cfs = cf.split('/')
sss = '/'
homefolder=sss.join(cfs[:-1])

dataset = pd.read_csv(homefolder+'/Datasets/classAB_45.csv',header=None)

dataset =  dataset.dropna()

X = np.array(dataset.iloc[1:, 0:9].values,dtype='float64')
y = np.array(dataset.iloc[1:, 9:20].values,dtype='float64')
X = np.delete(X, -1, axis=1)
parname = [ 'fbias','lbias','fin','fp','lin','lp','vcmo','mamp']

#remfilt = [not d<0 for d in y[:,2]]
#X = X[remfilt]
#y = y[remfilt]
#y[:,1]=y[:,1]-((y[:,1]+1e-9)//(1e-9))*1e-9


np.random.seed(1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

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
reg.add(Dense(units = 128, kernel_initializer = init.glorot_uniform(), activation = 'elu', input_dim = 8))
reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 512, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 128, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 11, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.001),loss = losses.mse)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, batch_size = 256, epochs = 50)

score = reg.evaluate(sX_test, sy_test, batch_size = 250)
print(score)
#plt.hist(y[:,2]);

#==================================================================
#********************  Saving the regressor  **********************
#==================================================================

import pickle
name  = 'classAB'
addr = homefolder + '/Reg_files/class_ab_45/'

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
json_file = open(homefolder+'/Reg_files/class_ab_45/model_classAB.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
reg.load_weights(homefolder+'/Reg_files/class_ab_45/reg_classAB.h5')
 
#Sc_X = joblib.load('scX_th65.pkl') 
#Sc_y = joblib.load('scY_th65.pkl')

#==================================================================
#**************************  Prediction  **************************
#==================================================================
sy_pred=reg.predict(sX_test)
from sklearn.metrics import mean_squared_error
scores = [mean_squared_error(sy_pred[:,i],sy_test[:,i]) for i in range(len(y_test[0,:]))]
#==================================================================
#************************  Visualization  *************************
#==================================================================

y_pred=sc_y.inverse_transform(sy_pred)
z=(y_pred-y_test)
lst_metric_names=['cin(pF)','cout(pF)','gain', 'gm(mS)','pole1(MHz)','rout(MOhm)','zero(MHz)','pwr(mW)','swingn(mV)','swingp(mV)']
lst_metric_coef=[1e12,1e12,1,1e3,1e6,1e6,1e6,1e3,1e3,1e3]

i=14
plt.figure()
plt.grid();plt.scatter(y_test[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])
j=2
plt.grid();plt.scatter(np.log(y[:,i]),y[:,j]*lst_metric_coef[j]);plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel(lst_metric_names[j])
"""