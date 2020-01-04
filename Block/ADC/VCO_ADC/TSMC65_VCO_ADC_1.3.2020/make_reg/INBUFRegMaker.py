# ANN for TH

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns

sns.set()

# Importing the dataset
dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/CADtoolForRegression/VCO_ADC/Datasets/PY_INBUF201_TT.csv',header=None)

dataset=dataset.dropna()
#dataset = pd.read_csv('Sweep.csv',header=None)

#dataset=dataset.drop([0],axis=0)


#X = np.array(dataset.iloc[:, 1:6].values,dtype='float64')
#yupdated=np.array(dataset.iloc[:, 6:9].values,dtype='float64')

X = np.array(dataset.iloc[1:, [0,1,2,3,4,6,7]].values,dtype='float64')
y = np.array(dataset.iloc[1:, 8:].values,dtype='float64')

# power	gain	bw	outvcm	swingp	avcm	kickn	irn

remfilt = [d<10e9 for d in y[:,2]]
X = X[remfilt]
y = y[remfilt]
#plt.hist(y[:,1],101)

#parname=['mn','mp','cl','linnn','lppp', 'vdd','winnn','wppp']
#minpar=[1  ,1  ,0.5e-15 ,60e-9,60e-9,1.0,200e-9 ,200e-9 ]
#maxpar=[32 ,32 ,20e-15  ,60e-9,60e-9,1.0,1.2e-6 ,1.2e-6 ]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler(feature_range=(-1,1))
sc_y = StandardScaler()

sX_train = sc_X.fit_transform ( X_train )
sy_train = sc_y.fit_transform ( y_train )
sX_test  = sc_X.transform     ( X_test  )
sy_test  = sc_y.transform     ( y_test  )

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
reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 7))
reg.add(Dense(units = 512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 11, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.0001),loss = losses.mean_absolute_error)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.1, batch_size = 500, epochs = 1000)

score = reg.evaluate(sX_test, sy_test, batch_size = 500)

#==================================================================
#**************************  Prediction  **************************
#==================================================================
sy_pred=reg.predict(sX_test)
y_pred=sc_y.inverse_transform(sy_pred)
z=(y_pred-y_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error
scores = [mean_absolute_error(sy_pred[:,i],sy_test[:,i]) for i in range(len(y_test[0,:]))]

#==================================================================
#********************  Saving the regressor  **********************
#==================================================================
"""
name  = 'inbuf265'
addr ='/home/mohsen/PYTHON_PHD/CADtoolForRegression/VCO_ADC/Reg_files/INBUF2/'
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
#**************************  Prediction  **************************
#==================================================================

#plt.hist(y[:,4],51)

#==================================================================
#************************  Visualization  *************************
#==================================================================
lst_params_names = ['multi','fing_in','l_ttt','fing_ttt','VCM','dvv','wpppp','fpppp']
lst_metrics_names = ['power','gain','bw','outvcm','avcm','kickn','irn','outn3','outn1','outp1','outp3']

lst_metric_coef=[1000,1.0,1e-9,1.0,1.0,1.0,1.0e6,1,1,1,1]
lst_scale = scores*sc_y.scale_


#i=2
#sns.distplot(y[:,i]*lst_metric_coef[i])


i=7
sns.jointplot(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i],kind='reg');plt.xlabel('SPICE Simulated '+lst_metrics_names[i]);plt.ylabel('Error of '+lst_metrics_names[i])
plt.plot(y_pred[:,i]*lst_metric_coef[i],+3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')
plt.plot(y_pred[:,i]*lst_metric_coef[i],-3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')