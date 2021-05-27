# ANN for Comparator

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()
homeaddress = '/home/mohsen/PYTHON_PHD/CADtoolForRegression/SAR_ADC/'

# Importing the dataset
dataset = pd.read_csv(homeaddress+'Datasets/PY_DACTH6501_TT.csv',header=None)

dataset = dataset.dropna()


X = np.array(dataset.iloc[1:, 0:7].values,dtype='float64')
y = np.array(dataset.iloc[1:, 7:10].values,dtype='float64')
X[:,0] = np.log2(X[:,0])


remfilt = [d<5e-9 for d in y[:,0]]
X = X[remfilt]
y = y[remfilt]
remfilt = [d<1.0e-9 for d in y[:,2]]
X = X[remfilt]
y = y[remfilt]


#sum(y[:,2]>2.5e-9)
#plt.hist(y[:,2])
#plt.hist(y[:,2])

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


#sX = sc_X.fit_transform(X)
#sy = sc_y.fit_transform(y)
#
#from sklearn.model_selection import train_test_split
#sX_train, sX_test, sy_train, sy_test = train_test_split(sX, sy, test_size = 0.25)

#i=0
#plt.hist(sX_train[:,i],31)
#plt.hist(sX_test[:,i],31)
#j=3
#plt.hist(sy_train[:,j],31)
#plt.hist(sy_test[:,j],31)
#i=3
#plt.hist(y_train[:,i],31)
#plt.hist(y_test[:,i],31)

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
reg.add(Dense(units = 256 , kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 7))
reg.add(Dense(units = 256 , kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 256 , kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 3  , kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.00001),loss = losses.mean_absolute_error)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.1, batch_size = 500, epochs = 100)

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
import pickle
name  = 'th65'
addr = homeaddress+ 'Reg_files/PY_THDAC6501_TT/'

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



#plt.scatter(X_train[:,5],y_train[:,2]*1e9)
#==================================================================
#************************  Visualization  *************************
#==================================================================
lst_metric_names=['delayth (ps)','msb (V)','dlydac (ps)']
lst_metric_coef=[1e12,1.0,1e12]
lst_scale = scores*sc_y.scale_
#array([1.12640554e-04, 4.18343790e-12, 2.15038772e-12, 2.66576727e-02,1.77480215e-06, 1.25306277e-16, 7.77895422e-18, 6.81829886e-06])
i=0
#plt.figure()
#sns.kdeplot(z[:,i]*lst_metric_coef[i], shade=True)
sns.jointplot(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i],kind='reg');plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])
plt.plot(y_pred[:,i]*lst_metric_coef[i],+3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')
plt.plot(y_pred[:,i]*lst_metric_coef[i],-3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')

#remfilt = [d<2.5e-9 for d in y[:,2]]
#X = X[remfilt]
#y = y[remfilt]
