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



X = np.array(dataset.iloc[1:, 0:15].values,dtype='float64')
y = np.array(dataset.iloc[1:, 20:28].values,dtype='float64')

# ['power','readyp','delayr','delayf','kickn','cin','scin','irn']

#plt.hist(y[:,6])

parname = ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload','frdyinv','fcompnand','vos','VCM','VDD']


remfilt = [not math.isnan(d) for d in y[:,-1]]
X = X[remfilt]
y = y[remfilt]
remfilt = [not d<0 for d in y[:,1]]
X = X[remfilt]
y = y[remfilt]
remfilt = [not d<0 for d in y[:,2]]
X = X[remfilt]
y = y[remfilt]



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


sy_train[:,3]=sy_train[:,3]+0.2


#==================================================================
#********************  Learning  **********************************
#==================================================================

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import tensorflow.keras.initializers as init


# Initialising the ANN
reg = Sequential()
reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 15))
reg.add(Dense(units = 512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
reg.add(Dense(units = 8, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.0001),loss = losses.mean_absolute_error)


# Fitting the ANN to the Training set
reg.fit(sX_train, sy_train, validation_split=0.1, batch_size = 500, epochs = 800)

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
name  = 'compp65'
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
name  = 'compp65'
addr = homeaddress+'Reg_files/PY_COMPPin6502_TT/'
from sklearn.externals import joblib
from tensorflow.keras.models import model_from_json 
json_file = open(addr+'model_'+name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
reg = model_from_json(loaded_model_json)
reg.load_weights(addr+'reg_'+name+'.h5')
   """ 
#Sc_X = joblib.load('scX_th65.pkl') 
#Sc_y = joblib.load('scY_th65.pkl')




#==================================================================
#************************  Visualization  *************************
#==================================================================
lst_metric_names=['Power (mW)', 'ready (ps)','delay (ps)','false delay','kickn (uV)','Cin (fF)','Sigma Cin (fF)','IRN (mV)']
lst_metric_coef=[1000,1e12,1e12,1.0 , 1e6,1e15,1e15,1e3]
lst_scale = scores*sc_y.scale_
#array([1.12640554e-04, 4.18343790e-12, 2.15038772e-12, 2.66576727e-02,1.77480215e-06, 1.25306277e-16, 7.77895422e-18, 6.81829886e-06])
i=3
#plt.figure()
#sns.kdeplot(z[:,i]*lst_metric_coef[i], shade=True)
sns.jointplot(y_pred[:,i]*lst_metric_coef[i],z[:,i]*lst_metric_coef[i],kind='reg');plt.xlabel('SPICE Simulated '+lst_metric_names[i]);plt.ylabel('Error of '+lst_metric_names[i])
plt.plot(y_pred[:,i]*lst_metric_coef[i],+3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')
plt.plot(y_pred[:,i]*lst_metric_coef[i],-3*np.ones_like(z[:,i])*lst_metric_coef[i]*lst_scale[i],'r')

sum(z[:,i]>0.5)
sum(z[:,i]<-0.5)
