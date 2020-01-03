# ANN for TH

#==================================================================
#*******************  Initialization  *****************************
#==================================================================


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Importing the dataset
dataset = pd.read_csv('TH/PYDATA2_65.csv',header=None)

dataset=dataset.dropna()
#dataset = pd.read_csv('Sweep.csv',header=None)

#dataset=dataset.drop([0],axis=0)


#X = np.array(dataset.iloc[:, 1:6].values,dtype='float64')
#yupdated=np.array(dataset.iloc[:, 6:9].values,dtype='float64')

X = np.array(dataset.iloc[:, 0:6].values,dtype='float64')
y=np.array(dataset.iloc[:, 7:15].values,dtype='float64')

yupdated=y[:,0:6]
yupdated[:,5]=np.min(y[:,5:8],axis=1)



#parname=['mn','mp','cl','linnn','lppp', 'vdd','winnn','wppp']
#minpar=[1  ,1  ,0.5e-15 ,60e-9,60e-9,1.0,200e-9 ,200e-9 ]
#maxpar=[32 ,32 ,20e-15  ,60e-9,60e-9,1.0,1.2e-6 ,1.2e-6 ]


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler(feature_range=(-1,1))
sc_y = StandardScaler()
X_new = sc_X.fit_transform(X)
y_new = sc_y.fit_transform(yupdated)


np.random.seed(1234)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size = 0.25)

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
reg.add(Dense(units = 90, kernel_initializer = init.glorot_uniform(), activation = 'elu', input_dim = 6))
reg.add(Dense(units = 90, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 90, kernel_initializer = init.glorot_uniform(), activation = 'elu'))
reg.add(Dense(units = 6, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

# Compiling the ANN
reg.compile(optimizer = optimizers.Adam(lr=0.001),loss = losses.mse)


# Fitting the ANN to the Training set
reg.fit(X_train, y_train, batch_size = 500, epochs = 4000)



#==================================================================
#********************  Saving the regressor  **********************
#==================================================================
"""
reg_json=reg.to_json()
with open("model_th65.json", "w") as json_file:
    json_file.write(reg_json)
reg.save_weights("reg_th65.h5")  

from sklearn.externals import joblib
joblib.dump(sc_X, 'scX_th65.pkl') 
joblib.dump(sc_y, 'scY_th65.pkl')
pickle.dump( reg.get_weights(), open( "w8_th65.p", "wb" ) )

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

y_pred=reg.predict(X_test)

from sklearn.metrics import mean_squared_error

score=mean_squared_error(y_pred,y_test)
y_predi=sc_y.inverse_transform(y_pred)
y_testi=sc_y.inverse_transform(y_test)
z=(y_predi-y_testi)


#==================================================================
#************************  Visualization  *************************
#==================================================================

plt.hist(yupdated[:,-1],50)