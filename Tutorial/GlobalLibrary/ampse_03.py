# AMPSE version 0.3 Library Usage:
# The core requires Tensorflow 2.1
# 0.2:  capabale of choosing layers to train!
# 0.3:  bug fixed, shows overfitting if possible, scaling is choosable 

# from sklearn.svm import SVR
# from scipy.optimize import curve_fit
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor

import numpy as np

import pandas as pd
import warnings
from pickle import dump, load
from time import time

import tensorflow as tf
tf.keras.backend.set_floatx('float32')
# For poly regression:




# You have to design the models

# do not use min max scaler to 




class TF_Model():
    def __init__(self,name,model,load_address=None,err_save =False,p=[-1,1,0.1],dataset1=None,is_header=True,parloc=range(0,1),metricloc=range(1,2),dataset_cleaning=lambda x: x,dataset2= None):
        
        if load_address:
            
            self.name  = name
            self.model = model
            self.load_model(load_address,load_name=name,err_save =err_save)
            
            
            
        else:
            self.name = name
            self.model=model
            self.pmin = p[0]
            self.pmax = p[1]
            self.pstp = p[2]
            self.dataset1 = dataset1
            self.dataset2 = dataset2
            self.is_header = is_header
            self.parloc=parloc
            self.metricloc= metricloc
            self.ds_clean = dataset_cleaning
            self.s_weight = False


        self.train_loss = tf.keras.metrics.Mean(name='train_loss') 
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.version = 0.3
            
    def __version__(self):
        return self.version
    def read_dataset(self,ds):
        try:
            if self.is_header:
                ds_start = 1
            else:
                ds_start = 0
                
            X = np.array(ds.iloc[ds_start:, self.parloc].values,dtype='float64')
            y = np.array(ds.iloc[ds_start:, self.metricloc].values,dtype='float64')
            
            return X,y
            
            
            
        except ValueError:
                
            print("Dataset is not readable!")
            if self.is_header:
                print("Make sure the dataset has header")
                
    def define_parametersrange(self):
        ds1 = pd.read_csv(self.dataset1,head=None)
        ds1 = self.ds_clean(ds1)
        X,_ = self.read_dataset(ds1)
        self.pmin = np.min(X,axis=0)                 
        self.pmax = np.max(X,axis=0)                 
        z = np.sort(X,axis=0)
        zd = np.diff(z,axis=0)
        self.pstp = np.abs(np.max(zd,axis=0))
            
    def get_scaling(self,y):
        
        mmin = np.min(y,axis=0)
        mmax = np.max(y,axis=0)
        if np.any(mmin-mmax==0):
            warnings.warn("Constant value in the dataset. Please Fix the issue")
            
        self.slopeY = 2/(mmax-mmin*(1+1e-12))
        self.smeanY = -(mmax+mmin)/(mmax-mmin*(1+1e-12))
        self.slopeX = 2/(self.pmax-self.pmin*(1+1e-12))
        self.smeanX = -(self.pmax+self.pmin)/(self.pmax-self.pmin*(1+1e-12))
        
    
    def scaleX (self,X,chosen=[]):   
        if len(chosen)>0:
            return self.slopeX[chosen]*X+self.smeanX[chosen]
        else:
            return self.slopeX*X+self.smeanX
    
    def scaleY(self,y,chosen=[]):
        if len(chosen)>0:
            return self.slopeY[chosen]*y+self.smeanY[chosen]
        else:
            return self.slopeY*y+self.smeanY
    
    def iscaleX(self,sX,chosen=[]):
        if len(chosen)>0:
            return (sX-self.smeanX[chosen])/self.slopeX[chosen]
        else:
            return (sX-self.smeanX)/self.slopeX
    
    def iscaleY(self,sY,chosen=[]):
        if len(chosen)>0:
            return (sY-self.smeanY[chosen])/self.slopeY[chosen]    
        else:
            return (sY-self.smeanY)/self.slopeY    
    
    def train_test_split(self,X,y,s=[],random=False,training_size=1000,pointlists=[]):
        # Splits the dataset of X,y into training and testing dataset
        # X: np array is the variables
        # y: np array is the results
        # s: np array of size 1, is the sample_weight for loss
        # random: boolean if True, it splits randomly
        # training_size: integer, if set it defines training dataset size 
        # pointlists: list, if set it exactly chooses the training points from this list
        
        choosen_train = np.zeros(len(X))
        if pointlists:
            choosen_train[pointlists]=1
        else:
            
            if random:
                stack_rnd = np.random.choice(len(X),training_size,replace=False)
                choosen_train[stack_rnd]=1
                
            else:
                stack_uni = range(0,training_size)
                choosen_train[stack_uni]=1
                
        X_train = X[choosen_train>=0.5]
        X_test  = X[choosen_train<0.5]
        y_train = y[choosen_train>=0.5]
        y_test  = y[choosen_train<0.5]
        if len(s)>0:
            S_train = s[choosen_train>=0.5]
            S_test  = s[choosen_train<0.5 ]
            return X_train, y_train, X_test ,y_test, S_train, S_test
        
        return X_train, y_train, X_test, y_test
    
    def scale_all(self,lst_train_test, scaling = True):
        
        #lst_train_test =  X_train, y_train,X_test,y_test 
        
        X_train = lst_train_test[0]
        y_train = lst_train_test[1]
        X_test  = lst_train_test[2]
        y_test  = lst_train_test[3]
        if scaling:
            self.get_scaling(y_train)
        sX_train = self.scaleX(X_train)
        sX_test = self.scaleX(X_test)
        sy_train = self.scaleY(y_train)
        sy_test = self.scaleY(y_test)
        
        return sX_train, sy_train,sX_test,sy_test        
            
    def step1_preprocessing (self,training_size=None, sample_weight=[],random_training_samples=False, batch=512,training_points=[],scaled=True):
        # preprocessing before training of dataset 1
        # training_size: Integer number or None. It is the size of training dataset. It should be smaller than dataset size. None picks the whole dataset
        # sample_weight: training loss sample weight
        # random_training_sample: if True, the traning dataset will be chosen randomly, otherwise it will be from the start
        # batch: batch
        # training_points: List or 1D numpy array. if not empty chooses the training dataset points
        # scaled: If True, it scales the traning dataset automatically.
        
        ds1 = pd.read_csv(self.dataset1,header=None)
        ds1 = self.ds_clean(ds1)
        X,y=self.read_dataset(ds1)
        if not training_size:
            training_size = len(X)
        
        lst_train_test = self.train_test_split(X, y, sample_weight, random_training_samples,training_size, training_points)
        sX_train, sy_train, sX_test, sy_test = self.scale_all(lst_train_test,scaled)
        
        self.train_ds = tf.data.Dataset.from_tensor_slices((sX_train, sy_train)).batch(batch)
        self.test_ds  = tf.data.Dataset.from_tensor_slices((sX_test, sy_test)).batch(batch)
        
        
        if len(sample_weight)==len(X):
            self.train_ds =  tf.data.Dataset.from_tensor_slices((sX_train, sy_train, lst_train_test[4])).batch(batch)
            self.test_ds  =  tf.data.Dataset.from_tensor_slices((sX_test,  sy_test , lst_train_test[5])).batch(batch)
            self.s_weight = True
        
            
    def step1_preprocessing_ds2(self,training_size=None, sample_weight=[], training_points=[], test_points = None, batch=512,scaled=True):
        ds2 = pd.read_csv(self.dataset2,header=None)
        ds2 = self.ds_clean(ds2)
        X,y=self.read_dataset(ds2)
        if not training_size:
            training_size = len(X)
            
        lst_train_test = self.train_test_split(X, y, sample_weight, False,training_size, training_points)
        sX_train, sy_train, sX_test, sy_test = self.scale_all(lst_train_test,scaled)
        
        if test_points:
            sX_test=sX_test[-test_points:]
            sy_test=sy_test[-test_points:]
            
        self.train_ds2 = tf.data.Dataset.from_tensor_slices((sX_train, sy_train)).batch(batch)
        self.test_ds2  = tf.data.Dataset.from_tensor_slices((sX_test, sy_test)).batch(batch)
    
        if len(sample_weight)==len(X):
            self.train_ds2 = tf.data.Dataset.from_tensor_slices((sX_train, sy_train,lst_train_test[4])).batch(batch)
            self.test_ds2  = tf.data.Dataset.from_tensor_slices((sX_test, sy_test,lst_train_test[5])).batch(batch)
    
    def step2_training(self,loss_object,optimizer,EPOCHS=1000,dataset=1,see_validation=True,weights=[]):
        
        # self.model.compile(optimizer = optimizer, loss = loss_object)
        # self.model.fit(train_ds = self.train_ds, epochs=EPOCHS, validation_data=self.test_ds)
        self.loss_object = loss_object
        self.optimizer = optimizer
        if weights:
            self.trainable_var = self.model.trainable_variables[weights]
        else:
            self.trainable_var = self.model.trainable_variables
        lst_out = self.fitfit(EPOCHS = EPOCHS,dataset=dataset,see_test=see_validation)
        return lst_out
        # fit(self.train_ds,self.test_ds,self.train_loss,self.test_loss,self.model,loss_object,optimizer,EPOCHS)
    

    def step3_erroranalysis(self,division=100,err_type='std',poly_order=5,dataset_num=1):
        
        # Predicts the given dataset:
        #  dataset_num : 0-> training dataset 1
        #                1-> testing  dataset 1
        #                2-> testing  dataset 2
        #                3-> training dataset 2
        self.err_a=self.error_analysis_poly(division=division,err_type=err_type,poly_order=poly_order,dataset_num=dataset_num)
        
        pass
    
    def tf_scaled_predict(self,sX_test):
        sy_pred = self.model(sX_test)
        return sy_pred
    
    def tf_predict(self,X_test):
        sX_test = self.scaleX(X_test)
        sy_pred = self.tf_scaled_predict(sX_test)
        return self.iscaleY(sy_pred)        
    

    def predict_datasets(self, number =0):
        # Predicts the given dataset:
        #  number : 0-> training dataset 1
        #           1-> testing  dataset 1
        #           2-> testing  dataset 2
        #           3-> training dataset 2
        
        if number ==0:
            data = self.train_ds
        elif number ==1:
            data = self.test_ds
        elif number ==2:
            data = self.test_ds2
        elif number ==3:
            data = self.train_ds2
        lst_prediction=[]
        lst_inputs=[]
        lst_outputs=[]
        
        if len(data.element_spec)>2:
            train_with_three = True
        else:
            train_with_three = False
        if train_with_three:
            for test_inputs, test_outputs, _ in data:
                lst_prediction.append( np.array(self.model(test_inputs)))
                lst_inputs.append(test_inputs)
                lst_outputs.append(test_outputs)            
        else:
            for test_inputs, test_outputs in data:
                lst_prediction.append( np.array(self.model(test_inputs)))
                lst_inputs.append(test_inputs)
                lst_outputs.append(test_outputs)
        
        X_test = np.array(self.iscaleX(np.concatenate((lst_inputs),axis=0)))
        y_test = np.array(self.iscaleY(np.concatenate((lst_outputs),axis=0)))
        y_pred = np.array(self.iscaleY(np.concatenate((lst_prediction),axis=0)))
        
        d_pred = y_pred - y_test
        
        return X_test,y_test,y_pred,d_pred
    
    
    @tf.function
    def train_step(self,inputs, outputs):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.loss_object(outputs, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
        
    @tf.function
    def test_step(self,inputs, outputs):
        predictions = self.model(inputs)
        t_loss = self.loss_object(outputs, predictions)
        return t_loss
    
    @tf.function
    def train_step_sw(self,inputs, outputs, samplew):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self.loss_object(outputs, predictions,samplew)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
        
    @tf.function
    def test_step_sw(self,inputs, outputs, samplew):
        predictions = self.model(inputs)
        t_loss = self.loss_object(outputs, predictions,samplew)
        return t_loss
    
    def fitfit(self,EPOCHS=1000,dataset=1,see_test=True):
        
        if dataset==1:
            train_ds = self.train_ds
        elif dataset==2:
            train_ds = self.train_ds2
        if see_test:
            if dataset==1:
                test_ds = self.test_ds
            elif dataset==2:
                test_ds = self.test_ds2
                
        test_ds  = self.test_ds
        
        if len(self.train_ds.element_spec)>2:
            train_with_three = True
        else:
            train_with_three = False
        lst_out=[]
        for epoch in range(EPOCHS):
            
            tstart1 = time()
            if train_with_three:
                for inputs, outputs,sample_weight in train_ds:
                    self.train_loss( self.train_step_sw(inputs, outputs, sample_weight))
            else:            
                for inputs, outputs in train_ds:
                    self.train_loss( self.train_step(inputs, outputs))
                    
            tend1   = time()
            if see_test:
                
                tstart2 = time()
                if train_with_three:
                    for test_inputs, test_outputs, sample_weight in test_ds:
                        self.test_loss(self.test_step_sw(test_inputs, test_outputs, sample_weight))
                else:
                    for test_inputs, test_outputs in test_ds:
                        self.test_loss(self.test_step(test_inputs, test_outputs))
                
                tend2  = time()
                print('EPOCH #%1.0f, Train Loss: %1.5f, Test Loss: %1.5f, Take %1.2f Seconds, and %1.2f Seconds\n'
                      %(epoch+1,self.train_loss.result(),self.test_loss.result(),(tend1-tstart1),(tend2-tstart2)))
                
                train_loss = self.train_loss.result().numpy()
                test_loss  = self.test_loss.result().numpy()
                lst_out.append([epoch+1,train_loss,test_loss,(tend1-tstart1),(tend2-tstart2)])
                
                if epoch>100:
                    # _,train_lossb,test_lossb,_,_ = lst_out[epoch-10]
                    train_losses = [lst_out[x][1] for x in range(epoch-100,epoch) ]
                    test_losses  = [lst_out[x][2] for x in range(epoch-100,epoch) ]
                    
                    if sum(train_losses)<sum(train_losses) and sum(test_losses)>sum(test_losses):
                        print('WARNING: POSSIBLE OVERFITTING DETECTED! YOU CAN ABORT BY CTRL+C')
                
                
                self.test_loss.reset_states()
            else:
                print('EPOCH #%1.0f, Train Loss: %1.5f, Take %1.2f Seconds\n'
                      %(epoch+1,self.train_loss.result(),(tend1-tstart1)))    
                train_loss = self.train_loss.result().numpy()
                lst_out.append([epoch+1,train_loss,(tend1-tstart1)])
            self.train_loss.reset_states()
        
        return lst_out
        
    # def predict_dataset1(self):
    #     lst_prediction=[]
    #     lst_inputs=[]
    #     lst_outputs=[]
    #     for test_inputs, test_outputs in self.test_ds:
    #         lst_prediction.append( np.array(self.model(test_inputs)))
    #         lst_inputs.append(test_inputs)
    #         lst_outputs.append(test_outputs)
        
    #     X_test = np.array(self.iscaleX(np.concatenate((lst_inputs),axis=0)))
    #     y_test = np.array(self.iscaleY(np.concatenate((lst_outputs),axis=0)))
    #     y_pred = np.array(self.iscaleY(np.concatenate((lst_prediction),axis=0)))
        
    #     d_pred = y_pred - y_test
        
    #     return X_test,y_test,y_pred,d_pred
            
    # def predict_training_dataset1(self):
    #     lst_prediction=[]
    #     lst_inputs=[]
    #     lst_outputs=[]
    #     for test_inputs, test_outputs in self.train_ds:
    #         lst_prediction.append( np.array(self.model(test_inputs)))
    #         lst_inputs.append(test_inputs)
    #         lst_outputs.append(test_outputs)
        
    #     X_test = np.array(self.iscaleX(np.concatenate((lst_inputs),axis=0)))
    #     y_test = np.array(self.iscaleY(np.concatenate((lst_outputs),axis=0)))
    #     y_pred = np.array(self.iscaleY(np.concatenate((lst_prediction),axis=0)))
        
    #     d_pred = y_pred - y_test
        
    #     return X_test,y_test,y_pred,d_pred
    
    # def predict_dataset2(self):
    #     lst_prediction=[]
    #     lst_inputs=[]
    #     lst_outputs=[]
    #     for test_inputs, test_outputs in self.test_ds2:
    #         lst_prediction.append( np.array(self.model(test_inputs)))
    #         lst_inputs.append(test_inputs)
    #         lst_outputs.append(test_outputs)
        
        
    #     X_test = np.array(self.iscaleX(np.concatenate((lst_inputs),axis=0)))
    #     y_test = np.array(self.iscaleY(np.concatenate((lst_outputs),axis=0)))
    #     y_pred = np.array(self.iscaleY(np.concatenate((lst_prediction),axis=0)))
        
    #     d_pred = y_pred - y_test
        
    #     return X_test,y_test,y_pred,d_pred
        
    def save_model(self,save_address,save_name= None,err_save =False):
        
        if not save_name:
            save_name = self.name
        weights = self.model.get_weights()
        dump(weights,open( save_address+"/model_"+save_name+".pkl", "wb" ))
        
        if err_save:
            
            all_data = {'pmin':self.pmin,'pmax':self.pmax,'pstp':self.pstp,'slopeX':self.slopeX,'slopeY':self.slopeY,'meanX':self.smeanX,'meanY':self.smeanY,'is_header':self.is_header,
                    'dataset1':self.dataset1,'dataset2':self.dataset2,'parloc':self.parloc,'metloc':self.metricloc,'dsclean':self.ds_clean,'err_a':self.err_a}
        else:
            all_data = {'pmin':self.pmin,'pmax':self.pmax,'pstp':self.pstp,'slopeX':self.slopeX,'slopeY':self.slopeY,'meanX':self.smeanX,'meanY':self.smeanY,'is_header':self.is_header,
                    'dataset1':self.dataset1,'dataset2':self.dataset2,'parloc':self.parloc,'metloc':self.metricloc,'dsclean':self.ds_clean}
            
        dump(all_data,open( save_address+"/init_"+save_name+".pkl", "wb" ))
        pass
    
    def load_model(self,load_address,load_name=None,err_save =False):
        if not load_name:
            load_name = self.name
        all_data = load(open( load_address+"/init_"+load_name+".pkl", "rb" ))
        
        self.pmin = all_data['pmin']
        self.pmax = all_data['pmax']
        self.pstp = all_data['pstp']
        self.slopeX = all_data['slopeX']
        self.slopeY = all_data['slopeY']
        self.smeanX = all_data['meanX']
        self.smeanY = all_data['meanY']
        self.is_header = all_data['is_header']
        self.dataset1 = all_data['dataset1']
        self.dataset2 = all_data['dataset2']
        self.parloc = all_data['parloc']
        self.metricloc = all_data['metloc']
        self.ds_clean = all_data['dsclean']
        x = np.zeros((1,len(self.parloc)))
        self.model(x)
        self.model.set_weights(load(open( load_address+"/model_"+load_name+".pkl", "rb" )))
        if err_save:
            self.err_a = all_data['err_a']
        pass
        
    
    def tl_linear_dataset2(self,loss_object,optimizer,EPOCHS=1000):
        
        tlfit(self.train_ds2,self.test_ds2, self.train_loss,self.test_loss, linear_transferlearning_step,self.model,loss_object,optimizer,EPOCHS)
    

    def tl_nonlinear_dataset2(self,loss_object,optimizer,EPOCHS=1000):
        
        tlfit(self.train_ds2,self.test_ds2, self.train_loss,self.test_loss, nonlinear_transferlearning_step,self.model,loss_object,optimizer,EPOCHS)
    
    def tl_direct_dataset2(self,loss_object,optimizer,EPOCHS=1000):
        
        fit(self.train_ds2,self.test_ds2, self.train_loss,self.test_loss,self.model,loss_object,optimizer,EPOCHS)
    
    
    # def kowalski (self, ypred):
        
    #     error = np.exp(self.err_a*ypred+self.err_b)
    #     return error
    
    # def tfkowalski (self, tf_ypred):
        
    #     error = tf.exp(self.err_a*tf_ypred+self.err_b)
    #     return error
    
    
    # def error_analysis(self):
    #     xt,  yt,  yp,  dp  = self.predict_dataset1()
    
    #     k = len(yt[0,:])
    #     m = int(np.ceil(len(yt)/20))*2
    #     a = np.zeros(k)
    #     b = np.zeros(k)
    #     for i in range(k):
    #         ws = np.argsort(yt[:,i])
    #         ys = yt[ws,i]
    #         ds = dp[ws,i]
    #         sigma_pred = np.zeros(len(dp)-m)
    #         for j in range(len(dp)-m):
    #             wp = ds[j:j+m]
    #             sigma_pred[j] = np.std(wp)
            
    #         x=ys[int(m/2):-int(m/2)]
    #         y=np.log(sigma_pred+1e-21)
    #         a[i],b[i] = linear_reg(x,y)
        
    #     return a,b
    
    
    def error_analysis_poly(self,division=100,err_type='std',poly_order=5,dataset_num=1):
        # generates polynomial fitting of the estimation error vs y value
        # division   : It is the order of moving average filter
        # err_type   : Type of error, can be 'std', 'mae' or 'mse'
        # poly_order : Estimating the error with polynomial order
        # dataset_num: number of dataset for analysis, can be 1,2,3,4
        
        # Output is the polynomial coefficients, 
        # use np.poly1d(out) to generate the required polynomial function.
        _,  yt,  yp,  dp  = self.predict_datasets(number = dataset_num )

    
        k = len(yt[0,:])
        m = int(len(yt)/division)*2
        a = np.zeros((k,poly_order+1))
        # b = np.zeros(k)
        for i in range(k):
            ws = np.argsort(yt[:,i])
            ys = yt[ws,i]
            ds = dp[ws,i]
            sigma_pred = np.zeros(len(dp)-m)
            for j in range(len(dp)-m):
                wp = ds[j:j+m]
                if err_type == 'std':
                    sigma_pred[j] = np.std(wp)
                elif err_type == 'mae':
                    sigma_pred[j] = np.mean(np.abs(wp))
                elif err_type == 'mse':
                    sigma_pred[j] = np.mean(np.var(wp))
                
                
                
                
                
                
            x=ys[int(m/2):-int(m/2)]
            # x=np.expand_dims(x, axis=0)
            # print(np.shape(x))
            y=sigma_pred+1e-21
            # print(np.shape(y))
            b=np.polyfit(x,y,poly_order)
            
            a[i,:] = b
        return a
    
    def error_analysis_score(self,err_type='std',dataset_num=1):
        # generates polynomial fitting of the estimation error vs y value
        # err_type   : Type of error, can be 'std', 'mae' or 'mse'
        # dataset_num: number of dataset for analysis, can be 1,2,3,4
        
        # Output is the polynomial coefficients, 
        # use np.poly1d(out) to generate the required polynomial function.
        _,  _,  _,  dp  = self.predict_datasets(number = dataset_num )

        k = len(dp[0,:])
        a = np.zeros()
        # b = np.zeros(k)
        for i in range(k):
            a[i] = np.mean(np.abs(dp[:,i]))
        return a
    
    def error_analysis_sklearnmodel(self,model, dataset_num=1):
        # dataset_num: number of dataset for analysis, can be 1,2,3,4
        # model is the sklearn regression model used to recognize the error caused by NN model.
        xt,  _,  _,  dp  = self.predict_datasets(number = dataset_num )

        model.fit(xt,dp)        
        
        return model
    
    
    
    def error_analysis_sigmagen(self,  yt,  yp,  dp,division=100,err_type='std'):
        k = len(yt[0,:])
        m = int(len(yt)/division)*2
        
        sigma_pred = np.zeros((len(dp)-m,k))
        x = np.zeros((len(dp)-m,k))
        for i in range(k):
            ws = np.argsort(yt[:,i])
            ys = yt[ws,i]
            ds = dp[ws,i]
            
            for j in range(len(dp)-m):
                wp = ds[j:j+m]
                if err_type == 'std':
                    sigma_pred[j,i] = np.std(wp)
                elif err_type == 'mae':
                    sigma_pred[j,i] = np.mean(np.abs(wp))
                elif err_type == 'mse':
                    sigma_pred[j] = np.mean(np.var(wp))
            
            x[:,i]=ys[int(m/2):-int(m/2)]
            y=sigma_pred+1e-21
        return x,y,m
        
        
 
 

# def linear_reg(x,y):
#     a = sum((x - np.mean(x)) * (y - np.mean(y))) / sum( (x - np.mean(x))**2 )
#     b = np.mean(y) - a * np.mean(x)           
#     return a,b

# def poly_reg(xdata,ydata):
#     # func = lambda x, a : a[0] + a[1]*x+a[2]*x**2+a[3]*x**3+a[4]*x**4+a[5]*x**5
    
    
#     popt, pcov = curve_fit(poly_5, xdata, ydata)
    
#     return popt

# def poly_5(x, a, b, c, d, e, f):
#     return a + b*x+c*x**2+d*x**3+e*x**4+f*x**5




@tf.function
def train_step(inputs, outputs,model,loss_object,optimizer,train_loss):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # optimizer.apply_gradients(gradients, model.trainable_variables)
    train_loss(loss)
    
@tf.function
def linear_transferlearning_step(inputs, outputs,model,loss_object,optimizer,train_loss):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(outputs, predictions)
    w = model.trainable_variables
    neww = [w[0],w[1],w[-2],w[-1]]
    gradients = tape.gradient(loss, neww)
    optimizer.apply_gradients(zip(gradients, neww))
    train_loss(loss)
  
@tf.function
def nonlinear_transferlearning_step(inputs, outputs,model,loss_object,optimizer,train_loss):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_object(outputs, predictions)
    w = model.trainable_variables
    nlinw= w[2:-2]
    ngradients = tape.gradient(loss, nlinw)
    optimizer.apply_gradients(zip(ngradients, nlinw))
    train_loss(loss)
    
  
@tf.function
def test_step(inputs, outputs,model,loss_object,test_loss):
    predictions = model(inputs)
    t_loss = loss_object(outputs, predictions)
    test_loss(t_loss)


def tlfit(train_ds,test_ds,train_loss,test_loss,tlfunction,model,loss_object,optimizer,EPOCHS=1000):
    for epoch in range(EPOCHS):
        for inputs, outputs in train_ds:
            tlfunction(inputs, outputs,model,loss_object,optimizer,train_loss)
    
        for test_inputs, test_outputs in test_ds:
            test_step(test_inputs, test_outputs,model,loss_object,test_loss)
    
        print('%1.0f:, %1.5f,  %1.5f \n'%(epoch+1,train_loss.result(),test_loss.result()))
    
        train_loss.reset_states()
        test_loss.reset_states()
        
        
def tlonlyfit(train_ds,train_loss,tlfunction,model,loss_object,optimizer,EPOCHS=1000):
    for epoch in range(EPOCHS):
        for inputs, outputs in train_ds:
            tlfunction(inputs, outputs,model,loss_object,optimizer,train_loss)
    
        print('%1.0f:, %1.5f \n'%(epoch+1,train_loss.result()))    
        train_loss.reset_states()

def onlyfit(train_ds,train_loss,model,loss_object,optimizer,EPOCHS=1000):
    for epoch in range(EPOCHS):
        for inputs, outputs in train_ds:
            train_step(inputs, outputs,model,loss_object,optimizer,train_loss)
    
        print('%1.0f:, %1.5f \n'%(epoch+1,train_loss.result()))    
        train_loss.reset_states()

def fit(train_ds,test_ds,train_loss,test_loss,model,loss_object,optimizer,EPOCHS=1000):
    for epoch in range(EPOCHS):
        
        tstart1 = time()
        
        for inputs, outputs in train_ds:
            train_step(inputs, outputs,model,loss_object,optimizer,train_loss)
            
        tend1   = time()
        tstart2 = time()
        for test_inputs, test_outputs in test_ds:
            test_step(test_inputs, test_outputs,model,loss_object,test_loss)
        tend2  = time()
        print('EPOCH #%1.0f, Train Loss: %1.5f, Test Loss: %1.5f, Take %1.2f Seconds, and %1.2f Seconds\n'
              %(epoch+1,train_loss.result(),test_loss.result(),(tend1-tstart1),(tend2-tstart2)))
    
        train_loss.reset_states()
        test_loss.reset_states()

