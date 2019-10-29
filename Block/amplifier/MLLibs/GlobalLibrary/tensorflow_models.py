#Designed by Mohsen Hassanpourghadi
#Date May 2 2019
#Introduces classes that uses tensorflow for learning.



########################## Initialization ##########################

import numpy as np
import tensorflow as tf
import pickle
#import time
#from scipy.io import savemat 

############################# Functions ############################

def make_var(var_scope, var_name, var_shape, var_init):
    try:
        with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
            return tf.get_variable(shape=var_shape,dtype=tf.float32,name=var_name,initializer=var_init)
    except:
        print("Can't get the variables, since they are initialized with different shapes, if you want to change the shapes try this code first: tf.reset_default_graph()")
    
def glorot_init(shape):
    #return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
    #return tf.random_uniform_initializer(np.array(self.sminx),np.array(self.smaxx))
    return tf.random_normal_initializer(np.zeros(shape),np.ones(shape))

        

########################### The Model Class ########################
class TF_MODELS():
    
    
    def __init__(self,name='mohsen',scope='dudes',separation=False):
        self.layers=[2,2,2,1]
        self.actfuncs=['relu','relu']
        self.name=name
        self.scope=scope
        self.llayers=len(self.layers)
        self.cost=tf.constant(0)
        self.separation=separation
        pass
    
    def learn_init(self,layers,actfuncs):
        # Initializes the learning graphs
        # layers: a list containing the number of inputs, neuorons in each hidden layer, and the number of outputs
        # actfuncs: a list, containing the activation function of each layer.
        self.layers=layers[:]
        self.llayers=len(layers)
        self.actfuncs=actfuncs[:]
        lastfunc=actfuncs[-1]
        if self.llayers<2:
            print('Warning! : number of layers must be more than two!')
            return 0
        while self.llayers-2>len(self.actfuncs):
            self.actfuncs.append(lastfunc)
        self.actfuncs.append('linear')
        
        
        self.tf_w=[]
        nout=self.layers[-1]
        
        
        # Weights and Bias definition:
        if self.separation:
            
            for j in range(nout):
                tf_sw=[]
                for i in range(self.llayers-2):
                    tf_sw.append(make_var(self.scope,self.name+'_ws_'+str(j)+str(i),
                                          (self.layers[i],self.layers[i+1]),glorot_init((self.layers[i],self.layers[i+1]))))
                    tf_sw.append(make_var(self.scope,self.name+'_bs_'+str(j)+str(i),
                                          (self.layers[i+1],),glorot_init((self.layers[i+1],))))
                tf_sw.append(make_var(self.scope,self.name+'_ws_'+str(j)+str(i+1),
                                      (self.layers[i],1),glorot_init((self.layers[i],1))))
                tf_sw.append(make_var(self.scope,self.name+'_bs_'+str(j)+str(i+1),
                                      (1,),glorot_init((1,))))
                self.tf_w.append(tf_sw)
        else:
                               
            for i in range(self.llayers-1):
                self.tf_w.append(make_var(self.scope,self.name+'_w_'+str(i),(self.layers[i],self.layers[i+1]),glorot_init((self.layers[i],self.layers[i+1]))))
                self.tf_w.append(make_var(self.scope,self.name+'_b_'+str(i),(self.layers[i+1],),glorot_init((self.layers[i+1],))))
            
        
        
        
        
        # Input Placeholders:
        self.sin  = tf.placeholder(tf.float32,shape=[None,self.layers[0]],name=self.name+'_IN')
        self.sout = tf.placeholder(tf.float32,shape=[None,self.layers[-1]],name=self.name+'_OUT')
        
        # Build NN
        self.sy=self.forward(self.sin)  
        
        return 1

        
    def forward(self,sin):
        # Model's forward propagation
        # sin: Input to the forward propagation use standardize inputs
        if self.separation:
            outs=[]
            for tf_sw in self.tf_w:
                row=[sin]
                for i in range(self.llayers-1):
                    x=tf.add( tf.matmul(row[i],tf_sw[2*i]),tf_sw[2*i+1] )
                    if self.actfuncs[i]=='relu':
                        row.append(tf.nn.relu(x))
                    elif self.actfuncs[i]=='elu':
                        row.append(tf.nn.elu(x))
                    elif self.actfuncs[i]=='sigmoid':
                        row.append(tf.nn.sigmoid(x))
                    elif self.actfuncs[i]=='linear':
                        row.append(x)
                    else:    
                        row.append(x)
                
                outs.append(row[-1])
            sout=tf.stack(outs)
            return sout
        else:
            row=[sin]
            for i in range(self.llayers-1):
                x=tf.add( tf.matmul(row[i],self.tf_w[2*i]),self.tf_w[2*i+1] )
                if self.actfuncs[i]=='relu':
                    row.append(tf.nn.relu(x))
                elif self.actfuncs[i]=='elu':
                    row.append(tf.nn.elu(x))
                elif self.actfuncs[i]=='sigmoid':
                    row.append(tf.nn.sigmoid(x))
                elif self.actfuncs[i]=='linear':
                    row.append(x)
                else:    
                    row.append(x)
            
            sout=row[-1]
            return sout
        

    

    
    def learn_cost(self, costtype='mse',minimizer='gd',lr=0.1 ):
        # Cost function and optimizer definition
        # Cost function type, 
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if minimizer=='gd':
                self.trainer=tf.train.GradientDescentOptimizer(learning_rate=lr)
            elif minimizer=='adadelta':
                self.trainer=tf.train.AdadeltaOptimizer(learning_rate=lr)
            elif minimizer=='rmsprop':
                self.trainer=tf.train.RMSPropOptimizer(learning_rate=lr)
            elif minimizer=='adagrad':
                self.trainer=tf.train.AdagradOptimizer(learning_rate=lr)
            elif minimizer=='adam':
                self.trainer=tf.train.AdamOptimizer(learning_rate=lr) 
            else:
                self.trainer=tf.contrib.opt.NadamOptimizer(learning_rate=lr)


        if self.separation:
            self.cost=[]
            self.opt=[]
            for i in range(self.layers[-1]):
                if costtype=='mse':
                    self.cost.append(tf.reduce_mean(tf.square(self.sy[i]-self.sout[i])))
                elif costtype=='mle':
                    self.cost.append(tf.reduce_mean(tf.abs(self.sy[i]-self.sout[i])))
                else:
                    self.cost.append(tf.reduce_sum(tf.square(self.sy[i]-self.sout[i])))
                self.opt.append(self.trainer.minimize(self.cost[i], var_list=self.tf_w[i]))
                    
        else:
                
            if costtype=='mse':
                self.cost=tf.reduce_mean(tf.square(self.sy-self.sout))
            elif costtype=='mle':
                self.cost=tf.reduce_mean(tf.abs(self.sy-self.sout))
            else:
                self.cost=tf.reduce_sum(tf.square(self.sy-self.sout))
            self.opt=self.trainer.minimize(self.cost, var_list=self.tf_w)
        
        
        
        
    def tf_train(self,sX,sY,sess,epochs=10, batch=100, verbose=True):
        # It should be used inside the session:
        # Equivalent to fit, it trains the regressor
        # sX is the normalized input
        # sY is the normalized output
        # epochs is the number of iteration
        # batch size is the batch size 
        # verbose, if true it shows the output
        
        
        lbatch=int(np.ceil(len(sX)/batch))
        Sep=0
        if type(self.cost)==list:
            if not type(epochs)==list:
                Sep=1
            else:
                while len(epochs)<len(self.cost):
                    epochs.append(epochs[-1])
                Sep=2
        else:
            Sep=0
                
        if Sep<2:
            for i in range(epochs):
                loss=0
                for j in range(lbatch):
                    batch_x = sX[j*batch:(j+1)*batch,:]
                    batch_y = sY[j*batch:(j+1)*batch,:]
    
                    feed_dict = {self.sin: batch_x, self.sout: batch_y}
                    
                    gl, vl = sess.run([self.cost, self.sy],feed_dict=feed_dict)
                    sess.run(self.opt,feed_dict=feed_dict)    
                loss=loss+gl/lbatch
                if verbose:
                    print('%1.0f: loss = %1.5f\n' % (i, loss))
                
        else:
            loss=[]
            for k in range(len(self.cost)):
                for i in range(epochs[k]):
                    one_loss=0
                    for j in range(lbatch):
                        batch_x = sX[j*batch:(j+1)*batch,:]
                        batch_y = sY[j*batch:(j+1)*batch,:]
                        feed_dict = {self.sin: batch_x, self.sout: batch_y}
                        gl, vl = sess.run([self.cost[k], self.sy],feed_dict=feed_dict)
                        sess.run(self.opt,feed_dict=feed_dict)    
                    
                    one_loss=one_loss+gl/lbatch        
                    if verbose:
                        print('output: %1.0f, epoch: %1.0f, loss = %1.5f\n' % (k, i, one_loss))
                loss.append(one_loss)
                    
                
        w=sess.run(self.tf_w)
        return w,loss
    
    def tf_predict(self,sX,sess):
        # The regressor's prediction:
        # sX is the input values to be predicted
        # sess is the session
        feed_dict={self.sin: sX}
        
        sout=sess.run(self.sy,feed_dict=feed_dict)
        
        return sout
        
    
    def tf_saveweights(self,filename,sess):
        w=sess.run(self.tf_w)
        pickle.dump( w, open( filename, "wb" ) )
        pass
    def tf_loadweights(self,filename,sess):
        w = pickle.load( open( filename, "rb" ) )
        for i in range(len(self.tf_w)):
            self.tf_w[i].load(w[i], sess)
        pass
    
    def saveall(self,filename):
        
        
        dict_save={'layers':self.layers,'name':self.name,'scope':self.scope,
                   'actfuncs':self.actfuncs,'cost':self.cost,'sep':self.separation}
        pickle.dump(dict_save, open(filename, "wb"))
    
    def loadall(self,filename):
        dict_load=pickle.load(open(filename,"rb"))
        self.layers=dict_load['layers']
        self.name=dict_load['name']
        self.scope=dict_load['scope']
        self.actfuncs=dict_load['actfuncs']
        self.cost=dict_load['cost']
        self.separation=dict_load['sep']
        self.llayers=len(self.layers)
        self.learn_init(self.layers,self.actfuncs)
        
    def pre_fitnormalizer(self,X,Y):
        
        # sX= X*scale_+bias_
        # sY= Y*scale_+bias_
        # -1<sX,sY<+1
        self.minx=np.min(X)
        self.maxx=np.max(X)
        self.scale_=2/(self.maxx-self.minx)
        self.bias_=-(self.maxx+self.minx)/(self.maxx-self.minx)
        
        miny=np.min(Y)
        maxy=np.max(Y)
        self.scaley_=2/(maxy-miny)
        self.biasy_=-(maxy+miny)/(maxy-miny)
        
        sX=X*self.scale_+self.bias_
        sY=Y*self.scaley_+self.biasy_
        
        return sX, sY
    
    def pre_normalizer(self,X,Y):
        
        sX=X*self.scale_+self.bias_
        sY=Y*self.scaley_+self.biasy_
        
        return sX, sY
    
     
    
    
    
        
    
        
    

    
    
    
    


if __name__ == '__main__':
    
    import pandas as pd
    
    
    tf.reset_default_graph()
    
    
    #==================================================================
    #********************* Importing Data *****************************
    #==================================================================
    dataset = pd.read_csv('/home/mohsen/PYTHON_PHD/CADtoolForRegression/OpampSimp/PYDATA2_65.csv',header=None)
    headero=['gmb','gm','power','ron','rop','vdsn','vdsp','gain CM','gain vd', 'cin', 'ugb intrinsic','ugb']
    headeri=['ln','lp','lt','vn','vp','vt','wn','wp','wt','cl']
    dataset=dataset.dropna()
    X = np.array(dataset.iloc[:, 0:10].values,dtype='float64')
    yupdated=np.array(dataset.iloc[:, 10:22].values,dtype='float64')
    
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    sc_X = MinMaxScaler(feature_range=(-1,1))
    sc_y = StandardScaler()
    X_new = sc_X.fit_transform(X)
    y_new = sc_y.fit_transform(yupdated)
    
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size = 0.5)
    
    
    #==================================================================
    #********************  Learning  **********************************
    #==================================================================    
    layers=[10,205,205,205,12]
    actfunc=['elu' ]
    reg=TF_MODELS(name='simpamp65',scope='opamp',separation=False)
    reg.learn_init(layers=layers,actfuncs=actfunc)
    reg.learn_cost(costtype='mse',minimizer='adam',lr=0.001)
    
    
    writer = tf.summary.FileWriter('my_graph')
    
    init = tf.global_variables_initializer()
    #    with tf.Session() as sess:
    #        sess.run(init)
    with tf.Session() as sess:
        sess.run(init)
        #abas.tf_loadweights('weights.p',sess)
        #print(sess.run(abas.tf_w))
        w,loss=reg.tf_train(X_train,y_train,sess,epochs=2000)
        reg.tf_saveweights('weights.p',sess)
        
        
        y_pred=reg.tf_predict(X_test,sess)
        writer.add_graph(sess.graph)
        
        
        

