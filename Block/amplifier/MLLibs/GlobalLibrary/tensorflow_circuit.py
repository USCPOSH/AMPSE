import numpy as np

from sklearn.externals import joblib
from keras.models import model_from_json 
import tensorflow as tf
import pickle


def make_var(var_scope, var_name, var_shape, var_init):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        return tf.get_variable(shape=var_shape,dtype=tf.float32,name=var_name,initializer=var_init)
    
def make_reg(jsonfile,hfile)    :
    json_file = open(jsonfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    reg = model_from_json(loaded_model_json)
    reg.load_weights(hfile)
    return reg

class TF_DEFAULT():
    def __init__(self):
        pass
    def loading(self,sx_f,sy_f,w_f):
             
        with open(w_f,'rb') as f:
            self.weights=pickle.load(f)
        sc_X=joblib.load(sx_f) 
        sc_y=joblib.load(sy_f)
        
        self.scXmean=(sc_X.data_max_+sc_X.data_min_)/2
        self.scXscale=sc_X.scale_
        self.scYmean=sc_y.mean_
        self.scYscale=sc_y.scale_
            
        self.nnnl=len(self.weights)//2
        self.sminx=(self.minx-self.scXmean)*self.scXscale
        self.smaxx=(self.maxx-self.scXmean)*self.scXscale
        self.nparam=len(self.scXscale)
        self.nmetric=len(self.scYscale)
        self.tf_scXmean=tf.constant(self.scXmean,tf.float32)
        self.tf_scXscale=tf.constant(self.scXscale,tf.float32)
        self.tf_scYmean=tf.constant(self.scYmean,tf.float32)
        self.tf_scYscale=tf.constant(self.scYscale,tf.float32)
        
    def tf_reg_elu(self,sxin):
        WS=[];
        for items in self.weights:
            WS.append(tf.constant(items,tf.float32))
                
        cxin=tf.clip_by_value(sxin,self.sminx,self.smaxx)
        row=[cxin]
        for i in range(self.nnnl-1):
            row.append(tf.nn.elu(tf.matmul(row[i],WS[2*i])+WS[2*i+1]))
        i=self.nnnl-1    
        row.append(tf.matmul(row[i],WS[2*i])+WS[2*i+1])        
        return row[self.nnnl]

    def tf_reg_relu(self,sxin):
        WS=[];
        for items in self.weights:
            WS.append(tf.constant(items,tf.float32))
                
        cxin=tf.clip_by_value(sxin,self.sminx,self.smaxx)
        row=[cxin]
        for i in range(self.nnnl-1):
            row.append(tf.nn.relu(tf.matmul(row[i],WS[2*i])+WS[2*i+1]))
        i=self.nnnl-1    
        row.append(tf.matmul(row[i],WS[2*i])+WS[2*i+1])        
        return row[self.nnnl]
    
    def tf_reg_sigmoid(self,sxin):
        WS=[];
        for items in self.weights:
            WS.append(tf.constant(items,tf.float32))
                
        cxin=tf.clip_by_value(sxin,self.sminx,self.smaxx)
        row=[cxin]
        for i in range(self.nnnl-1):
            row.append(tf.nn.sigmoid(tf.matmul(row[i],WS[2*i])+WS[2*i+1]))
        i=self.nnnl-1    
        row.append(tf.matmul(row[i],WS[2*i])+WS[2*i+1])        
        return row[self.nnnl]
    
    
    def tf_init(self):
        return tf.random_uniform_initializer(np.array(self.minx),np.array(self.maxx))
        
    def tf_sinit(self):
        return tf.random_uniform_initializer(np.array(self.sminx),np.array(self.smaxx))
    
    def tf_sinit2(self,chosen):
        xmin=self.sminx[chosen>0]
        xmax=self.smaxx[chosen>0]
        return tf.random_uniform_initializer(np.array(xmin),np.array(xmax))
    
    def tf_scalex(self,xin):
        return tf.multiply(xin-self.tf_scXmean,self.tf_scXscale)
        
    def tf_rescalex(self,sxin):
        return tf.divide( sxin,self.tf_scXscale)+self.tf_scXmean
        
    def tf_rescaley(self,syin):
        return tf.multiply(syin, self.tf_scYscale)+self.tf_scYmean
    
    def tf_scaley(self,yin):
        return tf.divide( yin-self.tf_scYmean,self.tf_scYscale)
    
    def tf_scalex2(self,xin,chosen):    
        xmean  = tf.constant(self.scXmean[chosen>0] ,tf.float32)
        xscale = tf.constant(self.scXscale[chosen>0],tf.float32)
        return tf.multiply(xin-xmean,xscale)
    
    def np_scalex(self,xin):
        return (xin-self.scXmean)*self.scXscale
        
    def np_rescalex(self,sxin):
        return sxin/self.scXscale+self.scXmean
        
    def np_rescaley(self,syin):
        return syin*self.scYscale+self.scYmean
    
    def np_scaley(self,yin):
        return ( yin-self.scYmean)/self.scYscale
    
    def np_scalex2(self,xin,chosen):    
        xmean  = self.scXmean[chosen>0]
        xscale = self.scXscale[chosen>0]
        return (xin-xmean)*xscale
    
    
    def assign_minmax(self,minx,maxx):
        self.minx=minx
        self.maxx=maxx
        self.sminx=(self.minx-self.scXmean)*self.scXscale
        self.smaxx=(self.maxx-self.scXmean)*self.scXscale
        pass
    def forward(self,xin):
        sxin = self.tf_scalex(xin)
        sy   = self.tf_reg_elu(sxin)
        y    = self.tf_rescaley(sy)
        return y
    



def action2sxin(action,sxin,ssin):
    # action can be 0 ---  23
    n_action = len(sxin[0])
    signact=1.0
    vectoract = np.zeros_like(sxin)
    if action> (n_action-1):
        action=action-n_action
        signact=-1.0
    vectoract[0,action] = np.multiply(signact,ssin[0,action])        
    sxout = sxin + vectoract
    return sxout,vectoract
    
def rw1(n_action):
    action  =  np.random.randint(2*n_action)
    return action


def rw2(n_action,bad_action):
    action = np.random.randint(2*n_action-1)
    if action>=bad_action:
        action=action+1
    return action


def rw3(n_action,bad_action):
    if bad_action<2*n_action:
        action = np.mod(bad_action+n_action,2*n_action)
    else:
        action = rw1(n_action) 
    return action

def vector_constraints(prev_const,new_const,vec_reward):
    p_c = np.array(prev_const)
    n_c = np.array(new_const)
    reward = sum(-vec_reward*(n_c-p_c))
    return reward

def np_elu(x):
    y = np.ones_like(x)
    y[x>0] = x[x>0]
    y[x<=0] = np.exp(x[x<=0])-1
    return y

def np_sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def np_sigmoid_inv(x):
    y = - np.log(1/x-1)
    return y
    
    
