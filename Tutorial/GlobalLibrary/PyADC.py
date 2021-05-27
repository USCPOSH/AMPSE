from __future__ import division

import numpy as np



def ADC_static_ramp(signal,maxcode=None,mincode=None):
    
    codes=np.array(signal)
    codes.reshape([-1,])
    if maxcode==None:
        maxcode=np.max(codes)
    if mincode==None:
        mincode=np.min(codes)
    

    # Motonoticity check:    
    diffcode=np.sign(np.diff(codes))
    mono_up=np.sum(diffcode==1)
    mono_down=np.sum(diffcode==-1)
    monotonic=[]
    if (mono_up>mono_down) & mono_down>0:
        monotonic=[i for i in range(len(diffcode)) if diffcode[i]==-1]
    elif (mono_down>=mono_up)& mono_up>0:
        monotonic=[i for i in range(len(diffcode)) if diffcode[i]==1]
            
    #DNL and INL:
    unique, counts = np.unique(codes, return_counts=True)
    allpoints=np.sum(counts[1:-1])
    #avgpoints=np.mean(counts[1:-1])
    avgpoints=allpoints/(maxcode-mincode-1)
    LSB=avgpoints/allpoints
    
    DNL=counts[1:-1]/allpoints-LSB
    INL=np.cumsum(DNL)
    
    # missing code check:
    z=np.abs(np.diff(unique))
    missingcodes=[i for i in range(len(z)) if z[i]>1]
    

    return [DNL,INL,monotonic,missingcodes]




#import matplotlib.pyplot as plt


#v=np.floor(np.arange(-1,1,0.0001)*(1+0.5*np.cos(np.pi*np.arange(-1,1,0.0001)))*1024)
#z=ADC_static_sin(v)
#plt.plot(z[1])
