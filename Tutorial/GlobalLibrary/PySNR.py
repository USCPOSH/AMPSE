""" Designed by Mohsen Hassanpourghadi
    9/30/2018
    Module finds SNR, SNDR, THD, SFDR of a signal
    tested on Python v3.5,  numpy v1.14.2
    
    inputs:
        tsignal :  float array. Uniformly sampled signal
        Window  :  'Flat', 'Hamming', 'Hanning', 'Blackman' or 'Kaiser'. Different window type to remove fft leakage.
        fs      :  float. Sampling frequency
        aliased :  0 or 1. 1 If harmonics are aliasing back. 
        harmonics : List or array or range() of integers . SNR calculation with harmonics in the list.
        fin     : None, or float. 
        bins    : integer. Number of signal bins after windowing
    
    Sample code:
        t = np.arange(2**12)
        sp = np.sin(2*t*np.pi*0.01212)+0.01*np.sin(2*t*np.pi*0.01212*4)
        mysignal=signal(sp,window='Kaiser',bins=10)
        mysignal.compiler()
        
        print(mysignal.sndr)
        print(mysignal.snr)
        print(mysignal.sfdr)
        print(mysignal.thd)
    
    Notes:
        Always use .compiler() after changing the attributes of your signal to get correct output:
            mysignal.set_fs(10e9)
            mysignal.complier()
        The function .sigs  returns a list with all the information in of the signal in frequency domain:
            0- The signal information
            1- The DC information
            2- Harmonics if any chosen
            3- Noise
"""
from __future__ import division
import numpy as np
from numpy.fft import fft, fftshift



class Sig_process():
    def __init__(self,tsignal,window='Kaiser',fs=1,aliased=0,harmonics=[2,3,4,5,6], fin=None,bins=8):
        self.tsignal=np.array(tsignal)
        self.window=window
        self.fs=fs
        self.aliased=aliased
        self.harmonics=harmonics
        self.fin=fin
        self.bins=bins
        self.sigs=[]
        self.freqs=[]
        
        
        
        
        
        
        
    def set_window(self,window='Kaiser'):
        self.window=window
    
    def set_fs(self,fs=1):
        self.fs=fs
    
    def set_aliased(self):
        self.aliased=1
    def unset_aliased(self):
        self.aliased=0
    
    def set_harmonics(self,harmonics):
        self.harmonics=harmonics
        
    def set_fin(self,fin):
        self.fin=fin    
    def unset_fin(self):
        self.fin=None
    
    def windowing(self):
        if self.window=='Flat':
            beta=0
        elif self.window=='Hamming':
            beta=5
        elif self.window=='Hanning':
            beta=6
        elif self.window=='Blackman':
            beta=8.6
        else:
            beta=38
        M=len(self.tsignal)
        ary_win=np.kaiser(M,beta)
        return ary_win
    def fouriertransform(self,input_signal):
        return np.square(np.abs((fft(input_signal)))/len(input_signal))
    
    def ssb(self,input_fsignal):
        M=len(input_fsignal)
        return input_fsignal[0:(M+1)//2]
    
    def sig_noise(self,input_ssb,locs):
        M=len(input_ssb)
        N=self.bins
        noise=input_ssb
        OUT=[]
	
        for loc in locs:
            minloc= loc-N if loc>N else 0
            maxloc= loc+N if loc+N<M else M
            sig=np.zeros(M)
            x=range(minloc,maxloc)
            sig[x]=noise[x]
            noise=noise-sig
            OUT.append(sig)
        OUT.append(noise)    
        return OUT
        
        
    def Compile(self):
        
        sig_win=self.windowing()
        fsignal=self.fouriertransform(self.tsignal*sig_win)
        ssbsignal=self.ssb(fsignal)
        
        M=len(ssbsignal)
        self.freqs = np.linspace(self.fs/M,self.fs,M)/2
        
        
        if self.fin:
            xin=np.round(self.fin/self.fs*2*len(ssbsignal))
        else:
            xin=self.bins+np.argmax(ssbsignal[self.bins:])
        
        H=np.array([1,0]);
        for i in self.harmonics:
            if i>=2:
                H=np.append(H,int(i))
                
        H=H*xin
        
        upH=np.array([1,0])*xin
        if self.aliased:
            for i in H:
                frqlr=i//M
                if np.mod(frqlr,2):
                    upH=np.append(upH,i-frqlr*M)
                else:
                    upH=np.append(upH,(frqlr+1)*M-i)
        else:
            upH=H  
        self.sigs=self.sig_noise(ssbsignal,upH.astype(int))            
        pass
            
            
        
    def sndr(self):
        x=self.sigs
        powersig=np.sum(x[0])
        powernoise=np.sum(x[2:])
        return 10*np.log10(powersig/powernoise)
    
        
        
        
    
    def snr(self):
        x=self.sigs
        powersig=np.sum(x[0])
        powernoise=np.sum(x[len(x)-1])
        return 10*np.log10(powersig/powernoise)
        
    
    def sfdr(self):
        x=self.sigs
        powersig=np.max(x[0])
        powernoise=np.max(x[2:])
        return 10*np.log10(powersig/powernoise)

    
    def thd(self):
        x=self.sigs
        powersig=np.sum(x[0])
        powernoise=np.max(x[2:(len(x)-1)])
        return 10*np.log10(powersig/powernoise)
    
    
    
    
    

            
if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    import pandas as pd
      
    t = np.arange(2**12)
    sp = np.floor(512*(np.sin(2*t*np.pi*0.01212)+0.01*np.sin(2*t*np.pi*0.01212*4)))
    
    dataset = pd.read_csv('D:\ProjectPhD\Matlab\Calculating\ADC_result\PYADC_v1.csv',header=None)
    sp=dataset.iloc[0,:].values
    
    mysignal=Sig_process(sp,window='Kaiser',bins=10)
    mysignal.Compile()
    print(mysignal.snr())
    print(mysignal.sndr())
    print(mysignal.sfdr())
    print(mysignal.thd())
    
    sig=mysignal.sigs
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[0]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[1]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[2]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[3]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[4]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[5]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[6]))
    plt.plot(np.arange(len(sp)/2),10*np.log10( sig[7]))
    #plt.plot(mysignal.tsignal)
    #plt.plot(np.arange(len(t)/2),10*np.log10(myssbsignal))
    #plt.show()

    #np.sum(sig)/np.sum(noise)
