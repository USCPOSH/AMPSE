# This script shows the netlist database for the VCO-ADC Database Using the new version of spectreIOlib
# It demands Mentor Graphics: Analog Fast Spice to run netlists



import sys
sys.path.insert(0,'/GlobalLibrary')



import os
from spectreIOlib import TestSpice, Netlists
home_address  = os.getcwd()

import numpy as np
import pandas as pd


polynum=7
lside=10
rside=5
wside=10
    
    
     
class VCOSpice(Netlists):
    
    def __init__(self, tech = 65,testfolder=None):
        if tech ==65:
            self.testbench = home_address + '/netlists/VCO_testbenchstatic_TT.scs'
            self.testfolder = home_address + '/temp/TrashVCO_1_1' if testfolder ==None else home_address + '/temp/' + testfolder
            self.minpar  = np.array([60e-9 ,0.2e-6 ,2  ,0.2e-6,2  ,2000 ,0.9  ,1.0 ])
            self.maxpar  = np.array([60e-9 ,1.2e-6 ,20 ,1.2e-6,20 ,2000 ,0.9  ,1.0 ])
            self.stppar  = np.array([1e-9  ,10e-9  ,1  ,10e-9 ,1  ,1    ,0.1  ,0.1 ])
            self.parname = ['lastt','wnnn','fnnn','wpppn','fppp', 'rres','VBIAS','VDD']
            self.metricname = ['power','vcm','vfs','fnoise','f1','f2','f3','f4','f5','f6','f7','f8']
            self.par_line_number = 7
            self.lst_metrics=[{'read':'c','filename':self.testfolder + '/test.out/test.measure','number':2,'measurerange':range(9,10)},
                              {'read':'c','filename':self.testfolder + '/test.out/test_cont_vout1.mt0','number':3,'measurerange':range(4,600)},
                              {'read':'c','filename':self.testfolder + '/test.out/test_cont_tout1.mt0','number':3,'measurerange':range(4,600)}]

            
            self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},verbose = True)
#            self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},verbose = True)
            self.testbench2 = home_address + '/netlists/VCO_testbenchnoise_TT.scs'
            self.runspectre2=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench2,'trashfolder':self.testfolder},verbose = True)
#            self.runspectre2=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench2,'trashfolder':self.testfolder},verbose = True)
            self.finaldataset = home_address + '/datasets/PY_VCO01_TT.csv'
            
            
      
    
    def normal_run(self,param):
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters)
        if x:
            out1=[]
        else:
            out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1
    
    
    def analysis(self,lst_out):
        time_edges=np.array(lst_out[2])
        v_edges=np.array(lst_out[1])
        prd_edge=(time_edges[wside:]-time_edges[:-wside])/wside
        v_edges=v_edges[:-wside]    
        v_edges=v_edges[lside:-rside]
        prd_edge=prd_edge[lside:-rside]
        x=v_edges
        y=1/prd_edge
        cof = np.polyfit(x, y, polynum)
        vcm=(v_edges[0] + v_edges[-1])/2
        vfs=(v_edges[-1]- v_edges[0] )/2
        vs = np.linspace(vcm-vfs,vcm+vfs,8)
        fout=cof[7]+cof[6]*vs +cof[5]*vs**2+cof[4]*vs**3+cof[3]*vs**4+cof[2]*vs**5+cof[1]*vs**6+cof[0]*vs**7
        
        return [0,vcm,vfs,0]+list(fout)
    
    def secondrun(self,vcm):
        self.dict_parameters['value_params'][6]=vcm
        self.dict_parameters['value_params'][0]=30e-9
        
        x = self.runspectre2.runspectre(dict_parameters=self.dict_parameters)
        if x:
            out2=[]
        else:
            out2 = self.runspectre2.readmetrics(lst_metrics=self.lst_metrics)
        
        return out2

    def analysis2(self,lst_out,first_out):
        time_edge=np.array(lst_out[2])
        prd_edge=(time_edge[wside:]-time_edge[:-wside])/wside
        freq_edge=1/prd_edge
        power = lst_out[0][0]
        final_out = first_out
        final_out[0] = power
        final_out[3] = np.std(freq_edge)
        return final_out
    
    def wholerun_normal(self,param):
        x = self.normal_run(param)
        y = self.analysis(x)
        z = self.secondrun(y[1])
        w = self.analysis2(z,y)        
        return w



class INBUF2Spice(Netlists):
    
    def __init__(self, tech = 65,testfolder=None):
        if tech ==65:
            self.testbench  = home_address + '/netlists/input_buf2.scs'
            self.testfolder = home_address + '/temp/TrashINBUF2_1_1' if testfolder ==None else home_address + '/temp/' + testfolder
            self.minpar  = np.array([1      ,1        , 60e-9 ,1         ,0.55 ,0.0  ,0.2e-6 , 2     ])
            self.maxpar  = np.array([20     ,50       ,400e-9 ,50        ,0.90 ,0.0  ,1.2e-6 , 20    ])
            self.stppar  = np.array([1      ,1        , 10e-9 ,1         ,0.01 ,0.01 ,10.0e-9, 1     ])
            self.parname =          ['multi','fing_in','l_ttt','fing_ttt','VCM','dvv','wpppp','fpppp']
            self.metricname = ['power','gain','bw','outvcm', 'avcm','kickn','irn','outn3','outn1','outp1','outp3']
            self.par_line_number = 7
            self.lst_metrics=[{'read':'c','filename':self.testfolder + '/test.out/test.measure','number':2,'measurerange':range(10,17)},
                               {'read':'c','filename':self.testfolder + '/test.out/test.measure','number':5,'measurerange':[21,29,37,45]}]
            self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},verbose = True)
            self.finaldataset = home_address + '/datasets/PY_INBUF201_TT.csv'

    def normal_run(self,param):
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters)
        if x:
            out1=[]
        else:
            out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1  
        
    def analysis(self,lst_out):
        z=lst_out[0]
        pwr=z[0]
        gain=z[1]
        bw=z[2]
        outvcm=z[3]
        avcm=z[4]
        kickn=z[5]
        irn=z[6]
        z=lst_out[1]
        
        return [pwr,gain,bw,outvcm,avcm,kickn,irn]+z


    def wholerun_normal(self,param):
        x = self.normal_run(param)
        w = self.analysis(x)
        return w

if __name__ == '__main__':
    
    
    myinbuf2 = INBUF2Spice()
    p,m = myinbuf2.wholerun_random()
    
    print('p is :')
    print(p)
    print('m is :')
    print(m)
