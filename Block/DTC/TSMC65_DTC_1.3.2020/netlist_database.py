# This script shows the netlist database for the VCO-ADC Database Using the new version of spectreIOlib
import sys
import numpy as np
import pandas as pd
import os
home_address  = os.getcwd()
sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
from spectreIOlib import TestSpice, Netlists
import time

 

class DTC1_spice(Netlists):
    
    def __init__(self, tech = 65, testfolder =None):

        self.testbench = home_address + '/netlists_desanitized/tb_DTC_PNinj.scs'
        self.testfolder = home_address + '/temp/tb_DTC_PNinj' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
        self.minpar  = np.array([10 ,10 ,2000 ,10 ])
        self.maxpar  = np.array([21 ,21 ,4000 ,21 ])
        self.stppar  = np.array([1  ,1  ,100 ,1])
        
        self.par_line_number = 7
        self.parname = ['nf_cap','nf_load','res','nf_dif']
        self.metricname = ['delay','delay0','trf_full','trf_zero']
        self.make_metrics()
        
        self.finaldataset = home_address + '/datasets/tb_DTC_PNinj.csv'
            


            
              
    
    def make_metrics(self):
        z = [10,11,12,13]
        self.lst_metrics=[{'read':'c','filename':self.testfolder + '/test.measure','number':2,'measurerange':z}]
        
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},verbose = True)
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
       
        return z

    def normal_run(self,param,lst_alter=[]):

        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter)
        if x:
            out1=[]
        else:
            out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param):
        x = self.normal_run(param)
        w = self.analysis(x)       
        return w  



class DTC2_spice(Netlists):
    
    def __init__(self, tech = 65, testfolder =None):

        self.testbench = home_address + '/netlists_desanitized/tb_DTC_2nd_stage.scs'
        self.testfolder = home_address + '/temp/tb_DTC_2nd_stage' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
        self.minpar  = np.array([100e-12 ,10 ,2000 ,10 ])
        self.maxpar  = np.array([300e-12 ,21 ,4000 ,21 ])
        self.stppar  = np.array([10e-12  ,1  ,100 ,1])
        
        self.par_line_number = 7
        self.parname = ['trf','nf_load','res','nf_dif']
        self.metricname = ['delay','trf']
        self.make_metrics()
        
        self.finaldataset = home_address + '/datasets/tb_DTC_PNinj.csv'
            


            
              
    
    def make_metrics(self):
        z = [10,11]
        self.lst_metrics=[{'read':'c','filename':self.testfolder + '/test.measure','number':2,'measurerange':z}]
        
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},verbose = True)
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
       
        return z

    def normal_run(self,param,lst_alter=[]):

        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter)
        if x:
            out1=[]
        else:
            out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param):
        x = self.normal_run(param)
        w = self.analysis(x)       
        return w  


if __name__ == '__main__':
    

    DTC1 = DTC1_spice(65)
    DTC1.put_on_csv(tedad=1000,outcsv = home_address + '/datasets/tb_DTC_PNinj.csv',do_header = True)
    DTC2 = DTC2_spice(65)
    DTC2.put_on_csv(tedad=1000,outcsv = home_address + '/datasets/tb_DTC_2nd_stage.csv',do_header = True) 

    #p,w = myseqp2.wholerun_random()
#    myseqp2.put_on_csv(tedad=200,outcsv = myseqp2.finaldataset,do_header = False)
