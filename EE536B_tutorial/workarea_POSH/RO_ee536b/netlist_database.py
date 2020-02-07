# This script shows the netlist database for the Inverter Database Using the new version of spectreIOlib
# Example for EE536B tutorial
# Qiaochu Zhang, from Mike Chen's Mixed-Signal Group, Ming Hsieh Dept. of ECE, USC
# 02/06/2020

import sys
import numpy as np
import pandas as pd
import os
home_address  = os.getcwd()
from spectreIOlib import TestSpice, Netlists
import time

 

class inv(Netlists):# change the name of the class to the circuit that you are exploring
    
    def __init__(self, testfolder =None):

        self.testbench = home_address + '/netlists_sanitized/tb_inv.scs'# change netlist file name
        self.testfolder = home_address + '/temp/tb_inv' if testfolder ==None else home_address + '/temp/' + testfolder # create a folder in /temp and name by your circuit
            
        self.minpar  = np.array([100e-9,100e-9 ])# parameter lower bound
        self.maxpar  = np.array([1000e-9,1000e-9 ])# parameter upper bound
        self.stppar  = np.array([100e-9,100e-9])# sampling step
        
        self.par_line_number = 7 # the line number of your design parameters in the netlist, you can check tb_inv.scs line 8 in this case. Note that python starts with zero.
        self.parname = ['wn','wp'] # name of your design parameters
        self.metricname = ['delay'] # name of your design metrics
        self.make_metrics()
        
        self.finaldataset = home_address + '/datasets/tb_inv.csv' # path of your training dataset. You should change the name to your circuit's.
            


            
              
    
    def make_metrics(self):
        z = [10] # the line number of your mesurement result in test.measure file. You can find it in /temp/tb_inv/test.measure
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
    

    INV = inv() # create an object belongs to the class that you defined
    INV.put_on_csv(tedad=1,outcsv = home_address + '/datasets/tb_inv.csv',do_header = True) # tedad stands for the number of samples you want to generate. You also need to change the path which stores the training dataset.
