# This script shows the netlist database for the VCO-ADC Database Using the new version of spectreIOlib
import sys
import numpy as np
import pandas as pd
import os
home_address  = os.getcwd()
sys.path.insert(0, home_address+'/MLLibs/GlobalLibrary')
from spectreIOlib import TestSpice, Netlists
import time

 

class Folded_Cascode_spice(Netlists):
    
    def __init__(self, tech = 45, testfolder =None):
        if tech ==45:
            self.testbench = home_address + '/Netlists/foldedcascode_cmfb_PTM45nm.scs'
            self.testfolder = home_address + '/Garbage/foldedcascode_cmfb_45' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
            self.minpar  = np.array([  45e-9, 45e-9, 45e-9, 45e-9, 45e-9, 45e-9, 45e-9, 400e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            self.maxpar  = np.array([  600e-9, 600e-9, 600e-9, 600e-9, 600e-9, 600e-9, 600e-9,  600e-3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1e-13])
            self.stppar  = np.array([  10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 20e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            
            self.par_line_number = 7
            self.parname =          [ 'lbias','lbp','lbn','lin1','lin2','ltn','ltp','vcmo','mamp','fbias','fbp','fbn','fin1','fin2','ftn1','ftn2','ftp1','ftp2','cload']				
            self.metricname = ['cin', 'cout', 'gain', 'gm', 'pole1', 'pole2', 'rout', 'cmo', 'pwr', 'swing14', 'swing7', 'swingn', 'swingn1', 'swingn4', 'swingp', 'invn'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/foldedcascode_cmfb_PTM45.csv'

        if tech ==32:
            self.testbench = home_address + '/Netlists/foldedcascode_cmfb_PTM32nm.scs'
            self.testfolder = home_address + '/Garbage/foldedcascode_cmfb_32' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
            self.minpar  = np.array([  32e-9, 32e-9, 32e-9, 32e-9, 32e-9, 32e-9, 32e-9, 400e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            self.maxpar  = np.array([  450e-9, 450e-9, 450e-9, 450e-9, 450e-9, 450e-9, 450e-9, 600e-3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1e-13])
            self.stppar  = np.array([  10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 10e-9, 20e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            
            self.par_line_number = 7  
            self.parname =          [ 'lbias','lbp','lbn','lin1','lin2','ltn','ltp','vcmo','mamp','fbias','fbp','fbn','fin1','fin2','ftn1','ftn2','ftp1','ftp2','cload']				
            self.metricname = ['cin', 'cout', 'gain', 'gm', 'pole1', 'pole2', 'rout', 'cmo', 'pwr', 'swing14', 'swing7', 'swingn', 'swingn1', 'swingn4', 'swingp', 'irn'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/foldedcascode_cmfb_PTM32nm.csv'    
            
        if tech ==14:
            self.testbench = home_address + '/Netlists/foldedcascode_cmfb_PTM14nm.scs'
            self.testfolder = home_address + '/Garbage/foldedcascode_cmfb_14' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
            self.minpar  = np.array([  10e-9, 10e-9, 10e-9,10e-9, 10e-9, 10e-9, 10e-9, 400e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            self.maxpar  = np.array([  30e-9, 30e-9, 30e-9, 30e-9, 30e-9, 30e-9, 30e-9, 600e-3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1e-13])
            self.stppar  = np.array([  2e-9, 2e-9, 2e-9, 2e-9, 2e-9, 2e-9, 2e-9, 20e-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1e-13])
            
            self.par_line_number = 7  
            self.parname =          [ 'lbias','lbp','lbn','lin1','lin2','ltn','ltp','vcmo','mamp','fbias','fbp','fbn','fin1','fin2','ftn1','ftn2','ftp1','ftp2','cload']				
            self.metricname = ['cin', 'cout', 'gain', 'gm', 'pole1', 'pole2', 'rout', 'cmo', 'pwr', 'swing14', 'swing7', 'swingn', 'swingn1', 'swingn4', 'swingp', 'irn'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/foldedcascode_cmfb_PTM14nm.csv'                
    
    def make_metrics(self):
        z = [10,12,14,15,17,18,19,24,25,26,27,28,29,30,31,48]
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


class ClassAB_spice(Netlists):
    
    def __init__(self, tech = 45, testfolder =None):
        if tech ==45:
            self.testbench = home_address + '/Netlists/classab_PTM45nm.scs'
            self.testfolder = home_address + '/Garbage/classAB_45' if testfolder ==None else home_address + '/Garbage/' + testfolder

            self.minpar  = np.array([  1, 45e-9, 1, 1, 45e-9, 45e-9, 400e-3, 1, 1e-13])
            self.maxpar  = np.array([  100, 900e-9, 100, 100, 600e-9, 600e-9, 600e-3, 100, 1e-13])
            self.stppar  = np.array([  1, 20e-9, 1, 1, 10e-9, 10e-9, 20e-3, 1, 1e-13])
            
            self.par_line_number = 7  
            self.parname =          [ 'fbias','lbias','fin','fp','lin','lp','vcmo','mamp','cload']
            self.metricname = ['cin','cout','gain', 'gm','pole1','rout','zero','cmo','pwr','swingn','swingp'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/classAB_45.csv'

        if tech ==32:
            self.testbench = home_address + '/Netlists/classab_PTM32nm.scs'
            self.testfolder = home_address + '/Garbage/classAB_32' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
            self.minpar  = np.array([  1, 32e-9, 1, 1, 32e-9, 32e-9, 400e-3, 1, 1e-13])
            self.maxpar  = np.array([  100, 640e-9, 100, 100, 480e-9, 480e-9, 600e-3, 100, 1e-13])
            self.stppar  = np.array([  1, 20e-9, 1, 1, 10e-9, 10e-9, 20e-3, 1, 1e-13])
            
            self.par_line_number = 7  
            self.parname =          [ 'fbias','lbias','fin','fp','lin','lp','vcmo','mamp','cload']
            self.metricname = ['cin','cout','gain', 'gm','pole1','rout','zero','cmo','pwr','swingn','swingp'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/classAB_32.csv'    
            
        if tech ==14:
            self.testbench = home_address + '/Netlists/classab_PTM14nm.scs'
            self.testfolder = home_address + '/Garbage/classAB_14' if testfolder ==None else home_address + '/Garbage/' + testfolder
            
            self.minpar  = np.array([  1, 10e-9, 1, 1, 10e-9, 10e-9, 400e-3, 1, 1e-13])
            self.maxpar  = np.array([  100, 30e-9, 100, 100, 30e-9, 30e-9, 600e-3, 100, 1e-13])
            self.stppar  = np.array([  1, 2e-9, 1, 1, 2e-9, 2e-9, 20e-3, 1, 1e-13])
            
            self.par_line_number = 7  
            self.parname =          [ 'fbias','lbias','fin','fp','lin','lp','vcmo','mamp','cload']
            self.metricname = ['cin','cout','gain', 'gm','pole1','rout','zero','cmo','pwr','swingn','swingp'] 
            self.make_metrics()
                    
            self.finaldataset = home_address + '/Datasets/classAB_14.csv'                
    
    def make_metrics(self):
        z = [10,12,14,15,17,18,19,24,25,26,27]
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
    

    fc = Folded_Cascode_spice(45)
    fc.put_on_csv(tedad=10,outcsv = home_address + '/Datasets/foldedcascode_cmfb_PTM45_v2.csv',do_header = True)
    cab = ClassAB_spice(45)
    cab.put_on_csv(tedad=10,outcsv = home_address + '/Datasets/classAB_45_v2.csv',do_header = True)
    #p,w = myseqp2.wholerun_random()
#    myseqp2.put_on_csv(tedad=200,outcsv = myseqp2.finaldataset,do_header = False)
