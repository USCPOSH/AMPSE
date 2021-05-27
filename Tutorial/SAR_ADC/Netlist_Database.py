


import numpy as np
import pandas as pd
import os
import math
from spectreIOlib import TestSpice, Netlists
import time
home_address  = os.getcwd()


    
class Compp_spice3(Netlists):
    
    def __init__(self, tech = 65, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)
        if tech ==65:
            self.testbench = home_address + '/Netlists/TSMC65nm/complatch_v1_65_TT.scs'
            self.testfolder = home_address + '/Garbage/Comp65_1_3' if testfolder ==None else home_address + '/Garbage/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPin6501_TT.csv'
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/complatch_v3_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/CompPTM65_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPinPTM6503.csv' 
        
        elif tech ==14:
            self.testbench = home_address + '/Netlists/GF14nm/complatch_v1_14_TT.scs'
            self.testfolder = home_address + '/Garbage/Comp14nm_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPin14nm01_TT.csv' 
            
        # self.minpar  = np.array([1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ,  4    ,   1     ,    1      ,100e-6, 0.4 , 1.0 ])
        # self.maxpar  = np.array([12   , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   10  ,   48 ,  12  ,  16   ,  12   ,   1     ,    1      ,100e-6, 0.4 , 1.0 ])
        # self.stppar  = np.array([1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ,  1    ,   1     ,    1      , 10e-6, 0.2 , 0.01])
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 7            
        self.minpar  = np.array([1     , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ])
        self.maxpar  = np.array([12    , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   10  ,   48 ,  12  ,  16   ])
        self.stppar  = np.array([1     , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ])
        self.parname =          ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
        self.metricname =       ['power','readyp','delayr','vomin','kickn','cin','scin','irn']
        self.make_metrics()    


    def change_testfolder(self, testfolder=None):
        self.testfolder = self.testfolder if testfolder ==None else home_address + '/Garbage/' + testfolder
        self.make_metrics()
        pass
        
        
    def make_metrics(self):
        x=np.array([9,10,11,12,15,17,21])
        z=[]
        for i in range(7):
            y = i*13+x;
            z = z + list(y)
        
        self.metricpositions =z
        self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass   
            
    def normal_run(self,param,lst_alter=[],parallelid=None):
        
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)

        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
        
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':5,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1
    
    
    def analysis(self,lst_out,length=0):
        # 0, 11,22,33,44,55,66 power
        
        out1=[]
        for i in range(length+1):
            x      = np.array(lst_out[i])
            power  =-x[0]
            ready  = x[1]
            delayr = x[2]
            vomin  = x[3]
            kickn  = np.std  (x[4::7])
            cin    = x[5]
            scin   = np.std  (x[5::7])
            irn    = 2*x[-1]
            if length>0:
                out1.append([power,ready,delayr,vomin,kickn,cin,scin,irn])
            else:
                out1 = [power,ready,delayr,vomin,kickn,cin,scin,irn]
        return out1
    
    
    def wholerun_normal(self,param,parallelid=None,lst_alter=[]):
        x = self.normal_run(param,lst_alter)
        w = self.analysis(x,len(lst_alter))       
        return w
    
    
   


class DACTH2_spice(Netlists):
    
    def __init__(self, tech = 65, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)
        if tech ==65:
            self.testbench = home_address + '/Netlists/dacth_v2_65_TT.scs'
            self.testfolder = home_address + '/Garbage/dacth65_2_1' if testfolder ==None else home_address + '/Garbage/' + testfolder     
            self.minpar  = np.array([2    ,  4    ,0.5e-15, 2     , 2     , 1     , 1     , 2.0e-15 ])
            self.maxpar  = np.array([16   ,  12   ,5.0e-15, 40    , 60    , 1     , 1     ,  30e-15 ])
            self.stppar  = np.array([2    ,  1    ,0.1e-15, 2     , 2     , 1     , 1     , 0.5e-15 ])
           

        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/dacth_v2_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/dacthPTM65_2_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([2    ,  4    ,0.5e-15, 2     , 2     , 1     , 1     , 2.0e-15 ])
            self.maxpar  = np.array([16   ,  12   ,5.0e-15, 40    , 60    , 1     , 1     ,  30e-15 ])
            self.stppar  = np.array([2    ,  1    ,0.1e-15, 1     , 1     , 1     , 1     , 0.5e-15 ])
             
            
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 7   
        self.parname =          [ 'div','mdac',   'cs', 'fthn', 'fthp','frefp','frefn',   'cp']            
        self.metricname = ['bw1','bw2','msb','dlydac']
        self.make_metrics()
        self.finaldataset = home_address + '/Datasets/PY_DACTHPTM6501.csv'  
    
    def make_metrics(self):
        z = [9,10,15,16]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        bw1 = z[2]
        bw2 = z[3]
        msb = 1 - z[0]
        delaydac= z[1]

        out1 = [bw1,bw2, msb, delaydac]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):
        p = param
        p[1] = 2**param[1]
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':p}
        
        
        
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)
        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w  
    


class Seqpart1_spice(Netlists):
    
    def __init__(self, tech = 65, testfolder =None, paralleling=False,max_parallel_sim=4, memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)

        if tech ==65:
            self.testbench = home_address + '/Netlists/sequential_v2_part1_65_TT.scs'
            self.testfolder = home_address + '/Garbage/sequential65p1_1_2' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        3 ])
            self.maxpar  = np.array([         12,        24,        96,         10,       16,        24,      16,    16,     16,       1,       1,   16,        11])
            self.stppar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        1 ])
            self.finaldataset = home_address + '/Datasets/PY_Seqp1_6501_TT.csv'
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/sequential_v2_part1_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/sequentialPTM65p1_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        3 ])
            self.maxpar  = np.array([         12,        24,        96,         10,       16,        32,      16,    16,     16,       1,       1,   16,        11])
            self.stppar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        1 ])
            self.finaldataset = home_address + '/Datasets/PY_Seqp1_PTM6501.csv'  
            
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 7   
        self.parname =          [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','div','mdacbig']            
        self.metricname = ['pwrdac','pwrdff','dlydac', 'dlydff']
        self.make_metrics()
        
    
    def make_metrics(self):
        z = [9,10,11,12]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass
        
        
        
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
        pwrdac  = -z[0]/2
        pwrdff  = -z[1]/2
        dlydac  = z[2]
        dlydff  = z[3]

        out1 = [pwrdac, pwrdff, dlydac, dlydff]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):
        p = param
        p[-1] = 2**param[-1]
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':p}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)        
        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w  

class Seqpart2_spice(Netlists):
    
    def __init__(self, tech = 65, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)

        if tech ==65:
            self.testbench = home_address + '/Netlists/sequential_v1_part2_65_TT.scs'
            self.testfolder = home_address + '/Garbage/sequential65p2_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([      1.,     1.])
            self.maxpar  = np.array([     10.,    12.])
            self.stppar  = np.array([      1.,     1.])
                       
            self.finaldataset = home_address + '/Datasets/PY_Seqp2_6501_TT.csv'
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/sequential_v1_part2_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/sequentialPTM65p2_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([      1.,     1.])
            self.maxpar  = np.array([     10.,    12.])
            self.stppar  = np.array([      1.,     1.])                     
            self.finaldataset = home_address + '/Datasets/PY_Seqp2_PTM6501.csv' 
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 7   
        self.parname =          [ 'nor3','fck1']            
        self.metricname = ['pwrnor','pwrbuf','pwrcmp', 'dlynor','dlybuf','dlycmp']
        self.make_metrics() 
    
    def make_metrics(self):
        z = [9,10,11,12,13,14]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='afs',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
        pwrnor  = -z[0]/5
        pwrbuf  = -z[1]/5
        pwrcmp  = -z[2]/5
        dlynor  =  z[3]
        dlybuf  =  z[4]
        dlycmp  =  z[5]
        
        out1 = [pwrnor, pwrbuf, pwrcmp, dlynor, dlybuf, dlycmp]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):

        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)        
        
        
        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w  


if __name__ == '__main__':
    
    # mydacth1 = DACTH2_spice(tech=65)
    # p,w = mydacth1.wholerun_random()
    
    # mycomp3 = Compp_spice3(tech=14,paralleling=True,max_parallel_sim=200)
    # w3 = mycomp1.wholerun_random()
    
    mycomp1 = Compp_spice3(tech=65)
    mycomp1.put_on_csv(tedad=100,outcsv = mycomp1.finaldataset,do_header=False)
    
    
    # p = np.array([11,	18,	32,	20,	29,	24,	8,	7,	1,	56, 2, 12, 4, 8])
    # w3 = mycomp3.wholerun_normal(param=p)
    # w2 = mycomp3.wholerun_std(param=p)
    # mycomp3 = Compp_spice3(tech=65)
    # w3 = mycomp3.wholerun_random()
    # myseq1 = Seqpart1_spice(tech=65)
    # p,w = myseq1.wholerun_random()    

    # myseq2 = Seqpart2_spice(tech=65)
    # p,w = myseq2.wholerun_random()  
    
#    mydacth1.put_on_csv(tedad=10,outcsv = mydacth1.finaldataset,do_header = True)
    
#    myseqp1 = Compp_spice2(testfolder = 'Comp65_1_3')
#    p,w = myseqp1.wholerun_random()
#    myseqp1.put_on_csv(tedad=10,outcsv = myseqp1.finaldataset,do_header = True)
#    p=[]
#    myseqp2 = Seqpart2_spice()
#    for i  in range(500):
#        p.append(myseqp2.random_param())
##        
#    npp=np.array(p)
#    
#    xmin = myseqp2.minpar
#    xmax = myseqp2.maxpar
#    xstp = myseqp2.stppar
#    w=[]
#    for i  in range(500):
#        w.append(randomchoice(xmax,xmin,xstp))
#        
#    npw=np.array(w)
    
    
#    p,w = myseqp2.wholerun_random()
#    myseqp2.put_on_csv(tedad=500,outcsv = myseqp2.finaldataset,do_header = True)
    