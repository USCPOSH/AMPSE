
# Modification 3: Added the class TestSpectre
# Designed By Mohsen Hassanpourghadi

# Modification Date: Apr  5, 2021,  v1.0.5
# Certain codes to run on other servers are added
# Modification Date: Oct 15, 2020,  v1.0.4
# We add the MDL control for spectre here
# Modification Date: May 15, 2020,  v1.0.3
# Now we add memory, checks if the results previously exists whole_run_STD
# We can calculate gradient now in the Netlists
# Modification Date: May 15, 2020,  v1.0.2
# Now inside the temp folder several parallel simulation are running
# addition of the self.counter to TestSpice
# Modification Date: Sep 25, 2019,  v1.0.1
# 




import warnings
import numpy as np
from os.path import exists
import os
import time
import pandas as pd
import multiprocessing as mp
import shutil


class TestSpice():
    # Generates a file test.scs inside the folder trash, and keeps the initial file unchanged
    # 3 dictionaries are presented:
    # dict_folder : 
    #               testbench: Folder where the testbench is there
    #               trashfolder Folder where you want the testbench be there
    # dict_parameters : 
    #               line_number: the parameter line in the test-bench that must be replaced
    #               name_params: a list of all parameter names
    #               value_params: a list of parameters values
    # lst_metrics : 
    #               Is a list of dictionaries, each dictionary includes:
    #               read: column or row
    #               filename: the file name for the measured results
    #               measurerange: a list of column or row values that we want to read
    # paralleling :
    #               Is a Boolean
    #               Set it true if you want paralleling, the test folder will have 100 testbenches
    # parallelid  : 
    #               Is an integer
    #               For paralleling adds as testid number 
    # maxparallelsim:
    #               Is an integer number.
    #               Only defines the maximum possible parallel simulation in a test folder
    
    def __init__(self,simulator,dict_folder, paralleling= False,maxparallelsim = 16,verbose = True):
        
        
        self.version = '1.0.4'
        self.verbose = verbose
        self.mdl     = False
        # Setting the simulator as afs or spectre
        if simulator.lower() == 'afs':
            self.simulator = 'afs'
        elif simulator.lower() == 'aps':
            self.simulator = 'aps'
        elif simulator.lower() == 'apsplus':
            self.simulator = 'apsplus'
        elif simulator.lower() == 'aps+mdl':
            self.mdl = True
            self.simulator = 'aps+mdl'
        elif simulator.lower() == 'apsplus+mdl':
            self.mdl = True
            self.simulator = 'apsplus+mdl'
        elif simulator.lower() == 'spectre+mdl':
            self.mdl = True
            self.simulator = 'spectre+mdl'
        else:
            self.simulator = 'spectre'
        
        self.exception = False
        
        # Setting the test-bench
        try:
            self.testbench = dict_folder['testbench']
            self.trash_folder = dict_folder['trashfolder']
            if self.mdl: self.mdlbench = dict_folder['mdl']
            if not exists(self.testbench):
                self.exception = True
                warnings.warn('The testbench is unreachable')
            if not exists(self.trash_folder):
                self.exception = True
                warnings.warn('The trashfolder is unreachable')
            
        except:
            warnings.warn("dict_folder has no attribute 'testbench' or 'trashfolder' please set these two value: dict_folder = {'testbench':'input.scs','trashfolder':'/trash'}" )
            self.testbench = 'input.scs'
            self.trash_folder = '/trash'
            if not exists(self.testbench):
                self.exception = True
                warnings.warn('The testbench is unreachable')
            if not exists(self.trash_folder):
                self.exception = True
                warnings.warn('The trashfolder is unreachable')
            
        self.verbose = verbose
        self.counter = 0
        self.maxps   = maxparallelsim 
        self.parallel = paralleling
        
        
        
    
    def __version__(self):
        print(self.version)
        pass            
    
    def runspectre(self,dict_parameters,alter_list=[],parallelid=None):
        
        # Setting the parameters
        
        self.counter+=1
        if self.counter>self.maxps:
            self.counter=1
        
        self.exception = False    
        try:
            self.line_number = dict_parameters['line_number']
            self.name_params = dict_parameters['name_params']
            self.value_params= dict_parameters['value_params']
        except Exception:
            print("Error : dict_parameters is not set with desired keys.")
            print("Please set it as dict_parameters={'line_numebr':int_number,'name_params':list of parameters names, 'value_params': list of parameters value}.")
            if self.verbose:
                print("Ex.: dict_parameters={'line_numebr':7,'name_params':['w','l'], 'value_params': [500e-9,100e-9]}")
            self.exception = True
            return self.exception
        if not len(self.name_params)==len(self.value_params):
            print('Error: name_params and value_params are not the same size')
            self.exception = True
            return self.exception
            

        
        # Make the parameters line
        sline = 'parameters  '
        for i in range(len(self.name_params)):
            if i==10:
                sline +=' \ \n'
            sline +=" " + str(self.name_params[i]) + " = " +str(self.value_params[i])
        sline +=' \n'
        
        
        
        
        # Open the testbench
        f = open(self.testbench, 'r')
        lines = f.readlines()
        f.close

        # Replace the parameter line number
        lines[self.line_number] = sline
        
        # .ALTER case
        if len(alter_list)>0:
            lines.append( 'simulator lang = spice \n')
            j = 1
            for items in alter_list:
                j+=1    
                lines.append('.ALTER case ' + str(j) + ' \n')
                if makeparamline(items) == ' ':
                    continue
                else:
                    lines.append('.PARAM' + makeparamline(items) + '\n')
        
        
        
        
        # Assiging  the test-bench ID:
        
        # print(parallelid)
        if self.parallel:
            np.random.seed()
            if type(parallelid)==int:
                testid = parallelid % self.maxps
                # print('The id is: ',testid )
            else:
        
                testid = int(time.time()*10000 % self.maxps)
                # print('Bye!')
        else:
            testid=0
                
        testname = 'test'+str(testid)+'.scs'
        
        
        # Save the test.scs
        f = open(self.trash_folder + '/'+testname,'w' )
        f.writelines(lines)
        f.close()
        
        # If mdl based testbench is chosen:
        mdlname = 'test'+str(testid)+'.mdl'
        if self.mdl:
            shutil.copy(self.mdlbench,self.trash_folder + '/'+mdlname )
            
        
        
        
        # making the command line
        if self.simulator == 'spectre':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+'  =log spectre.log'
        elif self.simulator == 'aps':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' +aps =log aps.log'
        elif self.simulator == 'apsplus':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' ++aps =log apsplus.log'
        elif self.simulator == 'spectre+mdl':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+'  =log spectre.log'+ ';spectremdl -batch '+mdlname+'  =log mdl.log'
        elif self.simulator == 'aps+mdl':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' +aps =log aps.log'+ ';spectremdl -batch '+mdlname+'  =log mdl.log'
        elif self.simulator == 'apsplus+mdl':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' ++aps =log apsplus.log'+ ';spectremdl -batch '+mdlname+'  =log mdl.log'
        elif self.simulator == 'afs':
            commandline = 'cd ' + self.trash_folder + '; afs     '+testname+' -f psfbin --nolog >& afs.log'   
        elif self.simulator == 'spectre_ascii':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+'  =log spectre.log -f psfascii'
        elif self.simulator == 'aps_ascii':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' +aps =log aps.log -f psfascii'
        elif self.simulator == 'apsplus_ascii':
            commandline = 'cd ' + self.trash_folder + '; spectre '+testname+' ++aps =log apsplus.log -f psfascii'
        elif self.simulator == 'afs_ascii':
            commandline = 'cd ' + self.trash_folder + '; afs     '+testname+' -f psfascii --nolog >& afs.log' 
        if self.verbose:
            print(" Running the code: " + commandline)
        
        # Running the command line
        os.system(commandline)
        
        
        return testid
    
    def readmetrics(self,lst_metrics):
        # Setting the metrics
        self.exception = False
        out=[]
        try:
            self.sizemetrics = len(lst_metrics)
            self.lst_metrics = lst_metrics
            for item in self.lst_metrics:
                x = item['read']

                if x == 'row':
                    out.append(rowread(item['filename'],item['number'],item['measurerange']))
                else:
                    out.append(columnread(item['filename'],item['measurerange'],item['number']))
                                        
                
        except:
            print("Error : lst_metrics is not set with desired keys.")
            print("Please set it as lst_metrics=[{'read':'column' or 'row','filename':THE OUTFILE FILE,'number': the column or row number,'measurerange':list of numbers that should be read}.")
            if self.verbose:
                print("Ex.: lst_metrics=[{'read':'column','filename':'test.measure','number': 2,'measurerange':[7,12,14,15,16]")
            self.exception = True
            return out
        
        return out
    

def makeparamline(dict_param):
    # generates a line format for the parameters:
    # dict_param is a dictionary like : {'wn':500e-9,'ln':60e-9}
    # The output string is wn = 500e-9, ln = 69e-9
    sline = ' '
    if type(dict_param)==dict:
        thekeys = dict_param.keys()
        for key in thekeys:
            value = dict_param[key]
            sline +=str(key)+' = ' + str(value)+ ' '
    else:
        print('The input is not dict type')
    return sline
        
        
def randomchoice(xmax,xmin,xstep):
    # generates data between xmax and xmin with step eqaul to xstep
    # all three should be numpy arrays
    # no values in xstep should be 0
    try:
        xamp = (xmax-xmin)
        c = np.random.random(np.shape(xamp))*(xamp+xstep)+xmin
        xout = np.floor((c-xmin)/xstep)*xstep+xmin
        
        
        
        xout[xamp==0]=xmax[xamp==0]
        
    except:
        print('Error: Something went wrong. Check the input shapes, xstep cannot have zero element!')
        xout = np.array([0])
    return xout    
        
        


class Netlists():
    
    def __init__(self, paralleling=False,max_parallel_sim=4, memory_capacity=10):
        self.memory_capacity = memory_capacity
        self.mps     = max_parallel_sim
        self.prl     = paralleling    
        self.lst_memory = []
    
    def paramset(self,xmin,xmax,xstp):
        self.minpar = xmin
        self.maxpar = xmax
        self.stppar = xstp
        pass
    
    @property
    def psize(self):
        return len(self.parname)
    @property
    def msize(self):
        return len(self.metricname)
    
    def scale_p(self,p):
        x = 2*(p-self.minpar)/(self.maxpar-self.minpar)-1
        return x
    
    def iscale_x(self,x):
        p = (x+1)/2*(self.maxpar-self.minpar)+self.minpar
        return p
    
    def range_analysis(self):
        ranges = (self.maxpar- self.minpar)/self.stppar
        return ranges
    def return_domain(self):
        
        return [self.minpar,self.maxpar,self.stppar]
        
    
    def random_param(self):
        
        return randomchoice(self.maxpar,self.minpar,self.stppar)
    
    def random_run(self):
        param = randomchoice(self.maxpar,self.minpar,self.stppar)
        return param, self.normal_run(param)
    
    def wholerun_random(self,prlid=None):
        p = randomchoice(self.maxpar,self.minpar,self.stppar)
        w = self.wholerun_normal(p,parallelid=prlid)
        return p, w
    
    def param_std(self,param):
        param[param<self.minpar]=self.minpar[param<self.minpar]
        param[param>self.maxpar]=self.maxpar[param>self.maxpar]        
        sparam = np.round(param/self.stppar)*self.stppar        
        return sparam
    def standard_run(self,param,prlid=None):        
        sparam = self.param_std(param)
        w = self.wholerun_normal(sparam,parallelid=prlid)
        return sparam, w
    
    def wholerun_std(self,param,isscaled = False, put_on_csv = False):
        if isscaled ==True:
            nparam = self.iscale_x(param)
            sparam = self.param_std(nparam)
        else:
            sparam = self.param_std(param)
        lst_w = self.check_memory(sparam)
        if len(lst_w)>0:
            w = lst_w
        else:
            zparam = np.copy(sparam)
            w = self.wholerun_normal(sparam)
            self.memory_add(zparam,w)
        
            if put_on_csv:
                with open(self.finaldataset,'a') as f:
                    df_pred = pd.DataFrame(list(sparam)+list(w)).transpose()
                    df_pred.to_csv(f, header=False, index=False)
            
        return sparam, w
    
    def put_on_csv(self,tedad=1,outcsv='out.csv',do_header=True,nparallel=0):
        tstart=time.time()
        
        if nparallel>mp.cpu_count():
            print('Maximum number of cpu is used! Please decrease nparallel')
        
        if nparallel>0:
            pool = mp.Pool(nparallel)
            with open(outcsv, 'a') as f:
                if do_header:
                    header = self.parname + self.metricname
                    df_pred = pd.DataFrame(header).transpose()
                    df_pred.to_csv(f, header=False, index=False)
                for j in range(0,tedad,nparallel):
                    # try:
                        
                    lst_out=[]
                    results = [ pool.apply(self.wholerun_random,args=(prlid,)) for prlid in range(nparallel)]
                    for items in results:
                        lst_out.append(list(items[0])+list(items[1]))
                    # print(lst_out)
                    df_pred = pd.DataFrame(lst_out)
                    df_pred.to_csv(f, header=False, index=False)
                    ts= time.time() - tstart
                    print('Simulation number %1.0f was succcesful!:  %d s passed' %(j+1,ts))
                    # except:
                    #     ts= time.time() - tstart
                    #     print('Simulation number %1.0f was unsuccesful!:  %d s passed' %(j+1,ts))
            pool.close()
        else:
            # if no paralleling is needed!
            
            with open(outcsv, 'a') as f:
                if do_header:
                    header = self.parname + self.metricname
                    df_pred = pd.DataFrame(header).transpose()
                    df_pred.to_csv(f, header=False, index=False)
                for j in range(0,tedad):
                    try:
                        p,m = self.wholerun_random()
                        df_pred = pd.DataFrame(list(p)+list(m)).transpose()
                        df_pred.to_csv(f, header=False, index=False)
                        ts= time.time() - tstart
                        print('Simulation number %1.0f was succcesful!:  %d s passed' %(j+1,ts))
                    except:
                        ts= time.time() - tstart
                        print('Simulation number %1.0f was unsuccesful!:  %d s passed' %(j+1,ts))
        pass
    
    
    
    def run_from_csv(self,incsv,outcsv,do_header=True):
        tstart=time.time()
        data = pd.read_csv(incsv,header=None)
        length = len(self.parname)
        params = np.array(data.iloc[:, 0:length].values,dtype='float64')
        i=0
        with open(outcsv, 'a') as f:
            for param in params:
                i+=1
                try:
                    
                    m = self.wholerun_normal(param)
                    df_pred = pd.DataFrame(list(param)+list(m)).transpose()
                    df_pred.to_csv(f, header=False, index=False)
                    ts= time.time() - tstart
                    print('Simulation number %1.0f was succcesful!:  %d s passed' %(i,ts))
                except:
                    print('Simulation number %1.0f was unsuccesful!:  %d s passed' %(i,ts))
        pass
        
    def exhaustive_gradient(self,p0,chosen_variables=[],m0=0):
        
        tstart=time.time()
        lst_out=[]
        len_p = len(p0)
        if len(chosen_variables)<len_p:
            chosen_variables = np.ones(len_p)
        
        for i in range(len_p):
            if chosen_variables[i]==1:
                p1 = p0
                p1[i] += self.stppar[i]
                m1 = np.array(self.wholerun_normal(p1))-m0
                lst_out.append(m1)

        ts= time.time() - tstart
        print('Simulation was succcesful!:  %d s passed' %(ts))
        
        np_out = np.array(lst_out)
        
        return np_out
    
    
    def alter_gradients(self,p0,chosen_variables):
        lst_var=[]
        
        for i in range(chosen_variables):
            if chosen_variables[i]>0:
                name = self.parname [i]
                newvalue = p0[i] + self.stppar[i]
                lst_var.append({name:newvalue})
        j = len(lst_var)
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':p0}
        self.alter_metrics(j)
        lst_out = self.wholerun_normal(p0,j)
        np_out = np.array(lst_out)
        for i in range(j):
            np_out[i,:] += -np_out[0,:]
        
        return np_out[1:,:]

    def memory_add(self,p,m):
        lst_p = list(p)
        lst_m = list(m)
        self.lst_memory.append((lst_p,lst_m))
        if len(self.lst_memory)>self.memory_capacity:
            self.lst_memory.pop(0)
    
    def check_memory(self,p):
        lst_p = list(p)
        for i, par_met in enumerate(self.lst_memory):
            if lst_p == par_met[0]:
                lst_m = par_met[1]
                return lst_m
        return []
            
                
        

def testspectre(filename,abspath,lineparam,paramnames,paramvalues):
# Generates the file test.scs inside the folder Trash.
# filename is the template *.scs file for running in specte. Run file by spectre *.scs to make sure you have the write output
# abspath is the current folder of the input.scs and the Trash folder
# lineparam is the line number in the *.scs file that has all the parameters
# parameters is the list that has both parameters name and their values
	# 
	lenparams=len(paramnames)
	sline="parameters  "
	for i in range(0,lenparams):
		if i==10:
			sline=sline+' \ \n'
		sline=sline+ " " + str(paramnames[i])+ " = " + str(paramvalues[i])
	sline=sline+'\n'
	f = open(abspath+filename, 'r')    # pass an appropriate path of the required file
	lines = f.readlines()
	lines[lineparam] = sline    
	f.close()  
	f = open(abspath+'Trash/test.scs', 'w')
	f.writelines(lines)
	f.close()
	warnings.warn('depricated',DeprecationWarning)
    
def testspectrelibdev(filename,abspath,lineparam,paramnames,paramvalues, linelibrary, library_address, linedevice, device_addincode):
# Generates the file test.scs inside the folder Trash.
# filename is the template *.scs file for running in specte. Run file by spectre *.scs to make sure you have the write output
# abspath is the current folder of the input.scs and the Trash folder    
# lineparam is the line number in the *.scs file that has all the parameters
# parameters is the list that has both parameters name and their values
# linelibrary is the line number in the *.scs file that we want to add the new library
# library_address is the new library address
# linedevice is the line number in the *scs file that we want to add the new device
# device_addincode is the adding device spectre code defining the device name and its node connection
	# 
    lenparams=len(paramnames)
    sline="parameters  "
    for i in range(0,lenparams):
        if i==10:
            sline=sline+' \ \n'
        sline=sline+ " " + str(paramnames[i])+ " = " + str(paramvalues[i])

    sline=sline+'\n'
	
    lline=library_address
    #lline='include' +'"'+library_address+'" \n'
    f = open(abspath+filename, 'r')    # pass an appropriate path of the required file
    lines = f.readlines()
	
	
    lines[linedevice]=device_addincode
    lines[linelibrary]=lline
    lines[lineparam]=sline
    f.close()  
    f = open(abspath+'Trash/test.scs', 'w')
    f.writelines(lines)
    f.close()
    warnings.warn('depricated',DeprecationWarning)


########################### Run on remote server ################################
import socket
import sys

def list_to_byte(lst_in):
    
    
    lst_in = list(lst_in)
    
    str_in = str(lst_in)
    byte_in = str_in.encode('utf-8')
    
    return byte_in

def byte_to_list(byte_in, type_in = None):
    
    str_in = byte_in.decode()
    str_in = str_in.replace('[','')
    str_in = str_in.replace(']','')
    lst_o = str_in.split(',')
    
    if type_in ==None or type_in=='float':
        return list(map(float,lst_o))
    elif type_in=='int':
        return  list(map(int,lst_o))
    else:
        return lst_o
    
def get_command(port):
    for res in socket.getaddrinfo(None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except OSError as msg:
            s = None
            continue
        try:
            s.bind(sa)
            s.listen(1)
        except OSError as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print('could not open socket')
        sys.exit(1)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data1 = conn.recv(1024)
            data = conn.recv(1024)
            break
            
    return data.decode()


def run_command(port,dict_global,running = False):
    for res in socket.getaddrinfo(None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except OSError as msg:
            s = None
            continue
        try:
            s.bind(sa)
            s.listen(1)
        except OSError as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print('could not open socket')
        sys.exit(1)
    conn, addr = s.accept()
    try:
        with conn:
            print('Connected by', addr)
            while True:
                data1 = conn.recv(1024)
                print(data1)
                num1 = list(data1)[0]
                conn.send(b'1',1024)            
                
                
                data2 = conn.recv(1024)
                
                command = data2.decode()
                if command =='end.':
                    command ='end.'
                    conn.send(b'0',1024)
                    break
                conn.send(b'1',1024)
                exec(command,dict_global)
                # lst_vars=[]
                
                for i in range(num1):
                    data3 = conn.recv(1024)
                    lst_out = eval(data3.decode(),dict_global)
                    if type(lst_out)==int or type(lst_out)==float or type(lst_out)==str:
                        lst_out =[lst_out]
                    
                    byte_out = list_to_byte(lst_out)
                    conn.send(byte_out)
                        
                
                
                print('Results sent back to', addr)
                if running==False:
                    break
    
    except Exception as msg:
        conn.close()
        print(msg)
        s.close()
        s = None
      
        
    return command



def prepare_socket(server, port):
    s = None
    for res in socket.getaddrinfo(server, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except OSError as msg:
            s = None
            continue
        try:
            s.connect(sa)
        except OSError as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print('could not open socket')
        sys.exit(1)
    return s


def send_command(command, s, dict_global, variables=[]):

    lenv = len(variables)
    try:
        s.send(bytes([lenv]))
        r1 = s.recv(1024)
        
        
        s.send(command.encode('utf-8'))
        r2 = s.recv(1024)
        if r2==b'0':
            print('Termination')
            s.close()
            s=None
            return ''
        print('Command sent succesfully!')
        
        new_command=''
        for variable in variables:
            s.send(variable.encode('utf-8'))
            v = s.recv(1024)
            lst_out = byte_to_list(v)
            new_command = new_command+str(variable)+'='+str(lst_out)+';\n'
        exec(new_command,dict_global)
        print('Command executed succesfully!')
    except Exception as msg:
        print(msg)
        s.close()
        s=None

    return new_command





def randominput(minparam,maxparam,intrange):
#Generates a random number between minparam to maxparam
#minparam is the minimum values selected for all the parameters
#maxparam is the maximum values selected for all the parameters
#seedstart is the random seed you can choose any numer
#intrange specifys the range that the output must be random integer. define it by range(n1,n2-1)
	randvalues=np.random.rand(1,len(minparam))
	x=np.add(np.multiply(np.subtract(maxparam,minparam),randvalues),minparam).T
	x[intrange]=np.floor(x[intrange])
	parvalue=np.reshape(x,len(x))	
	return parvalue.tolist()

def columnread(filename,linerange,colnumber):
# Takes the file to read one column of numbers 
# filename is the file name
# linerange is the line number range that contains the required information
# colnumber is the column number in the specified lines
# returns the list of float numbers from that column
    f = open(filename, 'r')
    lines = f.readlines()
    f.close() 
    results=[];
    for i in linerange:
        try:
            x=lines[i].split()
            results.append(float(x[colnumber]))
        except:
            continue        
    return results

def rowread(filename,linenum,colrange):
# Takes the file to read one row of numbers 
# filename is the file name
# linenum is the line number that contains the required information
# colrange is the column number range that has required information
# returns the list of float numbers from that row
	f = open(filename, 'r')
	lines = f.readlines()
	f.close() 
	results=[];
	x=lines[linenum].split()
	results=x[colrange]
	return results


def randominput_int(minparam,maxparam,even,odd,same):
#Generates a random number between minparam to maxparam
#minparam is the minimum values selected for all the parameters
#maxparam is the maximum values selected for all the parameters
#seedstart is the random seed you can choose any numer
#even: the range that should be even output
#odd: the range that should be odd output
#same: the range that should be equal in even and odd
	
	x=np.array([])
	for i in range(0,len(minparam)):
		x=np.append(x,np.random.randint(minparam[i],maxparam[i]+1))
	for i in even:
		x[i]=(x[i]//2)*2
	for i in odd:
		x[i]=(x[i]//2)*2+1
	for i in same:
		print(i[0])
		if np.mod(x[i[0]],2)!=np.mod(x[i[1]],2):
			if np.random.randint(0,2)==0:
				x[i[0]]=x[i[0]]+1
			else:
				x[i[1]]=x[i[1]]+1

	parvalue=x		
	return parvalue.tolist()
