# Modification 3: Added the class TestSpectre
# Designed By Mohsen Hassanpourghadi
# Modification Date: Sep 25, 2019





import warnings
import numpy as np
from os.path import exists
import os
import time
import pandas as pd



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
    
    def __init__(self,simulator,dict_folder, verbose = True):
        
        
        self.version = '1.0.1'
        self.verbose = verbose
        # Setting the simulator as afs or spectre
        if simulator.lower() == 'afs':
            self.simulator = 'afs'
        elif simulator.lower() == 'aps':
            self.simulator = 'aps'
        elif simulator.lower() == 'apsplus':
            self.simulator = 'apsplus'
        else:
            self.simulator = 'spectre'
        self.exception = False
        
        # Setting the test-bench
        try:
            self.testbench = dict_folder['testbench']
            self.trash_folder = dict_folder['trashfolder']
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
        
        
    
    def __version__(self):
        print(self.version)
        pass            
    
    def runspectre(self,dict_parameters,alter_list=[]):
        
        # Setting the parameters
        
        self.exception = False    
        try:
            self.line_number = dict_parameters['line_number']
            self.name_params = dict_parameters['name_params']
            self.value_params= dict_parameters['value_params']
        except:
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
        
                
        
        # Save the test.scs
        f = open(self.trash_folder + '/test.scs','w' )
        f.writelines(lines)
        f.close()
        
        
        # making the command line
        if self.simulator == 'spectre':
            commandline = 'cd ' + self.trash_folder + '; spectre test.scs  =log spectre.log'
        elif self.simulator == 'aps':
            commandline = 'cd ' + self.trash_folder + '; spectre test.scs +aps =log aps.log'
        elif self.simulator == 'apsplus':
            commandline = 'cd ' + self.trash_folder + '; spectre test.scs ++aps =log apsplus.log'
        elif self.simulator == 'afs':
#            commandline = 'afs ' + self.trash_folder + '/test.scs -o ' + self.trash_folder + '/psf.out  -f psfascii --nolog >& afs.log'
            commandline = 'cd ' + self.trash_folder + '; afs test.scs -f psfbin --nolog >& afs.log'        
        if self.verbose:
            print(" Running the code: " + commandline)
        
        # Running the command line
        os.system(commandline)
        
        
        return self.exception
    
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
        c = np.random.random(np.shape(xamp))*xamp+xmin
        xout = (c//xstep)*xstep+xstep
        
        xout[xamp==0]=xmax[xamp==0]
        
    except:
        print('Error: Something went wrong. Check the input shapes, xstep cannot have zero element!')
        xout = np.array([0])
    return xout    
        
        


class Netlists():
    
    def paramset(self,xmin,xmax,xstp):
        self.minpar = xmin
        self.maxpar = xmax
        self.stppar = xstp
        pass

    def random_run(self):
        param = randomchoice(self.minpar,self.maxpar,self.stppar)
        return param, self.normal_run(param)
    
    def wholerun_random(self):
        p = randomchoice(self.minpar,self.maxpar,self.stppar)
        w = self.wholerun_normal(p)
        return p, w
    
    def param_std(self,param):
        param[param<self.minpar]=self.minpar[param<self.minpar]
        param[param>self.maxpar]=self.maxpar[param>self.maxpar]        
        sparam = np.round(param/self.stppar)*self.stppar        
        return sparam
    def standard_run(self,param):        
        sparam = self.param_std(param)
        w = self.wholerun_normal(sparam)
        return sparam, w
    
    def wholerun_std(self,param):
        sparam = self.param_std(param)
        w = self.wholerun_normal(sparam)
        return sparam, w
    
    def put_on_csv(self,tedad=1,outcsv='out.csv',do_header=True):
        tstart=time.time()
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
        length = len(self.paramname)
        params = np.array(data.iloc[:, 0:length].values,dtype='float64')
        with open(outcsv, 'a') as f:
            for param in params:
                try:
                    m = self.wholerun_normal(param)
                    df_pred = pd.DataFrame(list(param)+list(m)).transpose()
                    df_pred.to_csv(f, header=False, index=False)
                    ts= time.time() - tstart
                    print('Simulation was succcesful!:  %d s passed' %(ts))
                except:
                    print('Simulation was unsuccesful!:  %d s passed' %(ts))
        pass
        
    def exhaustive_gradient(self,p0,chosen_variables,m0=None):
        
        tstart=time.time()
        lst_out=[]
        for i in range(len(p0)):
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
