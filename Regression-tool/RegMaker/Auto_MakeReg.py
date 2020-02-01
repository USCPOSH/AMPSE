#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:28:50 2019

@author: mutian
"""

import os
import sys
import numpy as np
import pandas as pd
import math

sys.path.insert(0, '/home/mutian/python_codes/AMPSE2')
# import spectreIOlib as silib
# import analib as analib

import multiprocessing as mp

import time

# Importing the Keras libraries and packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------
# gobal set
#
# -------------------------------------------------------------------------------------
# parname=['cpar','vbias','vdd','wn','wp']

inputfile = "user.in"
tempscs = 'tbtemp'  # name of new scs file, but the file will delete at last
outcsv = 'out.csv'
# tedad = 10  # number of sampling and new scs will generate

# finputscs = sys.argv[2]


# keywords = ['parameters','tran tran','finalTimeOP info','element info','simulatorOptions options']
# don't change the first 'utianzh
# mutianzh
# parameters'
# add any line you want get from .scs which have the format same as the following example
# parameters cpar=2f vbias=500m vdd=1 wn=10u wp=10u

# keywords = ['parameters','tran tran','finalTimeOP info']
keywords = ['parameters']


# fpower='vco_data/spectre.dc'


block = input('Please choose the block level design (Options: SAR_ADC, XXX, XXX, etc): ')
tech = input('Please choose the technology (OPtions: 65, XXX, XXX, etc): ')

if block == "SAR_ADC":
    homeaddress = '/home/mutian/python_codes/AMPSE2/SAR_ADC'

    STOP = False
    while not STOP:

        name = input("Please choose one of the modules: Comparator, TrackandHold, Sequencer1 and Sequencer2 ")
        if name == "Comparator":
            finputscs = homeaddress + '/Netlists/complatch_v2_65_TT.scs'
            from SAR_ADC.Netlist_Database import Compp_spice2

            module = Compp_spice2(tech=int(tech))
            input_dim = 15
            output_dim = 8
            data_address = homeaddress + '/Datasets/PY_COMPPin6501_TT.csv'
            save_addr = homeaddress + '/Reg_files/PY_COMPPin6502_TT/'
            clear_command = 'rm' + ' ' + homeaddress + '/Datasets/PY_COMPPin6501_TT.csv'


            def preprocessing(dataset):
                X = np.array(dataset.iloc[1:, 0:15].values, dtype='float64')
                y = np.array(dataset.iloc[1:, 20:28].values, dtype='float64')
                remfilt = [not math.isnan(d) for d in y[:, -1]]
                X = X[remfilt]
                y = y[remfilt]
                remfilt = [not d < 0 for d in y[:, 1]]
                X = X[remfilt]
                y = y[remfilt]
                remfilt = [not d < 0 for d in y[:, 2]]
                X = X[remfilt]
                y = y[remfilt]

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

                from sklearn.preprocessing import StandardScaler
                from sklearn.preprocessing import MinMaxScaler

                sc_X = MinMaxScaler(feature_range=(-1, 1))
                sc_y = StandardScaler()

                sX_train = sc_X.fit_transform(X_train)
                sy_train = sc_y.fit_transform(y_train)
                sX_test = sc_X.transform(X_test)
                sy_test = sc_y.transform(y_test)

                sy_train[:, 3] = sy_train[:, 3] + 0.2

                return sX_train, sy_train, sX_test, sy_test, sc_X, sc_y


            STOP = True

        elif name == "TrackandHold":
            finputscs = homeaddress + '/Netlists/dacth_v2_65_TT.scs'
            from SAR_ADC.Netlist_Database import DACTH2_spice

            module = DACTH2_spice(tech=int(tech))
            STOP = True

            input_dim = 6
            output_dim = 4
            data_address = homeaddress + '/Datasets/PY_DACTH6502_TT.csv'
            save_addr = homeaddress + '/Reg_files/PY_THDAC6502_TT/'
            clear_command = 'rm' + ' ' + homeaddress + '/Datasets/PY_DACTH6502_TT.csv'


            def preprocessing(dataset):
                X = np.array(dataset.iloc[1:, [0, 1, 2, 3, 4, 7]].values, dtype='float64')
                y = np.array(dataset.iloc[1:, 8:12].values, dtype='float64')
                X[:, 1] = np.log2(X[:, 1])

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

                from sklearn.preprocessing import StandardScaler
                from sklearn.preprocessing import MinMaxScaler

                sc_X = MinMaxScaler(feature_range=(-1, 1))
                sc_y = StandardScaler()

                sX_train = sc_X.fit_transform(X_train)
                sy_train = sc_y.fit_transform(y_train)
                sX_test = sc_X.transform(X_test)
                sy_test = sc_y.transform(y_test)

                return sX_train, sy_train, sX_test, sy_test, sc_X, sc_y

        elif name == "Sequencer1":
            finputscs = homeaddress + '/Netlists/sequential_v1_part1_65_TT.scs'
            from SAR_ADC.Netlist_Database import Seqpart1_spice

            module = Seqpart1_spice(tech=int(tech))
            STOP = True

            input_dim = 13
            output_dim = 4
            data_address = homeaddress + '/Datasets/PY_Seqp1_6501_TT.csv'
            save_addr = homeaddress + '/Reg_files/PY_SEQ16501_TT/'
            clear_command = 'rm' + ' ' + homeaddress + '/Datasets/PY_Seqp1_6501_TT.csv'


            def preprocessing(dataset):
                X = np.array(dataset.iloc[1:, 0:13].values, dtype='float64')
                y = np.array(dataset.iloc[1:, 13:].values, dtype='float64')

                X[:, -1] = np.log2(X[:, -1])
                # y[:,1] = np.log10(y[:,1])
                # y[:,3] = np.log10(y[:,3])

                remfilt6 = [d < 5e-9 and d > -5e-9 for d in y[:, 2]]
                X = X[remfilt6]
                y = y[remfilt6]
                remfilt7 = [d < 1e-9 and d > -1e-9 for d in y[:, 3]]
                X = X[remfilt7]
                y = y[remfilt7]

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

                from sklearn.preprocessing import StandardScaler
                from sklearn.preprocessing import MinMaxScaler

                sc_X = MinMaxScaler(feature_range=(-1, 1))
                sc_y = StandardScaler()

                sX_train = sc_X.fit_transform(X_train)
                sy_train = sc_y.fit_transform(y_train)
                sX_test = sc_X.transform(X_test)
                sy_test = sc_y.transform(y_test)

                return sX_train, sy_train, sX_test, sy_test, sc_X, sc_y




        elif name == "Sequencer2":
            finputscs = homeaddress + '/Netlists/sequential_v1_part2_65_TT.scs'
            from SAR_ADC.Netlist_Database import Seqpart2_spice

            module = Seqpart2_spice(tech=int(tech))
            STOP = True

            input_dim = 2
            output_dim = 6
            data_address = homeaddress + '/Datasets/PY_Seqp2_6501_TT.csv'
            save_addr = homeaddress + '/Reg_files/PY_SEQ26501_TT/'
            clear_command = 'rm' + ' ' + homeaddress + '/Datasets/PY_Seqp2_6501_TT.csv'


            def preprocessing(dataset):

                dataset = dataset.dropna()
                X = np.array(dataset.iloc[1:, 0:2].values, dtype='float64')
                y = np.array(dataset.iloc[1:, 2:].values, dtype='float64')

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

                from sklearn.preprocessing import StandardScaler
                from sklearn.preprocessing import MinMaxScaler

                sc_X = MinMaxScaler(feature_range=(-1, 1))
                sc_y = StandardScaler()

                sX_train = sc_X.fit_transform(X_train)
                sy_train = sc_y.fit_transform(y_train)
                sX_test = sc_X.transform(X_test)
                sy_test = sc_y.transform(y_test)

                return sX_train, sy_train, sX_test, sy_test, sc_X, sc_y

        else:
            print("Please check the module name you typed and try again")

else:
    print("Block not available")
    sys.exit(1)
# -------------------------------------------------------------------------------------
# read information from original .scs file
# -------------------------------------------------------------------------------------
# get meausrement
# get linenum, parname, parvalue
# initiate user input
def val2minmax(parvalue, module):
    # transalte the KGD into defualt sample range
    vdef = []
    vmin = []
    vmax = []
    step = []
    vmatric = []
    i = 0
    for val in parvalue:
        vset = []
        temp = val.split('e')
        #vmin = module.minpar[i]
        #vmax = module.maxpar[i]
        vmin = float(temp[0]) * 0.8
        vmax = float(temp[0]) * 1.2
        step = 1
        vmatric = '1e' + temp[1]
        vset.append(temp[0])
        vset.append(vmin)
        vset.append(vmax)
        vset.append(step)
        vset.append(vmatric)
        vdef.append(vset)
        i += 1
    return vdef


def getlineparam(filename, keyword):
    # return the line number for the line contains keyword
    # return -1 when no exit
    # return
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    i = 1
    for line in lines:
        if keyword == line[0:len(keyword)]:
            return i
        i = i + 1
    return -1


def getparamnames(filename, lineparam, keywords):
    # get the names list and value list from line = lineparam in filename for
    # exp line: tran tran stop=100u errpreset=conservative write="spectre.ic" writefinal="spectre.fc" annotate=status maxiters=5
    # this function will return two list: ['stop','errpreset','write','writefinal','annotate','maxiters'],['100u', 'conservative', '"spectre.ic"', '"spectre.fc"', 'status', '5']
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    line = lines[lineparam - 1]
    i = 1
    while (line[-2] == '\\' or line[-1] == '\\'):
        line = line[0:-3] + lines[lineparam - 1 + i]
        i = i + 1
    line = line[len(keywords):]
    linestr = line.split()
    simname = []
    simvalue = []
    for st in linestr:
        temp = st.split("=")
        simname.append(temp[0])
        simvalue.append(temp[1])
    return simname, simvalue


def getinfo(filename, keywords):
    # this function return the linenumber, parameter name, parameter value for the line contains the keywords from filename
    # read the filename and keywords list
    # return the linenum, parname, parvalue for each keywords
    # call this function at first to get the basic information for keywords from filename
    linenumlist = []
    parname = []
    parvalue = []
    for st in keywords:
        linenum = getlineparam(filename=filename, keyword=st)
        linenumlist.append(linenum)
        parname_temp, parvalue_temp = getparamnames(filename=filename, lineparam=linenum, keywords=st)
        parname.append(parname_temp)
        parvalue.append(parvalue_temp)
    return linenumlist, parname, parvalue


def userinputinit(module, parname, parvalue, inputfile, gennames, genvalues, keywords):
    # initialize specs.in for user to set
    # 1. max - min - step - metrc
    # 2. simulation setting
    #
    fp = open(inputfile, 'w')
    # set the parname format
    fp.write('----------sampling parameter specs----------\n\
#name: parameter name\n\
#KGD: KGD parameter values\n\
#min: minimum range\n\
#max: maximum range\n\
#step: sample step\n\
#matric: metric for min,max,step\n')
    string = ['name', 'KGD', 'min', 'max', 'step', 'metric']
    dash = ''
    for st in string:
        fp.write(f'{st:10}' + '|')
        dash = dash + 10 * '-'
    fp.write("\n")
    fp.write(dash + "\n")
    n = 0
    vdef = val2minmax(parvalue,module)
    for parstr in parname:
        vdef_temp = vdef[n]
        fp.write(f'{parstr:<10}' + ' ')
        i = 0
        for st in string[1:]:
            fp.write(f'{vdef_temp[i]:<10}' + ' ')
            i = i + 1
        fp.write("\n")
        n = n + 1
    fp.write('\n--------------sampling method sepcs-----------\n')
    fp.write('#Command format:/sampling options/ /number of samples/\n')
    fp.write('#Sampling options:\n\
#uniform: uniform random sampling\n')
    # fp.write('sampling method = Uniform \n')
    fp.write('uniform 10000 \n')

    fp.write('\n--------------regression method sepcs-----------\n')
    fp.write('regressor options:\n\
#ANN: Aritificial Neuron Network\n')
    fp.write('ANN\n\
layers 250 250\n\
act elu \n\
epochs 800\n')
    fp.write('\n')
    fp.write('endfile')

    fp.close()


def getsample(parname, inputfile):
    # get sample setting from inputfile
    # return  (max - min - step)  * metrc
    nline_1 = 9  # number of lines to skip to get min, max, stp
    nline_2 = 7  # number of lines to skip to get sampling methods
    # nline_3 = # number of lines to skip to get regression settings

    nlen = len(parname)
    fp = open(inputfile, 'r')
    minpar = []
    maxpar = []
    stp = []
    metric = []
    parset = [minpar, maxpar, stp, metric]
    # go to the parname line
    for i in range(nline_1):
        fp.readline()
    # read parname into min,max,step,matric
    for i in range(nlen):
        linestr = fp.readline().split()
        # matrixstr = linestr[-1]
        # parset[2].append(float(matrixstr))
        n = 0
        for j in linestr[2:6]:
            parset[n].append(float(j))
            n = n + 1

    sp_methods = []
    reg_methods = []
    STOP = False
    annstart = False
    ANN = {'layers': [], 'act': [], 'epochs': 0}
    while not STOP:
        linestr = fp.readline().split()

        if len(linestr) > 0:

            if linestr[0] == 'uniform':
                sp_methods.append([linestr[0], int(linestr[1])])

            if linestr[0] == 'ANN':
                annstart = True

            if annstart:

                if linestr[0] == 'layers':
                    ANN[linestr[0]] = linestr[1:len(linestr)]
                if linestr[0] == 'act':
                    ANN[linestr[0]] = linestr[1:len(linestr)]
                if linestr[0] == 'epochs':
                    ANN[linestr[0]] = linestr[1:len(linestr)]
                    
                if linestr[0] == 'endfile':
                    STOP = True
                    reg_methods.append('ANN')
                    reg_methods.append(ANN)

            if linestr[0] == 'endfile':
                STOP = True

    fp.close()
    return minpar, maxpar, stp, metric, sp_methods, reg_methods


linenum, parname, parvalue = getinfo(filename=finputscs, keywords=keywords)
userinputinit(module = module, parname=parname[0], parvalue=parvalue[0], inputfile=inputfile, gennames=parname[1:],
              genvalues=parvalue[1:], keywords=keywords[1:])



#print(
#    "#Default simulation will use uniform sampling in the range of (0.8 ~ 1.2)*(target parameter) \n")

print("#Please modify %s file to specify your simulation\n" % inputfile)
while (input("If you compelete the %s file, please print yes: " % inputfile) != 'yes'):
    print("pls print yes if you complete the file")
# tedad = int(input("number of test to run:"))
print("user confirm the input file")
# get sample setting, simulation setting from user
minpar, maxpar, stp, metric, sp_methods, reg_methods = getsample(parname=parname[0], inputfile=inputfile)

# module.minpar = np.array(minpar)
# module.maxpar = np.array(maxpar)
# module.stppar = np.array(stp)

# module.put_on_csv(tedad=10,outcsv =  module.finaldataset,do_header = True)


os.system(clear_command)

print("Simulation start")
n = 0  # n is the number of samples
for i in range(len(sp_methods)):

    if sp_methods[i][0] == 'uniform':
        print('Start uniform sampling')
        tedad = sp_methods[i][1]
        module.put_on_csv(tedad=tedad, outcsv=module.finaldataset, do_header=True)

    else:
        print('Cant recognize the method')
        sys.exit(1)

print('-------------------------------------------------------\n')
print('-------------------------------------------------------\n')
print('                  Sampling end                       \n')
print('-------------------------------------------------------\n')
print('-------------------------------------------------------\n')

if (input("Do you want to run the Regression part?(yes/no): ") == 'yes'):
    if len(reg_methods) == 0:
        print('No regression model is specified')
        sys.exit(1)

    elif reg_methods[0] == 'ANN':

        dataset = pd.read_csv(data_address, header=None)
        sX_train, sy_train, sX_test, sy_test, sc_X, sc_y = preprocessing(dataset)

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, BatchNormalization
        from tensorflow.keras import losses
        from tensorflow.keras import optimizers
        import tensorflow.keras.initializers as init

        ANN = reg_methods[1]
        layers = ANN['layers']
        func = ANN['act'][0]
        epochs = int(ANN['epochs'][0])

        reg = Sequential()
        reg.add(Dense(units=layers[0], kernel_initializer=init.glorot_uniform(), activation=func, input_dim=input_dim))
        if len(layers) < 1:
            print('layers error: at least 1 hidden layer')
            sys.exit(1)
        else:
            for i in range(1, len(layers)):
                reg.add(Dense(units=layers[i], kernel_initializer=init.glorot_uniform(), activation=func))

            reg.add(Dense(units=output_dim, kernel_initializer=init.glorot_uniform(), activation='linear'))

            reg.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.mean_absolute_error)
            reg.fit(sX_train, sy_train, validation_split=0.1, batch_size=500, epochs=epochs)

            score = reg.evaluate(sX_test, sy_test, batch_size=500)

            print(score)

            import pickle

            name = 'compp65'

            reg_json = reg.to_json()
            with open(save_addr + 'model_' + name + '.json', "w") as json_file:
                json_file.write(reg_json)
            reg.save_weights(save_addr + 'reg_' + name + '.h5')

            from sklearn.externals import joblib

            joblib.dump(sc_X, save_addr + 'scX_' + name + '.pkl')
            joblib.dump(sc_y, save_addr + 'scY_' + name + '.pkl')
            pickle.dump(reg.get_weights(), open(save_addr + 'w8_' + name + '.p', "wb"))
            # pickle.dump( scores, open( save_addr+'err_'+name+'.p', "wb" ) )
    else:
        print('Reg method is not recongnized')
        sys.exit(1)

# reg = Sequential()
# reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid', input_dim = 15))
# reg.add(Dense(units = 512, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
# reg.add(Dense(units = 256, kernel_initializer = init.glorot_uniform(), activation = 'sigmoid'))
# reg.add(Dense(units = 8, kernel_initializer = init.glorot_uniform(), activation = 'linear'))

else:
    print('Quit')
    sys.exit(0)
print('=========================================================\n') 
print('======================== Regression Ends=================\n')    
print('***Regressor is saved in' + save_addr)
print('***If you have finished all the modules, please run Auto_Search.py to generate the block level design')

print('-------------------------------------------------------\n')
print('-------------------------------------------------------\n')
print('                    End                                \n')
print('-------------------------------------------------------\n')
print('-------------------------------------------------------\n')
