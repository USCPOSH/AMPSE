import os
import sys
import numpy as np

block = input('Choose the block level Design you want to generate (Options: SAR_ADC, XXX, XXX): ')
config = 'block.in'
if block == 'SAR_ADC':
    
    home_add = input('Please specify the home address: ')
    fp = open(config, 'w')
    fp.write('----------SAR_ADC specs----------\n\
Number of bits: 12\n\
Sampling frequency: 1e8\n\
\n\
----------Optimization specs----------\n\
Optimization method (GD, Adams): Adams\n\
\n\
end file')
    fp.close()
    
    print('Please modify the condigure file block.in\n')
    while (input("If you compelete the config file, please type yes: ") != 'yes'):
        print("pls print yes if you complete the file")
# tedad = int(input("number of test to run:"))
    print("user confirm the input file")
    
    
    STOP = False
    fp = open(config, 'r')
    while not STOP:
        line = fp.readline()
        linestr = line.split()
        
        if len(linestr) > 0:
            if linestr[0] == 'Number':
                nbits = line.split(':')[1].strip('\n')
            
            elif linestr[0] == 'Sampling':
                frequency = line.split(':')[1].strip('\n')
    
            elif linestr[0] == 'Optimization':
                opt = line.split(':')[1].strip('\n')
                
            elif linestr[0] == 'end':
                STOP = True
    
    #nbits = input('Please specify the number of bits: ')
    #frequency = input('Please specify the frequency: ')
    
    # Run AMMPSE Search

    #os.system('python AMPSE_Graphs.py' + ' ' + nbits + ' ' + frequency)
    
    command = 'python search.py' + ' ' + nbits + ' ' + frequency + ' ' + home_add + ' ' + opt
    
    print(command)
    os.system(command)
else:
    print('Please check the block name')
    sys.exit(1)