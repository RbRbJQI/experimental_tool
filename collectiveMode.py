'''
collective mode calculation
'''
import numpy as np

fx, fy = 16, 16/0.7 # unit Hz
f_avg = np.sqrt( fx**2 + fy**2 ) 
f1 =  np.sqrt( 3 * f_avg**2 - np.sqrt( 9 * f_avg**4 - 32 * fx**2 * fy**2 ) /2 ) 
f2 =  np.sqrt( 3 * f_avg**2 + np.sqrt( 9 * f_avg**4 - 32 * fx**2 * fy**2 ) /2 ) 

print('quadrupole mode freq (Hz):', f1, f2)
print('scissor mode freq (Hz):', f_avg)
