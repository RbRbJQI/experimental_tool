# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import timeit
from numpy import pi, cos, sin
import pickle
start = timeit.default_timer()

def find_mstar(s):
    hbar = 1.0545718e-34
    mass = 87*1.66054e-27
    lamb = 1e-6
    d = lamb/2
    Er = (hbar*pi/d)**2/2/mass
    kr = pi/d
    
    L = 1*d
    N = 150
    z = np.linspace(0,L,N)
    z_step = z[1] - z[0]
    def V_pot(z):
        return s*Er*sin(kr*z)**2
    kN = 31
    kdim = np.linspace(-kr,kr, kN)
    
    e_lb = [] #lowest band energy
    for k in kdim:
        M_bdg = np.zeros((N, N), dtype='complex')
        #u
        for iz in range(N):
            #V_pot
            M_bdg[iz, iz] += hbar**2/2/mass*k**2 + V_pot(z[iz])
            #
            #Kinetic
            #2nd derivative
            if iz<N-1:
                M_bdg[iz, iz+1] += -hbar**2/2/mass/z_step**2
            else:
                M_bdg[iz, 0] += -hbar**2/2/mass/z_step**2
            if iz>0:
                M_bdg[iz, iz-1] += -hbar**2/2/mass/z_step**2
            else:
                M_bdg[iz, N-1] += -hbar**2/2/mass/z_step**2
            M_bdg[iz, iz] += 2*hbar**2/2/mass/z_step**2
            #1st derivative
            if iz<N-1:
                M_bdg[iz, iz+1] += 1j* hbar**2/mass*k/2/z_step
            else:
                M_bdg[iz, 0] += 1j* hbar**2/mass*k/2/z_step
            if iz>0:
                M_bdg[iz, iz-1] += -1j* hbar**2/mass*k/2/z_step
            else:
                M_bdg[iz, N-1] += -1j* hbar**2/mass*k/2/z_step
        e, uvs = LA.eig(M_bdg)
        uvs = uvs.T
        e = e/hbar
        
        e = np.sort(np.abs(np.real(e)))
        e_lb.append(e[0])
#        for i in range(3): plt.scatter( (k/1e6), (e[i]), s=3)
#        plt.xlabel('k (1/um)')
#        plt.ylabel('w (Hz)')
#        print('k=',k/1e6,'e6, E0=',e[0])
    e_lb = np.array(e_lb)
    ik_c = int(len(e_lb)/2)
    ik_r = int(kN/8)
    k_fr = kdim[ik_c-ik_r:ik_c+ik_r]
    f = np.polyfit(k_fr, e_lb[ik_c-ik_r:ik_c+ik_r], 2)
#    plt.plot(kdim/1e6, f[0]*kdim**2+f[1]*kdim+f[2])
    m0 = 2739822552
    print(1/f[0]/m0)
    return 1/f[0]/m0

def compare_exp(f_inv):
    '''lattice'''
#    V_lat_scope_reading = [0, 76, 150, 256, 330, 412, 564, 616, 704]
#    fx = [20.7, 20.5, 17.2, 15.5, 13.8, 11.2, 9.5, 8.9, 8.0]
    V_lat_scope_reading = np.array([76, 256, 412, 512, 616, 704])
    fx = np.array( [19.1, 16.2, 12.6, 10.5, 11.4, 9.2] )
    V_lat_scope_reading, fx = np.array(V_lat_scope_reading), np.array(fx)
    m_star = 1/(fx/20)**2
#    m_star = 1/(fx/fx[0])**2
    V_lat = V_lat_scope_reading/V_lat_scope_reading[-1]*10.5
    plt.figure('lattice')
#    plt.scatter(V_lat, m_star)
#    plt.xlabel('lattice depth /Er')
#    plt.ylabel('m*/m')
#    plt.title('Effective mass')
    
#    V_lat_scope_reading = np.delete(V_lat_scope_reading,1)
#    m_star = np.delete(m_star,1)
#    U0_calc = f_inv(m_star[1:])
    U0_calc = f_inv(m_star)
    U0_calc = np.insert(U0_calc, 0, 0)
    V_lat_scope_reading = np.insert(V_lat_scope_reading, 0, 0)
    plt.scatter(V_lat_scope_reading, U0_calc)
    
    l_fit = np.polyfit(V_lat_scope_reading, U0_calc,1)
    plt.plot(V_lat_scope_reading, l_fit[1]+l_fit[0]*V_lat_scope_reading)
    plt.title('y=%f+%f*x'%(l_fit[1],l_fit[0]))
    
    plt.xlabel('lattice scope reading/mv')
    plt.ylabel('lattice depth/Er')

if __name__=='__main__':
    s = np.linspace(0,10,8)
#    s = np.linspace(0,5,2)
    m_s = [find_mstar(ss) for ss in s]
    plt.figure()
#    plt.scatter(s, m_s)
    plt.plot(s, m_s)
    plt.xlabel('s')
    plt.ylabel('m*/m')
    from scipy import interpolate
    f = interpolate.interp1d(s, m_s)
    f_inv = interpolate.interp1d(m_s, s)
    s_dict = {'f':f, 'f_inv':f_inv}
    pickle.dump( s_dict, open( "effective_mass.p", "wb" ) )
    
#    s_dict = pickle.load( open( "effective_mass.p", "rb" ) )
#    f, f_inv = s_dict['f'], s_dict['f_inv']
#    compare_exp(f_inv)
#    plt.plot(s, f(s))
    
    plt.show()
    
    