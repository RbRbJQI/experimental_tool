# -*- coding: utf-8 -*-


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import timeit
from numpy import pi, cos, sin
start = timeit.default_timer()

hbar = 1.0545718e-34
mass = 87*1.66054e-27
lamb = 0.532e-6
d = lamb/2
Er = (hbar*pi/d)**2/2/mass
kr = pi/d
s = 0

L = 1*d
N = 250
z = np.linspace(0,L,N)
z_step = z[1] - z[0]
def V_pot(z):
    return s*Er*sin(kr*z)**2
kN = 11
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
#    plt.figure()
#    plt.imshow(np.abs(M_bdg))
#    plt.figure()
    e, uvs = LA.eig(M_bdg)
#    M_bdg = csr_matrix(M_bdg)
#    e, uvs = eigs(M_bdg, k=11, which='LM', sigma=0)
    uvs = uvs.T
    e = e/hbar
    
    if k==0:
        sort_idx = np.argsort(e)
        uvs = uvs[sort_idx]
        idx_list = [0,1,2,3,4]
        for idx in idx_list:
            plt.figure('wfn')
            wfn = uvs[idx]
            plt.title(str(e[sort_idx][idx]))
            plt.plot(z,np.real(wfn))
            plt.figure('fft')
            ft_n = np.fft.fftshift(np.abs(np.fft.fft(wfn))**2)
            ft_n = ft_n/np.sum(ft_n)
            plt.plot(np.linspace(-pi/(d/N),pi/(d/N),len(wfn))/kr-1, ft_n)
#            plt.hist(np.linspace(-pi/(d/N),pi/(d/N),len(wfn))/kr-1, len(wfn), weights=ft_n)
            plt.xlabel('k/kr')
            plt.ylabel('n(k)')
        
    
    # e = np.sort(np.abs(np.imag(e)))[::-1]
    e = np.sort(np.abs(np.real(e)))
    e_lb.append(e[0])
    plt.figure('dispersion')
    for i in range(3): plt.scatter( (k/1e6), (e[i]), s=3)
    plt.xlabel('k (1/um)')
    plt.ylabel('w (Hz)')
    print('k=',k/1e6,'e6, E0=',e[0])
e_lb = np.array(e_lb)
ik_c = int(len(e_lb)/2)
ik_r = int(kN/8)
k_fr = kdim[ik_c-ik_r:ik_c+ik_r]
#print(e_lb.shape)
f = np.polyfit(k_fr, e_lb[ik_c-ik_r:ik_c+ik_r], 2)
plt.plot(kdim/1e6, f[0]*kdim**2+f[1]*kdim+f[2])
m0 = 2739822552
print(1/f[0]/m0)
plt.show()