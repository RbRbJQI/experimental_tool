'''
Copyright: Junheng Tao, Mingshu Zhao
Date: 5/13/2022
'''
#------------------------------
def lat(t, y, a, b, n):
    dydt = []
    for i in range(2*n+1):
        j = i-n
        if i==0:
            dydt.append( (a*j**2+b/2)*y[i] + b/4*y[i+1] )
        elif i==2*n:
            dydt.append( b/4*y[i-1] + (a*j**2+b/2)*y[i] )
        else:
            dydt.append( b/4*y[i-1] + (a*j**2+b/2)*y[i] + b/4*y[i+1] )
    return np.array( dydt )/1j
from scipy.integrate import complex_ode
hbar = 1.055e-34
mass = 87* 1.66e-27
lam = 532e-9

Er = ( hbar*(2*np.pi/lam) )**2/2/mass
U0 = 12.5*Er
a = 4*Er/hbar
b = U0/hbar
n = 3
y0 = np.zeros(2*n+1)
y0[n] = 1

class myfuncs(object):
    def __init__(self, f, fargs=[]):
        self._f = f
        self.fargs=fargs
    def f(self, t, y):
        return self._f(t, y, *self.fargs)
        
case = myfuncs(lat, fargs=[a,b,n])
r = complex_ode(case.f)
r.set_initial_value(y0, 0)

t1 = 20e-6
dt = 0.1e-6
t = []
y = []
y1 = []
y2 = []
while r.successful() and r.t < t1:
    t.append(r.t+dt)
    y.append(abs(r.integrate(r.t+dt)[n])**2)
    y1.append(abs(r.integrate(r.t+dt)[n+1])**2)
    y2.append(abs(r.integrate(r.t+dt)[n+2])**2)
    # print(r.t+dt, r.integrate(r.t+dt))
t = np.array(t) *1e3
t += 0.5e-3
y, y1 = np.array(y), np.array(y1)
# plt.plot(t, y)
# plt.plot(t, y1)
# plt.plot(t, y2)
axes[0].plot(t, y1/(y+y1))