import numpy as np
import matplotlib.pyplot as plt

# Parametry symulacji

h = 6.58e-16  # (eV*s)
e = 1.6e-19  # (C)
m_e = 0.067*9.1e-31  # masa elektronu * 0.067 (kg)
a = 1e-9  # (m)
E_c = 0  # Dno pasma przewodnictwa (eV)
E_e = 1e-3  # (eV/m)
dk = 1e-5  # Mała zmiana k
B = 0.001  

V_ss_sigma = (-h**2 / (2 * m_e * a**2))


# częstość oscylacji Blocha
w_b =  e * E_e * a / h

T = 1
#T = 2 * np.pi / w_b  
#time =  1e-7 #3 * T  

T_cykl = (2*np.pi*m_e)/(e*B)  #okres ruchu cyklotronowego
time = 2*T_cykl

#dt = T / 1000  
dt = 0.0000000002

def E(kx, ky):
    return E_c + 2 * (-h**2 / (2 * m_e * a**2)) * (np.cos(kx * a) + np.cos(ky * a))

def dE_dkx(kx, ky):
    return (E(kx + dk, ky) - E(kx - dk, ky)) / (2 * dk)

def dE_dky(kx, ky):
    return (E(kx, ky + dk) - E(kx, ky - dk)) / (2 * dk)

#wzor 6.63 rownania x z kropka itd

def dx_dt(t, kx, ky): # x
    return 1 / h * dE_dkx(kx, ky)

def dy_dt(t, kx, ky): #y
    return 1 / h * dE_dky(kx,ky)

def dkx_dt(t, vx, vy,kx, ky):
    vy = 1 / h * dE_dky(kx, ky)
    return -e / h * (E_e + B * vy)

def dky_dt(t, vx, vy, kx, ky):
    vx = 1 / h * dE_dkx(kx,ky)
    return -e / h * (E_e - B * vx)



# Metoda RK4
def RK4_step(f, t, y, dt):
    k1 = f(t, *y)
    k2 = f(t + dt/2, *(yi + dt/2 * ki for yi, ki in zip(y, k1)))
    k3 = f(t + dt/2, *(yi + dt/2 * ki for yi, ki in zip(y, k2)))
    k4 = f(t + dt, *(yi + dt * ki for yi, ki in zip(y, k3)))
    return [yi + dt/6 * (ki1 + 2*ki2 + 2*ki3 + ki4) for yi, ki1, ki2, ki3, ki4 in zip(y, k1, k2, k3, k4)]

time_values = np.arange(0, time, dt)
x_values = []
y_values = []
kx_values = []
ky_values = []


#poczatkowe wartosci
x_0 = 0
y_0 = 0
kx_0 = 1e9
ky_0 = 1e9

x_1 = x_0
y_1 = y_0
kx_1 = kx_0
ky_1 = ky_0

def f(t, vx, vy, kx, ky):
    return [dx_dt(t, kx, ky), dy_dt(t, kx, ky), dkx_dt(t, vx, vy, kx, ky), dky_dt(t, vx, vy, kx, ky)]

for t in time_values:
    x_values.append(x_1)
    y_values.append(y_1)
    kx_values.append(kx_1)
    ky_values.append(ky_1)
    
    x_1, y_1, kx_1, ky_1 = RK4_step(f, t, [x_1, y_1, kx_1, ky_1], dt)
plt.figure(figsize=(10, 8))

# Wykres vx(t) ---> x(t)
plt.subplot(4, 1, 1)
plt.plot(np.array(time_values)/T_cykl, np.array(x_values)/ (4*V_ss_sigma*a/h/w_b) )
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.grid(True)

# Wykres vy(t) ---> y(t)
plt.subplot(4, 1, 2)
plt.plot(np.array(time_values)/T_cykl, np.array(y_values)/ (4*V_ss_sigma*a/h/w_b) )
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.grid(True)

# Wykres kx(t)
plt.subplot(4, 1, 3)
plt.plot(np.array(time_values)/T_cykl, np.array(kx_values)/(w_b*T/a))
plt.xlabel('t (s)')
plt.ylabel('kx(t)')
plt.grid(True)

# Wykres ky(t)
plt.subplot(4, 1, 4)
plt.plot(np.array(time_values)/T_cykl, np.array(ky_values)/(w_b*T/a))
plt.xlabel('t (s)')
plt.ylabel('ky(t)')
plt.grid(True)

plt.tight_layout()
plt.show()


# Mapa y(x)
plt.scatter(np.array(x_values)/ (4*V_ss_sigma*a/h/w_b), np.array(y_values)/ (4*V_ss_sigma*a/h/w_b), c=np.array(y_values), cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa y(x)')
plt.colorbar(label='y')
plt.grid(True)
plt.show()

# Mapa ky(kx)
plt.scatter(np.array(kx_values)/(w_b*T/a), np.array(ky_values)/(w_b*T/a), c=np.array(ky_values), cmap='viridis')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Mapa ky(kx)')
plt.colorbar(label='ky')
plt.grid(True)
plt.show()


