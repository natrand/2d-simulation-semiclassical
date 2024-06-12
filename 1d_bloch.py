import numpy as np
import matplotlib.pyplot as plt

#V_ss_sigma = 1  
h = 6.58e-16  # (eV*s)
e = 1.6e-19  # (C)
m_e = 0.067*9.1e-31  # masa elektronu (kg)
a = 1e-9  # (m)
E_c = 0  # Dno pasma przewodnictwa (eV)
E_e = 1e-3  # (eV/m)
dk = 1e-5  # Mała zmiana k

V_ss_sigma = (-h**2 / (2 * m_e * a**2))

# częstość oscylacji Blocha
w_b = e * E_e * a / h

# Zakres czasu
T = 2 * np.pi / w_b  
time = 3 * T 

dt = T / 1000  

def E(k):
    return E_c - 2 * (-h**2 / (2 * m_e * a**2)) * np.cos(k * a)

def dE_dk(k):
    dk = 1000
    return (E(k + dk) - E(k - dk)) / (dk*2)
#    return (E(k + dk) + E(k - dk) - 2 * E(k)) / (dk**2) # to jest druga pochodna :(

def dv_dx_dt(t, v_x_k):
    v, x, k = v_x_k
    dv_dt = 0#2 * V_ss_sigma * a * np.cos(w_b * t) / h
    dx_dt = dE_dk(k) / h #4 * V_ss_sigma * a * np.sin(w_b * t / 2) * np.cos(w_b * t / 2) / (h * w_b)
#    dx_dt = 2 * V_ss_sigma * a * np.sin(k * a) / h
    dk_dt = -e * E_e / h #nie w_b
    return [dv_dt, dx_dt, dk_dt]

# Metoda RK4
def RK4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, [yi + dt/2 * ki for yi, ki in zip(y, k1)])
    k3 = f(t + dt/2, [yi + dt/2 * ki for yi, ki in zip(y, k2)])
    k4 = f(t + dt, [yi + dt * ki for yi, ki in zip(y, k3)])
    return [yi + dt/6 * (ki1 + 2*ki2 + 2*ki3 + ki4) for yi, ki1, ki2, ki3, ki4 in zip(y, k1, k2, k3, k4)]

time_values = np.arange(0, time, dt)
v_values = []
x_values = []
k_values = []

v_0 = 0
x_0 = 0
k_0 = 0

v_1 = v_0
x_1 = x_0
k_1 = k_0
for t in time_values:
    v_values.append(v_1)
    x_values.append(x_1)
    k_values.append(k_1)
    v_1, x_1, k_1 = RK4_step(dv_dx_dt, t, [v_1, x_1, k_1], dt)

plt.figure(figsize=(10, 8))

# Wykres v(t)
plt.subplot(3, 1, 1)
plt.plot(np.array(time_values)/T, np.array(v_values) / ((h**2 / (2 * m_e * a**2))*a*2/h) )
plt.xlabel('t (s)')
plt.ylabel('v(t)')
plt.grid(True)

# Wykres x(t)
plt.subplot(3, 1, 2)
plt.plot(np.array(time_values)/T, np.array(x_values) / (4*V_ss_sigma*a/h/w_b) )
#plt.plot(np.array(time_values)/T, np.array(x_values) )
plt.plot(np.array(time_values)/T, -np.sin(w_b*np.array(time_values)/2)**2, "--")#*(4*V_ss_sigma*a/h/w_b))
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.grid(True)

# Wykres k(t)
plt.subplot(3, 1, 3)
plt.plot(np.array(time_values)/T, np.array(k_values)/(w_b*T/a))
#plt.plot(np.array(time_values), np.array(k_values))
plt.xlabel('t (s)')
plt.ylabel('k(t)')
plt.grid(True)

plt.tight_layout()
plt.show()
