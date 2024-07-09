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
w_b =  e * E_e * a / h
T = 1
T_cykl = (2*np.pi*m_e)/(e*B)  #okres ruchu cyklotronowego
time = 2*T_cykl
dt = 0.00000000002

V_max = 1e-4
y_bariera = 1e-3  # polozenie schodka potencjału
step = 1e-5  #nachylenie schodka

def potencjal(x, y):
    return V_max / (np.exp(-((y - y_bariera) / step)) + 1)

def pole_elektryczne(x, y):
    E_x = 0
    dy = 1e-9
    E_y = -(potencjal(x, y + dy) - potencjal(x, y - dy)) / (2 * dy)
    return E_x, E_y

def E(kx, ky):
    return E_c + 2 * (-h**2 / (2 * m_e * a**2)) * (np.cos(kx * a) + np.cos(ky * a))

def dE_dkx(kx, ky):
    return (E(kx + dk, ky) - E(kx - dk, ky)) / (2 * dk)

def dE_dky(kx, ky):
    return (E(kx, ky + dk) - E(kx, ky - dk)) / (2 * dk)

def dx_dt(t, kx, ky):
    return 1 / h * dE_dkx(kx, ky)

def dy_dt(t, kx, ky):
    return 1 / h * dE_dky(kx, ky)

def dkx_dt(t, vx, vy, kx, ky):
    vy = 1 / h * dE_dky(kx, ky)
    return -e / h * (E_e + B * vy)

def dky_dt(t, vx, vy, kx, ky):
    vx = 1 / h * dE_dkx(kx, ky)
    return -e / h * (E_e - B * vx)

def kolizja(x, y, vx, vy):
    print(f"==== Kolizja: Polozenie: x={x:.2e}, y={y:.2e}, Predkosc: vx={vx:.2e}, vy={vy:.2e}")

def warunki(x, y, vx, vy, L, dt, xp, yp, vxp, vyp):
    lambda_val = 1.0  
    
    if y < -L:
        dy = yp + L
        dx = np.abs(dy) / np.abs(y - yp) * np.abs(x - xp)
        y = -L + 1e-6
        n = np.array([0, 1])  # wektor normalny
        v = np.array([vx, vy])
        V_n = np.dot(n, v) * n
        V_s = v - V_n
        v_new = V_s - lambda_val * V_n
        
        ##Ek_0 = 0.5 * m_e * (vx ** 2 + vy ** 2)
        vx, vy = v_new
       ## Ek_1 = 0.5 * m_e * (vx ** 2 + vy ** 2)
        
        ##if Ek_1 > Ek_0:
         ##   r = np.sqrt(Ek_0 / Ek_1)
         ##   vx *= r
         ##   vy *= r
        
        x = xp + dx
        kolizja(xp + dx, y, vx, vy)
    
    return x, y, vx, vy

def RK4_step(f, t, y, dt):
    k1 = f(t, *y)
    k2 = f(t + dt/2, *(yi + dt/2 * ki for yi, ki in zip(y, k1)))
    k3 = f(t + dt/2, *(yi + dt/2 * ki for yi, ki in zip(y, k2)))
    k4 = f(t + dt, *(yi + dt * ki for yi, ki in zip(y, k3)))
    return [yi + dt/6 * (ki1 + 2 * ki2 + 2 * ki3 + ki4) for yi, ki1, ki2, ki3, ki4 in zip(y, k1, k2, k3, k4)]

time_values = np.arange(0, time, dt)
x_values = []
y_values = []
kx_values = []
ky_values = []

x_0 = 0
y_0 = 0
kx_0 = 0.1e9
ky_0 = 0.1e9

x_1 = x_0
y_1 = y_0
kx_1 = kx_0
ky_1 = ky_0

def f(t, x, y, kx, ky):
    vx = 1 / h * dE_dkx(kx, ky)
    vy = 1 / h * dE_dky(kx, ky)
    dkx = dkx_dt(t, vx, vy, kx, ky)
    dky = dky_dt(t, vx, vy, kx, ky)
    E_x, E_y = pole_elektryczne(x, y)
    return [vx, vy, dkx + e * E_x / h, dky + e * E_y / h]

for t in time_values:
    x_values.append(x_1)
    y_values.append(y_1)
    kx_values.append(kx_1)
    ky_values.append(ky_1)
    
    x_1, y_1, kx_1, ky_1 = RK4_step(f, t, [x_1, y_1, kx_1, ky_1], dt)
    x_1, y_1, kx_1, ky_1 = warunki(x_1, y_1, kx_1, ky_1, y_bariera, dt, x_values[-1], y_values[-1], kx_values[-1], ky_values[-1])

plt.figure(figsize=(10, 8))

# Wykresy
plt.subplot(4, 1, 1)
plt.plot(np.array(time_values)/T_cykl, np.array(x_values)/(4 * V_ss_sigma * a / h / w_b))
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(np.array(time_values)/T_cykl, np.array(y_values)/(4 * V_ss_sigma * a / h / w_b))
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(np.array(time_values)/T_cykl, np.array(kx_values)/(w_b * T / a))
plt.xlabel('t (s)')
plt.ylabel('kx(t)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(np.array(time_values)/T_cykl, np.array(ky_values)/(w_b * T / a))
plt.xlabel('t (s)')
plt.ylabel('ky(t)')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.scatter(np.array(x_values)/(4 * V_ss_sigma * a / h / w_b), np.array(y_values)/(4 * V_ss_sigma * a / h / w_b), c=np.array(y_values), cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa y(x)')
plt.colorbar(label='y')
plt.grid(True)
plt.show()

plt.scatter(np.array(kx_values)/(w_b * T / a), np.array(ky_values)/(w_b * T / a), c=np.array(ky_values), cmap='viridis')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Mapa ky(kx)')
plt.colorbar(label='ky')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(np.array(x_values)/(4 * V_ss_sigma * a / h / w_b), np.array(y_values)/(4 * V_ss_sigma * a / h / w_b), c='blue', label='y(x)')
plt.scatter(np.array(kx_values)/(w_b * T_cykl / a), np.array(ky_values)/(w_b * T_cykl / a), c='red', label='ky(kx)')
plt.xlabel('x / kx')
plt.ylabel('y / ky')
plt.title('y(x) i ky(kx)')
plt.legend()
plt.grid(True)
plt.show()

# Wykres schodka potencjału
# y_values_potential = np.linspace(-1e-2, 1e-2, 1000)
# potential_values = [potencjal(0, y) for y in y_values_potential]

# plt.figure(figsize=(10, 8))
# plt.plot(y_values_potential, potential_values)
# plt.xlabel('y')
#plt.ylabel('V')
# plt.grid(True)
# plt.show()

