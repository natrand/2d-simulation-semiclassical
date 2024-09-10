import numpy as np
import matplotlib.pyplot as plt

# Parametry symulacji
h = 6.58e-16  # (eV*s)
e = 1.6e-19  # (C)
m_e = 0.067*9.1e-31  # masa elektronu * 0.067 (kg)
a = 1e-9  # (m)
E_c = 0  # Dno pasma przewodnictwa (eV)
E_e = 0 #1e-3  # (eV/m)
dk = 1e-5  # Mała zmiana k
dk = 1e-2  # Mała zmiana k
B = 0.001  

V_ss_sigma = (-h**2 / (2 * m_e * a**2))
 
# częstość oscylacji Blocha
w_b = 1 # e * E_e * a / h

T = 1
# T = 2 * np.pi / w_b  
# time =  1e-7 #3 * T  

T_cykl = (2*np.pi*m_e)/(e*B)  # okres ruchu cyklotronowego
time = 2*T_cykl
print("okres ", T_cykl)

# dt = T / 1000  
dt = 0.0000000000002
dt = 2e-13
dt = 2e-11


def E(kx, ky):
    return E_c + 2 * (-h**2 / (2 * m_e * a**2)*e) * (np.cos(kx * a) + np.cos(ky * a)) #h^2 * (kx^2 + ky^2)/2m +eC
#    return E_c + 2 * (-h**2 / (2 * m_e * a**2)) * (np.cos(kx * a) + np.cos(ky * a))

def dE_dkx(kx, ky):
    return (E(kx + dk, ky) - E(kx - dk, ky)) / (2 * dk)

def dE_dky(kx, ky):
    return (E(kx, ky + dk) - E(kx, ky - dk)) / (2 * dk)

def dx_dt(t, kx, ky): # x
    return 1 / h * dE_dkx(kx, ky)

def dy_dt(t, kx, ky): # y
    return 1 / h * dE_dky(kx, ky)

def dkx_dt(t, x, y, kx, ky):
    vy = 1 / h * dE_dky(kx, ky)
    return -e / h * (E_e + B * vy)/(e)

def dky_dt(t, x, y, kx, ky):
    vx = 1 / h * dE_dkx(kx, ky)
    return -e / h * (E_e - B * vx)/(e)

def promien_cyklotronowy(kx, ky):
    vx = 1 / h * dE_dkx(kx, ky)
    vy = 1 / h * dE_dky(kx, ky)
    v_prostopadla = np.sqrt(vx**2 + vy**2)  
    return (v_prostopadla * m_e) / (e * B)


def promien_cyklotronowy_teoretyczny(kx, ky):
    # E_kinetyczna = (h^2* k^2 / 2m) 
    k_m = np.sqrt(kx**2 + ky**2)
    E_kinetyczna = (h**2 * k_m**2) / (2 * m_e)
    v_teoretyczna = np.sqrt(2 * E_kinetyczna / m_e)
    return (m_e * v_teoretyczna) / (e * B)



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
vx_values = []
vy_values = []

# Początkowe wartości
x_0 = 0
y_0 = 0
kx_0 = 0.1e9
ky_0 = 0.1e9

# Obliczenie promienia cyklotronowego i porownanie wynikow
r_c = promien_cyklotronowy(kx_0, ky_0)
r_c_teoretyczny = promien_cyklotronowy_teoretyczny(kx_0, ky_0)
print(f"Promień cyklotronowy: {r_c:.3e}")
print(f"Promień cyklotronowy teoretyczny: {r_c_teoretyczny:.3e}")

x_1 = x_0
y_1 = y_0
kx_1 = kx_0
ky_1 = ky_0

def f(t, x, y, kx, ky):
    return [dx_dt(t, kx, ky), dy_dt(t, kx, ky), dkx_dt(t, x, y, kx, ky), dky_dt(t, x, y, kx, ky)]

for t in time_values:
    x_values.append(x_1)
    y_values.append(y_1)
    kx_values.append(kx_1)
    ky_values.append(ky_1)
    
    x_1, y_1, kx_1, ky_1 = RK4_step(f, t, [x_1, y_1, kx_1, ky_1], dt)
    
    # Oblicz prędkości vx, vy na podstawie kx, ky
    vx = 1 / h * dE_dkx(kx_1, ky_1)
    vy = 1 / h * dE_dky(kx_1, ky_1)
    vx_values.append(vx)
    vy_values.append(vy)





# Wykresy

plt.figure(figsize=(10, 8))

# Wykres x(t)
plt.subplot(4, 1, 1)
plt.plot(np.array(time_values)/T_cykl, np.array(x_values) / (4 * V_ss_sigma * a / h / w_b))
plt.xlabel('t (s)')
plt.ylabel('x(t)')
plt.grid(True)

# Wykres y(t)
plt.subplot(4, 1, 2)
plt.plot(np.array(time_values)/T_cykl, np.array(y_values) / (4 * V_ss_sigma * a / h / w_b))
plt.xlabel('t (s)')
plt.ylabel('y(t)')
plt.grid(True)

# Wykres kx(t)
plt.subplot(4, 1, 3)
plt.plot(np.array(time_values)/T_cykl, np.array(kx_values) / (w_b * T / a))
plt.xlabel('t (s)')
plt.ylabel('kx(t)')
plt.grid(True)

# Wykres ky(t)
plt.subplot(4, 1, 4)
plt.plot(np.array(time_values)/T_cykl, np.array(ky_values) / (w_b * T / a))
plt.xlabel('t (s)')
plt.ylabel('ky(t)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Mapa y(x)
plt.axes().set_aspect('equal')
plt.scatter(np.array(x_values) , np.array(y_values) , c=np.array(y_values), cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mapa y(x)')
plt.colorbar(label='y')
plt.grid(True)
plt.show()


# Mapa ky(kx)
plt.axes().set_aspect('equal')
plt.scatter(np.array(kx_values) , np.array(ky_values), c=np.array(ky_values), cmap='viridis')
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('Mapa ky(kx)')
plt.colorbar(label='ky')
plt.grid(True)
plt.show()



# Wykres y(x) i ky(x) nałożonych na siebie
plt.figure(figsize=(8, 6))
plt.axes().set_aspect('equal')
plt.scatter(np.array(x_values) / (4 * V_ss_sigma * a / h / w_b), np.array(y_values) / (4 * V_ss_sigma * a / h / w_b), c='blue', label='y(x)')
plt.scatter(np.array(kx_values) / (w_b * T_cykl / a), np.array(ky_values) / (w_b * T_cykl / a), c='red', label='ky(kx)')
plt.xlabel('x / kx')
plt.ylabel('y / ky')
plt.title('y(x) i ky(kx)')
plt.legend()
plt.grid(True)
plt.show()

#f = open('traj.txt', 'w') #note 'w' = write mode
#for i in range(0,len(x_values)):
#    f.write('%f\t %f\t %f\t %f\t %f\n ' % (time_values[i], x_values[i], y_values[i], kx_values[i], ky_values[i]))
#    #f.write('\n ')
#f.close()



#plt.figure()
#kxx = np.linspace(-np.pi/a, np.pi/a, 40)
#kyy = np.linspace(-np.pi/a, np.pi/a, 40)
#kx, ky = np.meshgrid(kxx,kyy)
#Ekk = E(kx, ky)
#plt.imshow(Ekk, extent=[kxx[0],kxx[-1],kyy[0],kyy[-1]])
#plt.colorbar()
#plt.show()


def scale_and_rotate(kx, ky, h, e, B):
    factor = h / (1 * B)
    # Obrót o 90 stopni zgodnie z ruchem wskazówek zegara: (kx, ky) -> (ky, -kx)
    x_rotated = np.array(ky_values) * factor
    y_rotated = -np.array(kx_values) * factor
    return x_rotated, y_rotated

x_scaled_rotated, y_scaled_rotated = scale_and_rotate(np.array(kx_values), np.array(ky_values), h, e, B)

plt.figure(figsize=(8, 6))
plt.axes().set_aspect('equal')
plt.scatter(np.array(x_values), np.array(y_values), c='blue', label='y(x)')
plt.scatter(x_scaled_rotated, y_scaled_rotated, c='red', label='Po przeskalowaniu ky(kx)')
#plt.scatter(np.array(kx_values), np.array(ky_values) , c='blue', label='ky(kx)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y(x) i przesklowane ky(kx)')
plt.legend()
plt.grid(True)
plt.show()

#sprawdizc przesuniecie o promien cyklotronowy
