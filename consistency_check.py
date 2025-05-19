import numpy as np
import numba
from numba import cuda, float32, njit
import math
from matplotlib import pyplot as plt
from numpy import sqrt, log10, log, radians, exp, sin, tan, linspace, pi, array, sum
import time as tempo
import spiral_BLR

device = cuda.get_current_device()
print(device.MAX_THREADS_PER_BLOCK)

# PARAMETRI FISSI
# 0. geometria BLR
inc = np.radians(30.)
xi_in = 200.
xi_out = 1800.
period = 120.

# 1. emissività
x_c = np.float32(300.0)
A = 84.
d = np.radians(134)
phi_0 = np.radians(24)
p = np.radians(45)
xi_sp = 870.

# 2. luminosità
period = 120.

# 3. specific intensity
lambda_emessa = 6563.
c = 3e5
sigma = np.float32(6563*1200/c)

#dimensioni per array di raggio, angolo, tempo e lunghezze d'onda
size_x = 175
size_phi = 175
size_time = 175
size_lam = 175

#creazione array (memoria CPU)
xi = np.linspace(xi_in, xi_out, size_x)
dr = xi[1]-xi[0]
xi = np.array(xi).astype(np.float32)
phi = np.linspace(0, 2*np.pi, size_phi)
phi = np.array(phi).astype(np.float32)
time = np.linspace(0, 150, size_time)
time = np.array(time).astype(np.float32)
lam = np.linspace(6200, 6900, size_lam)
lam = np.array(lam).astype(np.float32)

#creazione devices per salvare i risultati dei kernel CUDA
#1. Creo gli array necessari sulla memoria della CPU
area = np.zeros([size_lam, size_x,size_phi])
emissivity = np.zeros([size_lam, size_x,size_phi])
luminosity =np.zeros([size_lam, size_x,size_phi])
gauss = np.zeros([size_lam, size_x,size_phi])
flux = np.zeros([size_lam, size_x,size_phi])
flux_def = np.zeros([size_lam])
       
#2. Copio i device sulla memoria della GPU
d_xi = cuda.to_device(xi)
d_phi = cuda.to_device(phi)
d_time = cuda.to_device(time)
d_lam = cuda.to_device(lam)

#threads_per_block/blocks_per_grid per kernel area, emissività, luminosità e specific intensity, kernel prodotto
threads_per_block =  (4,4,4)
blocks_per_grid = (
      (d_lam.size + (threads_per_block[0] - 1)) // threads_per_block[0],
      (d_xi.size + (threads_per_block[1] - 1)) // threads_per_block[1],
      (d_phi.size + (threads_per_block[2] - 1)) // threads_per_block[2]
  )

#not used for now
threads_per_block_1 =  (4,4)
blocks_per_grid_1 = (
      (d_lam.size + (threads_per_block_1[0] - 1)) // threads_per_block_1[0],
      (d_time.size + (threads_per_block_1[1] - 1)) // threads_per_block_1[1]
  )

#threads_per_block/blocks_per_grid per kernel somma
threads_per_block_2 =  (4)
blocks_per_grid_2 = (
      (d_lam.size + (threads_per_block_1[0] - 1)) // threads_per_block_1[0],
      )

#time = [1.]
start_time = tempo.time()
#3. Stessa cosa per i device che conterranno il flusso
for t in time:
    d_area = cuda.to_device(area)
    d_emissivity = cuda.to_device(emissivity)
    d_luminosity = cuda.to_device(luminosity)
    d_gaussian = cuda.to_device(gauss)
    d_flux = cuda.to_device(flux)
    d_flux_def = cuda.to_device(flux_def)

#CALCOLO FLUSSO
#chiamo i singoli kernel che agiscono in questo modo sui device definiti precedentemente
#d_area, d_emissivity, d_luminosity, d_gaussian
#tra i vari kernel, SEMPRE usare cuda.synchronize() per assicurarsi che la GPU abbia finito 
#inoltre copio i valori ottenuti sulla memoria della CPU con copy_to_host()
#area

    spiral_BLR.area_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_area)
    cuda.synchronize()
    #area = d_area.copy_to_host()
    #emissività
    spiral_BLR.emissivity_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, t, d_lam, d_emissivity, A, phi_0, d, p, xi_sp, x_c, inc, period)
    cuda.synchronize()
    #emissivity = d_emissivity.copy_to_host()
    #luminosità
    spiral_BLR.luminosity_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, t, inc, period, d_luminosity)
    cuda.synchronize()
    #luminosity = d_luminosity.copy_to_host()
    #gaussian
    spiral_BLR.gaussian_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_gaussian, inc, lambda_emessa, sigma)
    cuda.synchronize()
    #gaussian = d_gaussian.copy_to_host()
    #prodotto tra i vari contributi (area, emissività, luminosità e gaussiana)
    spiral_BLR.product_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_area, d_luminosity, d_emissivity, d_gaussian, d_flux)
    cuda.synchronize()
    #flux = d_flux.copy_to_host()
    #somma dei vari contributi, di fatto il flusso
    cuda.synchronize()
    spiral_BLR.sum_kernel[blocks_per_grid_2, threads_per_block_2](d_xi, d_phi, d_lam, d_flux, d_flux_def)
    cuda.synchronize()

flux_def = d_flux_def.copy_to_host()
end_time = tempo.time()

plt.plot(lam, flux_def, linestyle = '--', label = 'Flusso GPU')
plt.legend()
plt.xlabel(r'Wavelength $[\AA]$')
plt.ylabel(r'Spectrum [arbitrary units]')
plt.savefig('Only GPU.png', dpi = 300)

def em(x, fi, l, t):
    delay = t-x*1.5e8/2.6e10*(1+math.sin(inc)*np.cos(fi))
    L = 10+5*math.sin(2*math.pi/period*delay)
    r_c = x_c*np.sqrt(L)
    sigma_centrata = r_c/2
    first = 1/x*1/(math.sqrt(2*math.pi)*sigma_centrata)*math.exp(-(x-r_c)**2/(2*sigma_centrata**2))
    second = A/2*math.exp(-4*math.log(2)/d**2 *(fi-phi_0 - math.log10(x/xi_sp)/math.tan(p) - 2*math.pi*math.floor((fi-phi_0 - math.log10(x/xi_sp)/math.tan(p))/(2*math.pi)))**2)
    third = A/2*math.exp(-4*math.log(2)/d**2 *(2*math.pi- fi + phi_0 + math.log10(x/xi_sp)/np.tan(p) + 2*math.pi*math.floor((fi-phi_0 - math.log10(x/xi_sp)/math.tan(p) )/(2*math.pi)))**2)
    return first*(1+second+third)

def lum(x, fi, l, t):
    return 10. + 5.*np.sin(2*np.pi/period*(t-x*1.5e8/2.6e10*(1+math.sin(np.radians(30.))*np.cos(fi))))

#funzione specific intensity per CPU
def gaussian_cpu(x, fi, lam):
    velocity = -math.sqrt(1/x)*math.sin(fi)*math.sin(inc)
    lambda_obs = np.float32(6563)*(1+velocity)
    gaussiana = []
    for l in lam:
        gaussiana.append(1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(l-lambda_obs)**2/(2*sigma**2)))
    return gaussiana
"""
flusso= []
area_cpu = []
emissivity_cpu = []
gauss_cpu = []
luminosity_cpu = []

for l in lam:
    for x in xi:
        for fi in phi:
            area_cpu.append(np.pi/phi.size*((x+dr)**2-x**2))
            emissivity_cpu.append(em(x,fi,l, t))
            luminosity_cpu.append(lum(x,fi,l, t))
            
for x in xi:
        for fi in phi:
            gauss_cpu.append(gaussian_cpu(x, fi, lam))

area_cpu = np.array(area_cpu).reshape(size_lam, size_x, size_phi)
emissivity_cpu = np.array(emissivity_cpu).reshape(size_lam, size_x, size_phi)
luminosity_cpu = np.array(luminosity_cpu).reshape(size_lam, size_x, size_phi)
gauss_cpu = np.array(gauss_cpu).reshape(size_lam, size_x, size_phi)

diff = 0
max_diff = -1
for i in range(len(xi)):
    for j in range(len(phi)):
        if np.abs(area[0][i][j] - area_cpu[0][i][j]) > max_diff:
            max_diff = np.abs(area[0][i][j] - area_cpu[0][i][j])

print('Massima differenza tra due aree: %.f' % max_diff)

diff = 0
max_diff = -1
for i in range(len(xi)):
    for j in range(len(phi)):
        if np.abs(emissivity[0][i][j] - emissivity_cpu[0][i][j]) > max_diff:
            max_diff = np.abs(emissivity[0][i][j] - emissivity_cpu[0][i][j])

print('Massima differenza per l\'emissività: %.f' % (max_diff))

diff = 0
max_diff = -1
for i in range(len(xi)):
    for j in range(len(phi)):
        if np.abs(luminosity[0][i][j] - luminosity_cpu[0][i][j]) > max_diff:
            max_diff = np.abs(luminosity[0][i][j] - luminosity_cpu[0][i][j])
            
print('Massima differenza per la luminosità: %f' % (max_diff))

diff = 0
max_diff = -1
for i in range(len(xi)):
    for j in range(len(phi)):
        if np.abs(gaussian[0][i][j] - gauss_cpu[0][i][j]) > max_diff:
            max_diff = np.abs(gaussian[0][i][j] - gauss_cpu[0][i][j])
            
print('Massima differenza per la specific intensity: %.f' % (max_diff))

diff = 0
max_diff = -1
for i in range(len(xi)):
    for j in range(len(phi)):
        cpu_tot = area_cpu[0, i, j]*luminosity_cpu[0, i, j]*emissivity_cpu[0, i, j]*gauss_cpu[0, i, j]
        gpu_tot = area[0, i, j]*luminosity[0, i, j]*emissivity[0, i, j]*gauss[0, i, j]
        if np.abs(cpu_tot-gpu_tot) > max_diff:
             max_diff = cpu_tot-gpu_tot
            
print('Massima differenza per il prodotto: %.f' % (max_diff))
"""
start_time_cpu = tempo.time()
for t in time:
    flusso = []
    for raggio in xi:
        area = np.pi/size_phi*((raggio + dr)**2-raggio**2)  
        for angolo in phi:  
            r_c  = x_c*(lum(raggio, angolo, 0., t))**0.5
            sigma1 = r_c/2
            emissionvalue = em(raggio, angolo, 0, t)
            v_lungo_osservatore = -sqrt(1/raggio)*np.sin(angolo)*sin(inc)
            lambda_osservata = lambda_emessa*(1+v_lungo_osservatore)#*grav_red #NOTE: modify to add grav_red 
            continuum = lum(raggio, angolo, 0., t)
            gauss_ = np.array(gaussian_cpu(raggio, angolo, lam))
            flusso.append(area*emissionvalue*continuum*gauss_) #here I save the contribute for each wavelength of the specific section of the BLR

flusso = array(flusso) #from list to array to be able to sum
flusso_def = sum(flusso, axis = 0) #now, the shape is just 175

end_time_cpu = tempo.time()

plt.plot(lam, flusso_def, label = 'Flusso CPU')
plt.plot(lam, flux_def, linestyle = '--', label = 'Flusso GPU')
plt.legend()
plt.xlabel(r'Wavelength $[\AA]$')
plt.ylabel(r'Spectrum [arbitrary units]')
plt.savefig('Comparison CPU-GPU.png', dpi = 300)

print('Tempo necessario per GPU:', end_time-start_time)
print('Tempo necessario per CPU:', end_time_cpu-start_time_cpu)