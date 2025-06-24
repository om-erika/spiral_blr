import numpy as np
import numba
from numba import cuda, float32, njit
import math
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import time
from numpy import sqrt, log10, log, radians, exp, sin, tan, linspace, pi, array, sum
from scipy.stats import uniform
import time as tempo

# AREA KERNEL
# da 3 vettori cuda per raggi, angoli e lunghezze d'onda, calcola il valore dell'area della singola sezione
# nota: la lunghezza d'onda viene inclusa solamente per ottimizzare la somma/il prodotto dei valori contributi
# d_area è il cuda device vuoto che viene riempito con i valori dell'area
@cuda.jit
def area_kernel(x, fi, d_lam, d_time, d_area):
    idx_time, idx_x, idx_fi = cuda.grid(3)

    dr = x[1]- x[0]
    if idx_time < d_time.size and idx_x < x.size and idx_fi < fi.size:
        for idx_lambda in range(d_lam.size):
            d_area[idx_lambda][idx_x][idx_fi][idx_time] = math.pi/fi.size*((x[idx_x]+dr)**2 - x[idx_x]**2)

            
#R_C KERNEL (DEVICE = TRUE)
#kernel che restituisce un CUDA device. Utilizzato per calcolare r_c nell'emissività
@cuda.jit(device = True)
def r_c_kernel(x, fi, x_c, t, inc, period):
    delay = t-x*1.5e8/2.6e10*(1+math.sin(inc)*math.cos(fi))
    L = 10+5*math.sin(2*math.pi/period*delay)
    return x_c*(L**0.5)


#EMISSIVITY KERNEL
#da 3 vettori cuda per raggi, angoli e lunghezze d'onda, calcola il valore dell'emissività della singola sezione
#i parametri A, phi_0, delta p e xi_sp per ora li considero fissati, ma possono essere aggiunti
#tra i parametri da dare in pasto d_emissivity è il cuda device vuoto che viene riempito con i valori dell'emissività
@cuda.jit
def emissivity_kernel(x, fi, t, d_lam, d_emissivity,A, phi_0, d, p, xi_sp, x_c, inc, period):
    idx_time, idx_x, idx_fi = cuda.grid(3)

    if idx_time < t.size and idx_x < x.size and idx_fi < fi.size: 
        for idx_lambda in range(d_lam.size):
            r_c = r_c_kernel(x[idx_x], fi[idx_fi], x_c, t[idx_time], inc, period)
            sigma_centrata = r_c/2
            first = 1/x[idx_x]*1/(math.sqrt(2*math.pi)*sigma_centrata)*math.exp(-(x[idx_x]-r_c)**2/(2*sigma_centrata**2))
            second = A/2*math.exp(-4*math.log(2)/d**2 *(fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p) - 2*math.pi*math.floor((fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p))/(2*math.pi)))**2)
            third = A/2*math.exp(-4*math.log(2)/d**2 *(2*math.pi- fi[idx_fi] + phi_0 + math.log10(x[idx_x]/xi_sp)/math.tan(p) + 2*math.pi*math.floor((fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p) )/(2*math.pi)))**2)
            d_emissivity[idx_lambda][idx_x][idx_fi][idx_time] = first*(1+second+third)

#LUMINOSITY KERNEL: need to be substituted with the accretion_disk.py results
@cuda.jit
def luminosity_kernel(x, fi, d_lam, t, inc, period, d_luminosity):
    idx_time, idx_x, idx_fi = cuda.grid(3)

    if idx_time < t.size and idx_x < x.size and idx_fi < fi.size:
        for idx_lambda in range(d_lam.size):
            d_luminosity[idx_lambda][idx_x][idx_fi][idx_time] = 10. + 5.*math.sin(2*math.pi/period*(t[idx_time]-x[idx_x]*1.5e8/2.6e10*(1+math.sin(inc)*math.cos(fi[idx_fi]))))

#da 3 vettori cuda per raggi, angoli e lunghezze d'onda, calcola il valore 
#della specific intensity della singola sezione
#tra i parametri da dare in pasto d_gaussian è il cuda device vuoto che viene riempito con i valori 
#della specific intensity
@cuda.jit
def gaussian_kernel(x, fi, d_lam, t, d_gaussian, inc, lambda_emessa, sigma):
    idx_time, idx_x, idx_fi = cuda.grid(3)

    if idx_time < t.size and idx_x < x.size and idx_fi < fi.size:
        for idx_lambda in range(d_lam.size):
            velocity = -math.sqrt(1/x[idx_x])*math.sin(fi[idx_fi])*math.sin(inc)
            lambda_obs = np.float64(lambda_emessa)*(1+velocity)
            d_gaussian[idx_lambda][idx_x][idx_fi][idx_time] = 1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(d_lam[idx_lambda]-lambda_obs)**2/(2*sigma**2))

#PRODUCT KERNEL
#calcola il prodotto dei risultati dei vari kernel.
@cuda.jit
def product_kernel(x, fi, d_lam, d_time, d_area, d_luminosity, d_emissivity, d_gaussian, d_flux):
    idx_time, idx_x, idx_fi = cuda.grid(3)
            
    if idx_time < d_time.size and idx_x < x.size and idx_fi < fi.size:
        for idx_lambda in range(d_lam.size):
            d_flux[idx_lambda][idx_x][idx_fi][idx_time] = d_area[idx_lambda][idx_x][idx_fi][idx_time]*d_luminosity[idx_lambda][idx_x][idx_fi][idx_time]*d_emissivity[idx_lambda][idx_x][idx_fi][idx_time]*d_gaussian[idx_lambda][idx_x][idx_fi][idx_time]


#SUM KERNEL
#permette di ottenere il flusso sommando i contributi sugli assi di raggio e angolo
@cuda.jit
def sum_kernel(x, fi, d_lam, d_time, d_flux, d_flux_def):
    idx_lambda, idx_time = cuda.grid(2)

    if idx_lambda < d_lam.size and idx_time < d_time.size: 
        for i_x in range(x.size):
            for i_fi in range(fi.size):
                d_flux_def[idx_lambda][idx_time] = d_flux_def[idx_lambda][idx_time] + d_flux[idx_lambda][i_x][i_fi][idx_time]

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
    lambda_obs = np.float64(6563.)*(1+velocity)
    gaussiana = []
    for l in lam:
        gaussiana.append(1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(l-lambda_obs)**2/(2*sigma**2)))
    return gaussiana

if __name__ == "__main__":     
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
    size_x = 50
    size_phi = 50
    size_time = 50
    size_lam = 50

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
    area = np.zeros([size_lam, size_x,size_phi, size_time])
    emissivity = np.zeros([size_lam, size_x,size_phi, size_time])
    luminosity =np.zeros([size_lam, size_x,size_phi, size_time])
    gauss = np.zeros([size_lam, size_x,size_phi, size_time])
    flux = np.zeros([size_lam, size_x,size_phi, size_time])
    flux_def = np.zeros([size_lam, size_time])
        
    #2. Copio i device sulla memoria della GPU
    d_xi = cuda.to_device(xi)
    d_phi = cuda.to_device(phi)
    d_time = cuda.to_device(time)
    d_lam = cuda.to_device(lam)

    #threads_per_block/blocks_per_grid per kernel area, emissività, luminosità e specific intensity, kernel prodotto
    threads_per_block =  (4,4,4)
    blocks_per_grid = (
        (d_xi.size + (threads_per_block[1] - 1)) // threads_per_block[1],
        (d_phi.size + (threads_per_block[2] - 1)) // threads_per_block[2],
        (d_time.size + (threads_per_block[0] - 1)) // threads_per_block[0],
        
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

    area_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_time, d_area)
    cuda.synchronize()
    area = d_area.copy_to_host()
    #emissività
    emissivity_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_time, d_lam, d_emissivity, A, phi_0, d, p, xi_sp, x_c, inc, period)
    cuda.synchronize()
    emissivity = d_emissivity.copy_to_host()
    #luminosità
    luminosity_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_time, inc, period, d_luminosity)
    cuda.synchronize()
    luminosity = d_luminosity.copy_to_host()
    #gaussian
    gaussian_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_time, d_gaussian, inc, lambda_emessa, sigma)
    cuda.synchronize()
    gaussian = d_gaussian.copy_to_host()
    #prodotto tra i vari contributi (area, emissività, luminosità e gaussiana)
    product_kernel[blocks_per_grid, threads_per_block](d_xi, d_phi, d_lam, d_time, d_area, d_luminosity, d_emissivity, d_gaussian, d_flux)
    cuda.synchronize()
    flux = d_flux.copy_to_host()
    #somma dei vari contributi, di fatto il flusso
    cuda.synchronize()
    sum_kernel[blocks_per_grid_1, threads_per_block_1](d_xi, d_phi, d_lam, d_time, d_flux, d_flux_def)
    cuda.synchronize()

    flux_def = d_flux_def.copy_to_host()

    end_time = tempo.time()

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
    plt.plot(lam, flux_def[:, -1], linestyle = '--', label = 'Flusso GPU')
    plt.legend()
    plt.xlabel(r'Wavelength $[\AA]$')
    plt.ylabel(r'Spectrum [arbitrary units]')
    plt.savefig('Comparison CPU-GPU.png', dpi = 300)

    print('Tempo necessario per GPU:', end_time-start_time)
    print('Tempo necessario per CPU:', end_time_cpu-start_time_cpu)