import numba
from numba import cuda, float32
from numpy import sqrt, log10, log, radians, exp, sin, tan, linspace, pi, array, sum
import numpy as np
import math

# AREA KERNEL
# da 3 vettori cuda per raggi, angoli e lunghezze d'onda, calcola il valore dell'area della singola sezione
# nota: la lunghezza d'onda viene inclusa solamente per ottimizzare la somma/il prodotto dei valori contributi
# d_area è il cuda device vuoto che viene riempito con i valori dell'area
@cuda.jit
def area_kernel(x, fi, d_lam, d_area):
    idx_lambda, idx_x, idx_fi = cuda.grid(3)

    dr = x[1]- x[0]
    if idx_lambda < d_lam.size and idx_x < x.size and idx_fi < fi.size:
        d_area[idx_lambda][idx_x][idx_fi] = math.pi/fi.size*((x[idx_x]+dr)**2 - x[idx_x]**2)


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
    idx_lambda, idx_x, idx_fi = cuda.grid(3)

    if idx_lambda < d_lam.size and idx_x < x.size and idx_fi < fi.size:        
        r_c = r_c_kernel(x[idx_x], fi[idx_fi], x_c, t, inc, period)
        sigma_centrata = r_c/2
        first = 1/x[idx_x]*1/(math.sqrt(2*math.pi)*sigma_centrata)*math.exp(-(x[idx_x]-r_c)**2/(2*sigma_centrata**2))
        second = A/2*math.exp(-4*math.log(2)/d**2 *(fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p) - 2*math.pi*math.floor((fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p))/(2*math.pi)))**2)
        third = A/2*math.exp(-4*math.log(2)/d**2 *(2*math.pi- fi[idx_fi] + phi_0 + math.log10(x[idx_x]/xi_sp)/np.tan(p) + 2*math.pi*math.floor((fi[idx_fi]-phi_0 - math.log10(x[idx_x]/xi_sp)/math.tan(p) )/(2*math.pi)))**2)
        d_emissivity[idx_lambda][idx_x][idx_fi] = first*(1+second+third)

#LUMINOSITY KERNEL: need to be substituted with the accretion_disk.py results
@cuda.jit
def luminosity_kernel(x, fi, d_lam, t, inc, period, d_luminosity):
    idx_lambda, idx_x, idx_fi = cuda.grid(3)

    if idx_lambda < d_lam.size and idx_x < x.size and idx_fi < fi.size:
        d_luminosity[idx_lambda][idx_x][idx_fi] = 10. + 5.*math.sin(2*math.pi/period*(t-x[idx_x]*1.5e8/2.6e10*(1+math.sin(inc)*np.cos(fi[idx_fi]))))

#da 3 vettori cuda per raggi, angoli e lunghezze d'onda, calcola il valore 
#della specific intensity della singola sezione
#tra i parametri da dare in pasto d_gaussian è il cuda device vuoto che viene riempito con i valori 
#della specific intensity
@cuda.jit
def gaussian_kernel(x, fi, d_lam, d_gaussian, inc, lambda_emessa, sigma):
    idx_lambda, idx_x, idx_fi = cuda.grid(3)

    if idx_lambda < d_lam.size and idx_x < x.size and idx_fi < fi.size:
        velocity = -math.sqrt(1/x[idx_x])*math.sin(fi[idx_fi])*math.sin(inc)
        lambda_obs = np.float32(lambda_emessa)*(1+velocity)
        d_gaussian[idx_lambda][idx_x][idx_fi] = 1/(math.sqrt(2*math.pi)*sigma)*math.exp(-(d_lam[idx_lambda]-lambda_obs)**2/(2*sigma**2))

#PRODUCT KERNEL
#calcola il prodotto dei risultati dei vari kernel.
@cuda.jit
def product_kernel(x, fi, d_lam, d_area, d_luminosity, d_emissivity, d_gaussian, d_flux):
    idx_lambda, idx_x, idx_fi = cuda.grid(3)
            
    if idx_lambda < d_lam.size and idx_x < x.size and idx_fi < fi.size:
        d_flux[idx_lambda][idx_x][idx_fi] = d_area[idx_lambda][idx_x][idx_fi]*d_luminosity[idx_lambda][idx_x][idx_fi]*d_emissivity[idx_lambda][idx_x][idx_fi]*d_gaussian[idx_lambda][idx_x][idx_fi]

#SUM KERNEL
#permette di ottenere il flusso sommando i contributi sugli assi di raggio e angolo
@cuda.jit
def sum_kernel(x, fi, d_lam, d_flux, d_flux_def):
    idx_lambda = cuda.grid(1)

    if idx_lambda < d_lam.size: 
        for i_x in range(x.size):
            for i_fi in range(fi.size):
                d_flux_def[idx_lambda] = d_flux_def[idx_lambda] + d_flux[idx_lambda][i_x][i_fi]

if __name__ == "__main__":
    c = 3e5 #velocità della luce in km/s
    lambda_emessa = 6563 #lunghezza d'onda emessa in angstrom
    inc = np.radians(30.) #inclinazione del disco
    """period = 120. #periodo luminosità in giorni
    x_c = 1200. #raggio centrale del disco in unità di raggi gravitazionali
    A = 50. #contrasto emissività gaussiana-spirale
    p = np.radians(30.) #quanto gira la spirale su se stessa
    d = 0.5 #larghezza della spirale
    xi_sp = 0.1 #raggio iniziale della spirale
    phi_0 = 0. #offset angolare della spirale
    sigma = lambda_emessa*1200.0/c #deviazione standard della gaussiana
    """
