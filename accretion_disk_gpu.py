import numba
from numba import cuda, float32
from numpy import sqrt, log10, log, radians, exp, sin, tan, linspace, pi, array, sum
import numpy as np
import math
from scipy.integrate import quad
import celerite
from celerite import terms

# 4. defining constants
h = 6.62607015e-34  # Planck's constant (Joule-seconds)
c = 3e8  # Speed of light (m/s)
k = 1.380649e-23  # Boltzmann constant (Joule/Kelvin)
G = 6.67e-11  # Gravitational constant (m^3/kg/s^2)
sigma_B = 1.28e-23  # Stefan-Boltzmann constant (W/m^2/K^4)
M_sun = 2e30  # Solar mass (kg)
M = 2e30*1e8  # Mass of the black hole (kg)
chi = 0.1  # Dimensionless spin parameter
inc = np.radians(30)


# Function to calculate the innermost stable circular orbit (ISCO) radius
# for a black hole with mass M and spin chi; returns 1 value in meters
@cuda.jit(device=True)
def R_ISCO(M, chi):
    Z1 = 1 + (1 - chi**2)**(1/3) * ((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z2 = np.sqrt(3 * chi**2 + Z1**2)
    isco_ = 3 + Z2 - np.sign(chi) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    return isco_ * G * M / c**2

# Function to calculate the Eddington luminosity 
# Returns 1 value in ? units
@cuda.jit(device=True)
def L_edd(M):
    return 4 * np.pi * G * M * c / 0.1

# Function to de-evolve the phase of an orbiting object
# Find where the emitting element of the accretion disc was at an arbitrary time
@cuda.jit(device = True)
def de_evolve(xi, theta, time, initial_phase, M):
    td = time_delay(xi, theta, inc)
    kf = keplerian_freq(M, xi) 
    return ((theta - (time -td)*kf + initial_phase)) % (2*math.pi)

#Function to compute time delay
# Modifies a pre-allocated CUDA device array (filled with zeros) with the de-evolved angles
@cuda.jit(device=True)
def time_delay(xi, theta, inc):
    return xi*(1+math.sin(inc)*math.cos(theta))

#x, y are separated cuda device 
#can be read by single array in CUDA kernel
@cuda.jit(device = True)
def position_components(angle, R):
    x = R * math.sin(angle)  # x-coordinate
    y = R * math.cos(angle)  # y-coordinate
    return (x,y)

#vx, vy are separated cuda device
#can be read by single array in CUDA kernel
@cuda.jit(device = True)
def velocity_components(angle, R, M, corot=0):
    v_kep = math.sqrt(G * M / R)  # Keplerian velocity
    vx = v_kep * math.cos(angle)  # x-component of velocity
    vy = -v_kep * math.sin(angle) # y-component of velocity

    if corot == 0:  # Corotating case
        return (vx, vy)
    else:  # Non-corotating case
        return (-vx, -vy)

#function to compute keplerian frequency
@cuda.jit(device = True)
def keplerian_freq(M, R):
    omega_k = math.sqrt(G * M / R**3)
    return omega_k

#substitute for np.dot()
@cuda.jit(device=True)
def dot_product(a, b):
    result = 0.0
    for i in range(a.size):
        result += a[i] * b[i]
    return result

#substitute for np.linalg.norm()
#not used for now
@cuda.jit(device=True)
def vector_norm(v):
    norm = 0.0
    for i in range(v.size):
        norm += v[i] * v[i]
    return math.sqrt(norm)

#function to compute doppler factor
#need dot_product to work
@cuda.jit(device=True)
def doppler_factor(v1, v2, r1, r2):
    #norms/dots v1, v2
    norm_v1_sq = dot_product(v1, v1)
    norm_v2_sq = dot_product(v2, v2) 
    v2_dot_v1 = dot_product(v2, v1) 
    norm_v1_minus_v2_sq = dot_product(v1-v2, v1-v2) 
    
    # Parallel components of v2
    v2_para_x = (v2_dot_v1 / norm_v1_sq) * v1[0]
    v2_para_y = (v2_dot_v1 / norm_v1_sq) * v1[1]

    # Perpendicular components of v2
    v2_perp_x = v2[0] - v2_para_x
    v2_perp_y = v2[1] - v2_para_y

    # Lorentz factor v2
    gamma_2 = 1.0 / math.sqrt(1.0 - norm_v2_sq / (c**2))

    # beta vector v2
    beta_x = (1.0/c)*(v1[0] - v2[0] + (1.0/gamma_2)*v2_perp_x) / (1.0 - (v2_dot_v1)/(c**2))
    beta_y = (1.0/c)*(v1[1] - v2[1] + (1.0/gamma_2)*v2_perp_y) / (1.0 - (v2_dot_v1)/(c**2))
    beta_norm_sq = beta_x*beta_x + beta_y*beta_y #without dot_product because not array
 
    # np.dot(r2-r1)/(norm(r2-r1)**2)
    dr_x = r2[0] - r1[0]
    dr_y = r2[1] - r1[1]
    norm_dr = math.sqrt(dr_x*dr_x + dr_y*dr_y) #without dot_product because not array
    r_vers_x = dr_x / norm_dr
    r_vers_y = dr_y / norm_dr

    #Lorentz factor v1-v2
    gamma = math.sqrt(1.0 / (1.0 - beta_norm_sq))

    # Beta dot r_vers
    beta_dot_r_vers = beta_x*r_vers_x + beta_y*r_vers_y

    # Doppler factor
    D = 1.0 / (gamma * (1.0 - beta_dot_r_vers))

    return D