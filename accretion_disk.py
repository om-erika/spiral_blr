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

# Function to calculate the radiative efficiency of a black hole
# Returns 1 adimensionless value
@cuda.jit(device = True)
def radiative_efficiency(chi):
    Z1 = 1 + (1 - chi**2)**(1/3) * ((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z2 = np.sqrt(3 * chi**2 + Z1**2)
    Eisco = (4 - chi * Z1 - np.sqrt(3 * Z2 - 2 * Z1)) / (3 * np.sqrt(3))
    eta = 1 - Eisco
    return eta

# Function to calculate the temperature profile of an accretion disk
# Returns CUDA device array of temperatures for the given radii
@cuda.jit(device=True)
def Temperature(R, M, f_edd, chi, R_in):
    eta = radiative_efficiency(chi)
    M_dot = f_edd * L_edd(M) / (eta * c**2)  # Mass accretion rate
    return ((3 * G * M * M_dot) / (8 * np.pi * sigma_B * R**3))**(1/4) * (1 - np.sqrt(R_in / R))**(1/4)

# Function to calculate the Planck spectrum for a given temperature
# Returns spectrum value for the given frequency and temperature
@cuda.jit(device = True)
def planck_spectrum(nu, T):
    return (2 * h * nu**3) / (c**2) * (1 / (np.exp(h * nu / (k * T)) - 1))

# Function to calculate the ionizing flux for a given temperature
# The temperature here should be D*T/(1+z) where D is the Doppler factor, 1+z is the gravitational redshift and T is the rest frame wavelength 
# Returns the ionizing flux for the given radius and dtheta (with dependence on T, D also)
@cuda.jit(device = True)
def ionizing_flux_element(T,D,R,dr,dtheta):
    def integrand(nu):
        return planck_spectrum(nu, T)
    flux, _ = quad(integrand, 13.6 / h * 1.60218e-19, 20 * 13.6 / h * 1.60218e-19)
    return flux*R*dr*dtheta

# Function to calculate the Doppler factor for relativistic motion
# Returns the Doppler factor for the given angles and positions
@cuda.jit(device = True)
def Doppler_factor(v1, v2, r1, r2):
    """
    Calculate the Doppler factor for a moving source and observer.
    Parameters:
    v1 (numpy.ndarray): Velocity vector of the observer (in units of m/s).
    v2 (numpy.ndarray): Velocity vector of the source (in units of m/s).
    r1 (numpy.ndarray): Position vector of the observer (in units of m).
    r2 (numpy.ndarray): Position vector of the source (in units of m).
    Returns:
    float: The Doppler factor, which accounts for the relativistic effects 
           of motion on the observed frequency of a signal.
    """

    v2_para = np.dot(v2, v1) * v1 / np.linalg.norm(v1)**2  # Parallel component of v2
    v2_perp = v2 - v2_para  # Perpendicular component of v2
    gamma_2 = 1 / np.sqrt(1 - np.linalg.norm(v2)**2 / c**2)  # Lorentz factor for v2
    beta = 1 / c * (v1 - v2 + 1 / gamma_2 * v2_perp) / (1 - np.dot(v1, v2) / c**2)
    r_vers = (r2 - r1) / np.linalg.norm(r2 - r1)  # Unit vector in the direction of r2 - r1
    gamma = np.sqrt(1 / (1 - beta**2))  # Lorentz factor for combined motion
    D = 1 / (gamma * (1 - np.dot(beta, r_vers)))  # Doppler factor
    return D

# Function to calculate the Keplerian frequency at a given radius
# Returns the Keplerian frequency for the given radius (no theta dependence)
@cuda.jit(device = True)
def keplerian_freq(M, R):
    omega_k = np.sqrt(G * M / R**3)
    return omega_k

@cuda.jit(device = True)
def gravitational_redshift(M,R):
    return 1/(1-(2*G*M/(c**2*R)))
### Assuming the vertical axis to be the y-axis and the horizontal to be the x-axis

# Function to calculate velocity components in a Keplerian orbit
# angle should be the de-evolved theta to take into account the time-delay for a given BLR element
# Can only run in CUDA device
@cuda.jit(device = True)
def velocity_components(angle, R, M, corot=0):
    v_kep = np.sqrt(G * M / R)  # Keplerian velocity
    vx = v_kep * np.cos(angle)  # x-component of velocity
    vy = -v_kep * np.sin(angle) # y-component of velocity

    if corot == 0:  # Corotating case
        return np.array([vx, vy])
    else:  # Non-corotating case
        return np.array([-vx, -vy])

# Function to calculate position components in a Keplerian orbit
# Can only run in CUDA device
@cuda.jit(device = True)
def position_components(angle, R):
    x = R * np.sin(angle)  # x-coordinate
    y = R * np.cos(angle)  # y-coordinate
    return np.array([x, y])


# Function to generate a set of temperatures using Gaussian processes
def generate_set_of_temperatures(times, Rs, thetas, M, f_edd, chi, R_in, tau, sigma):
    """
    Generate a set of Gaussian Process (GP) objects and initial temperature samples 
    for a grid of radii (Rs) and angles (thetas) over specified time points.

    Args:
        times (array-like): Array of time points at which the GP is computed.
        Rs (array-like): Array of radii values representing the grid's columns.
        thetas (array-like): Array of angular values (in radians) representing the grid's rows.
        M (float): Mass parameter used in the temperature calculation.
        f_edd (float): Eddington factor used in the temperature calculation.
        chi (float): Opacity parameter used in the temperature calculation.
        R_in (float): Inner radius parameter used in the temperature calculation.
        tau (float): Timescale parameter for the GP kernel.
        sigma (float): Amplitude parameter for the GP kernel.

    Returns:
        tuple: A tuple containing:
            - gps (list of lists): A 2D list of GP objects, where rows correspond to 
              angular values (thetas) and columns correspond to radii (Rs).
            - starting_ys (list of lists): A 2D list of initial temperature samples 
              generated from the GPs, with the same structure as `gps`.

    Notes:
        - The grid is structured such that the outer list corresponds to rows (thetas),
          and the inner list corresponds to columns (Rs).
        - Each GP object is initialized with a kernel defined by the given `tau` and `sigma`
          parameters and a mean temperature based on the radius `Rs[i]`.
    """
    # Initialize Gaussian Process (GP) objects and starting temperature samples
    gps = [[[] for _ in range(len(Rs))] for _ in range(len(thetas))]
    starting_ys = [[[] for _ in range(len(Rs))] for _ in range(len(thetas))]
    for i in range(len(Rs)):
        for j in range(len(thetas)):
            # Define a GP kernel with specified parameters
            kernel_ = terms.RealTerm(log_a=np.log(sigma), log_c=-np.log(tau))
            # Create a GP object with the kernel and mean temperature at the given radius
            # Here the radius is the mean radius of the element
            gp = celerite.GP(kernel_, mean=np.log10(Temperature(Rs[i]+(Rs[1]-Rs[0])/2, M, f_edd, chi, R_in)))
            gp.compute(times)  # Precompute the GP for the given time points
            starting_ys[i][j] = gp.sample()  # Sample initial temperatures
            gps[i][j] = gp  # Store the GP object
    return gps, 10**starting_ys