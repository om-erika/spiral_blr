import numba
from numba import cuda, float64
import numpy as np
import math
import celerite
from celerite import terms
import time as tempo

# Function to calculate the radius of the innermost stable circular orbit (ISCO)
def R_ISCO(M, chi):
    """
    Compute radius of the innermost stable circular orbit (ISCO) for a given mass M and chi parameter.

    Args:
        M (float): Mass of the black hole.
        chi (float): Chi parameter used in the ISCO calculation.

    Returns:
        float: Radius of the ISCO in units of G*M/c^2.
    """
    Z1 = 1. + (1. - chi**2)**(1/3) * ((1. + chi)**(1/3) + (1. - chi)**(1/3))
    Z2 = np.sqrt(3. * chi**2 + Z1**2)
    isco_ = 3 + Z2 - np.sign(chi) * np.sqrt((3. - Z1) * (3. + Z1 + 2. * Z2))
    return isco_ * G * M / c**2


@cuda.jit(device = True) 
def de_evolve(xi, theta, time, initial_phase, M, inc):
    """
    Compute the de-evolved angle for a given radius xi, angle theta, and time in the accretion disc.
    This function calculates the angle at which an orbiting object would be in order for its
    ionising flux to hit the BLR element at a given time,
    taking into account the initial phase and the mass parameter M.

    Args:
        xi (float): Radius at which to compute the de-evolved angle.
        theta (float): Angle in radians at which to compute the de-evolved angle.
        time (float): Time at which to compute the de-evolved angle.
        initial_phase (float): Initial phase of the orbiting object in radians.
        M (float): Mass parameter used in the de-evolution calculation.
        inc (float): # Inclination angle of the accretion disc in radians.

    Returns:
        float: Computed de-evolved angle for the given radius, angle, time, initial phase, and mass.
    """
    td = time_delay(xi, theta, inc)
    kf = keplerian_freq(M, xi) 
    return (theta - (time -td)*kf + initial_phase) - (2.*math.pi)*math.floor((theta - (time -td)*kf + initial_phase)/(2.*math.pi))

#Function to compute time delay
# Modifies a pre-allocated CUDA device array (filled with zeros) with the de-evolved angles
# @cuda.jit(device=True) would be more efficient if we don't need to save them in a device array
@cuda.jit(device=True)
def time_delay(xi, theta, inc):
    """
    Compute time delay for a given radius xi and angle theta in the accretion disc.
    This function calculates the time delay based on the radius, angle, and inclination
    of the accretion disc.

    Args:
        xi (float): Radius at which to compute the time delay.
        theta (float): Angle in radians at which to compute the time delay.
        inc (float): Inclination angle of the accretion disc in radians.

    Returns:
        float: Computed time delay for the given radius, angle, and inclination.
    """
    return xi*(1+math.sin(inc)*math.cos(theta))

#x, y are separated cuda device
@cuda.jit(device = True)
def position_components(angle, R):
    """
    Compute positions components for a given angle and radius R in the accretion disc.

    Args:
        angle (float): Angle in radians at which to compute the velocity components.
        R (float): Radius at which to compute the velocity components.
        
    Returns:
        float: Computed positions components (x, y) at the given angle and radius R.
    """
    x = R * math.sin(angle)  # x-coordinate
    y = R * math.cos(angle)  # y-coordinate
    return (x,y)

#vx, vy are separated cuda device
@cuda.jit(device = True)
def velocity_components(angle, R, M, corot=0):
    """
    Compute velocity components for a given angle and radius R in the accretion disc.

    Args:
        angle (float): Angle in radians at which to compute the velocity components.
        R (float): Radius at which to compute the velocity components.
        M (float): Mass parameter used in the velocity calculation.
        corot (int, optional): Flag indicating whether to compute corotating (0) or non-corotating (1) velocities. Defaults to 0.
        
    Returns:
        float: Computed velocity components (vx, vy) at the given angle and radius R.
    """
    v_kep = math.sqrt(G * M / R)  # Keplerian velocity
    vx = v_kep * math.cos(angle)  # x-component of velocity
    vy = -v_kep * math.sin(angle) # y-component of velocity

    if corot == 0:  # Corotating case
        return (vx, vy)
    else:  # Non-corotating case
        return (-vx, -vy)

@cuda.jit(device = True)
def keplerian_freq(M, R):
    """
    Compute keplerian frequency for a given mass M and radius R.

    Args:
        M (float): Mass parameter used in the frequency calculation.
        R (float): Radius parameter used in the frequency calculation.
        
    Returns:
        float: Computed keplerian frequency for the given mass and radius.
    """
    omega_k = math.sqrt(G * M/ R**3)
    return omega_k

@cuda.jit(device=True)
def doppler_factor(v1, v2, r1, r2):
    """
    Compute the Doppler factor for two velocity vectors v1 and v2,
    given the positions r1 in the accretion disk and r2 in the BLR disc.
    This function calculates the Doppler factor based on the velocities and positions
    of the emitting element and the observer, taking into account relativistic effects.
    The Doppler factor is computed using the formula:
    D = 1 / (gamma * (1 - beta_dot_r_vers))

    where:
        gamma = Lorentz factor for the relative velocity,
        beta = velocity vector normalized by the speed of light,
        r_vers = unit vector in the direction of the observer's position.


    Args:
        v1 (tuple): Velocity vector of the emitting element in the accretion disk (vx, vy).
        v2 (tuple): Velocity vector of the observer in the BLR disc (vx, vy).
        r1 (tuple): Position of the emitting element in the accretion disk (x, y).
        r2 (tuple): Position of the observer in the BLR disc (x, y).
        
    Returns:
        float: Computed Doppler factor D, which accounts for the relativistic effects
        of the relative motion between the emitting element and the observer.
    """
    #norms/dots v1, v2
    norm_v1_sq = v1[0]*v1[0] + v1[1]*v1[1]#dot_product(v1, v1)#v1[0]*v1[0] + v1[1]*v1[1]
    norm_v2_sq = v2[0]*v2[0] + v2[1]*v2[1]#dot_product(v2, v2) #v2[0]*v2[0] + v2[1]*v2[1]
    v2_dot_v1 = v2[0]*v1[0] + v2[1]*v1[1]#dot_product(v2, v1) #v2[0]*v1[0] + v2[1]*v1[1]
    norm_v1_minus_v2_sq = (v1[0]-v2[0])*(v1[0]-v2[0]) + (v1[1]-v2[1])*(v1[1]-v2[1])#dot_product(v1-v2, v1-v2) #(v1[0]-v2[0])*(v1[0]-v2[0]) + (v1[1]-v2[1])*(v1[1]-v2[1])
    
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

@cuda.jit(device=True) #not used since we explicate dot product in doppler_factor
def dot_product(a, b):
    """
    Compute dot product of two vectors a and b.

    Args:
        a (array-like): First vector.
        b (array-like): Second vector.
        
    Returns:
        float: Computed dot product of the two vectors.
    """
    result = 0.0
    for i in range(a.size):
        result += a[i] * b[i]
    return result

@cuda.jit(device=True)
def radiative_efficiency(chi):
    """
    Compute temperature for a given radius R using the parameters of the accretion disc.

    Args:
        chi (float): Opacity parameter used in the temperature calculation.
        
    Returns:
        float: Computed radiative efficiency for the given chi value.
    """
    Z1 = 1. + (1. - chi**2)**(1/3) * ((1. + chi)**(1/3) + (1. - chi)**(1/3))
    Z2 = math.sqrt(3. * chi**2 + Z1**2)
    Eisco = (4. - chi * Z1 - math.sqrt(3 * Z2 - 2. * Z1)) / (3. * math.sqrt(3))
    eta = 1.- Eisco
    return eta

@cuda.jit(device=True)
def L_edd(M):
    """
    Compute Eddington luminosity for a given mass M.

    Args:
        M (float): Mass parameter used in the temperature calculation.

    Returns:
        float: Computed L_edd for BH with mass M.
    """
    return 4.*math.pi*G*M*c/0.1
    
@cuda.jit(device=True)
def temperature(pos, M, f_edd, chi, R_in):
    """
    Compute temperature for a given radius R using the parameters of the accretion disc.

    Args:
        pos (tuple): Position in the accretion disc as a tuple (x, y).
        M (float): Mass parameter used in the temperature calculation.
        f_edd (float): Eddington factor used in the temperature calculation.
        chi (float): Opacity parameter used in the temperature calculation.
        R_in (float): Inner radius parameter used in the temperature calculation.
        
    Returns:
        float: Computed temperature at the position `pos`.
    """
    R = math.sqrt(pos[0]**2+pos[1]**2)
    eta = radiative_efficiency(chi)
    M_dot = f_edd*L_edd(M)/(eta*c**2)
    return ((3.*G*M*M_dot)/(8.*math.pi*sigma_B*R**3))**(1/4)*(1.-math.sqrt(R_in/R))**(1/4)  

@cuda.jit(device=True)
def compute_planck_spectrum(nu, T):
    """
    Compute the Planck spectrum for a given frequency `nu` and temperature `T`.
    This function uses the Planck law to calculate the spectral radiance of a black body
    at a specific frequency and temperature.

    Args:
        nu (float): Frequency at which to compute the Planck spectrum (in Hz).
        T (float): Temperature of the black body (in Kelvin).
        
    Returns:
        float: The spectral radiance at frequency `nu` and temperature `T`.
    """
    num = 2.*h*nu**3/c**2
    den = math.exp(h*nu/(k*T)) - 1.0
    return num /den

@cuda.jit(device=True)
def compute_ionizing_flux(T, D, z, pos, dr, dtheta):
    """
    Compute the ionizing flux for a given temperature, Doppler factor, redshift,
    and position in the accretion disc.
    This function integrates the Planck spectrum over a specified frequency range
    to compute the total flux emitted by the accretion disc element at the given position.
    The integration is performed over the frequency range corresponding to ionizing radiation
    (from 13.6 eV to 20 times the ionization energy of hydrogen).
    The result is scaled by the position in the disc and the differential area element
    defined by the radius and angle increments (dr, dtheta).
    The temperature is adjusted for the Doppler factor and redshift to account for relativistic effects.
    The function uses the Planck spectrum formula to compute the flux contribution
    for each frequency step in the integration.
    The integration is performed using a simple rectangular method with a fixed number of steps.

    Args:
        T (float): Temperature of the accretion disc element.
        D (float): Doppler factor accounting for relativistic effects.
        z (float): Redshift of the system.
        pos (tuple): Position in the accretion disc as a tuple (x, y).
        dr (float): Differential radius element for integration.
        dtheta (float): Differential angle element for integration.
        
    Returns:
        float: Total ionizing flux emitted by the accretion disc element.
    """

    nu_min = 13.6*1.60218e-19/h
    nu_max = 20.*13.6*1.60218e-19/h
    N = int(1e5)  # Number of integration steps
    dnu = (nu_max-nu_min)/N
    flux = 0.0
    
    T_eff = D*T/(1. + z)

    for i in range(N):
        nu = nu_min + i * dnu
        flux += compute_planck_spectrum(nu, T_eff)*dnu

    return flux*math.sqrt(pos[0]**2+pos[1]**2)*dr*dtheta

@cuda.jit
def sum_kernel(d_r, d_angle, d_time, d_flux, d_flux_def):
    """
    Sum contributes from all angles and radii to compute the total flux for each time step.
    This function iterates over the radii and angles, summing the flux contributions
    for each time step and storing the result in `d_flux_def`.
    The flux contributions are assumed to be stored in a 3D array `d_flux` with dimensions
    [time, angle, radius].
    The result is a 1D array `d_flux_def` where each element corresponds to the total flux
    at a specific time step, summing contributions from all angles and radii.

    Args:
        d_r (cuda.device_array): Device array containing radii values.
        d_angle (cuda.device_array): Device array containing angle values.
        d_time (cuda.device_array): Device array containing time values.
        d_flux (cuda.device_array): 3D device array containing flux contributions
                                     with dimensions [time, angle, radius].
        d_flux_def (cuda.device_array): 1D device array to store the summed flux
                                         contributions for each time step.
        
    Returns:
        None: The function modifies `d_flux_def` in place, summing the contributions
              from all angles and radii for each time step.
    """

    idx_t = cuda.grid(1)
    if idx_t < d_time.size:
        for i_x in range(d_r.size):
            for i_fi in range(d_angle.size):
                d_flux_def[idx_t] = d_flux_def[idx_t] + d_flux[idx_t][i_fi][i_x]

@cuda.jit(device=True)
def find_temperature(gp_times, starting_light_curve, t):#, idx_angle, idx_R):
    """
    Interpolate temperature at a given time `t` using the Gaussian Process (GP) light curve data.
    This function finds the nearest time points in `gp_times` to `t` and computes the average
    temperature from the corresponding light curve values in `starting_light_curve`.

    Args:
        gp_times (array-like): Array of time points corresponding to the GP light curve.
        starting_light_curve (array-like): Light curve values corresponding to `gp_times` for a given accretion disc element.
        t (float): The time at which to compute the temperature.

    Returns:
        float: Approximated temperature at time `t` based on the GP light curve data.
    """
    lower_diff = 1e10
    upper_diff = 1e10
    lower_idx = 0  # initialize to valid index 0
    upper_idx = gp_times.size - 1  # initialize to last valid index
    y_predicted = 0.

    for i in range(gp_times.size):
        val = gp_times[i]
        diff = abs(val - t)

        if val <= t and diff < lower_diff:
            lower_diff = diff
            lower_idx = i
        if val >= t and diff < upper_diff:
            upper_diff = diff
            upper_idx = i
            
        y_predicted = (starting_light_curve[lower_idx] + starting_light_curve[upper_idx]) / 2.

    return y_predicted

def radiative_efficiency_cpu(chi):
    """
    Compute temperature for a given radius R using the parameters of the accretion disc.

    Args:
        chi (float): Opacity parameter used in the temperature calculation.
        
    Returns:
        float: Computed radiative efficiency for the given chi value.
    """
    Z1 = 1 + (1 - chi**2)**(1/3) * ((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z2 = math.sqrt(3 * chi**2 + Z1**2)
    Eisco = (4 - chi * Z1 - math.sqrt(3 * Z2 - 2 * Z1)) / (3 * math.sqrt(3))
    eta = 1 - Eisco
    return eta

def L_edd_cpu(M):
    """
    Compute Eddington luminosity for a given mass M.

    Args:
        M (float): Mass parameter used in the temperature calculation.

    Returns:
        float: Computed L_edd for BH with mass M.
    """
    return 4*math.pi*G*M*c/0.1
    
def Temperature(R, M, f_edd, chi, R_in):
    """
    Compute temperature for a given radius R using the parameters of the accretion disc.

    Args:
        R (float): Radius at which to compute the temperature.
        M (float): Mass parameter used in the temperature calculation.
        f_edd (float): Eddington factor used in the temperature calculation.
        chi (float): Opacity parameter used in the temperature calculation.
        R_in (float): Inner radius parameter used in the temperature calculation.

    Returns:
        float: Computed temperature at radius R.
    """
    eta = radiative_efficiency_cpu(chi)
    M_dot = f_edd*L_edd_cpu(M)/(eta*c**2)
    return ((3*G*M*M_dot)/(8*math.pi*sigma_B*R**3))**(1/4)*(1-math.sqrt(R_in/R))**(1/4)  


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
    for i in range(len(thetas)):
        for j in range(len(Rs)):
            # Define a GP kernel with specified parameters
            kernel_ = terms.RealTerm(log_a=np.log(sigma), log_c=-np.log(tau))
            # Create a GP object with the kernel and mean temperature at the given radius
            # Here the radius is the mean radius of the element
            gp = celerite.GP(kernel_, mean=np.log10(Temperature(Rs[j]+(Rs[1]-Rs[0])/2, M, f_edd, chi, R_in)))
            gp.compute(times)  # Precompute the GP for the given time points
            starting_ys[i][j] = gp.sample()  # Sample initial temperatures
            gps[i][j] = gp  # Store the GP object
    return np.transpose(10.0**np.array(starting_ys), (2, 0, 1)) #to have [idx_t, idx_fi, idx_r] #,gps

def approximate_temperature(t, starting_light_curve, times):
    """
    Approximate the temperature at a given time `t` using a simple linear interpolation
    between the nearest time points in the `times` array.

    Args:
        t (float): The time at which to approximate the temperature.
        starting_light_curve (array-like): The light curve values corresponding to `times` for a given accretion disc element.
        times (array-like): The array of time points used to create the initial lightcurves.

    Returns:
        float: The approximated temperature at time `t`.
    """
    # Find the indices of the time points just before and after the given time `t`
    index1 = np.where(times < t)[-1]  # Indices where times are less than `t`
    index2 = np.where(times > t)[0]  # Indices where times are greater than `t`

    # Perform a simple average of the light curve values at the two nearest time points
    y_predicted = (starting_light_curve[index1[-1]] + starting_light_curve[index2[0]]) / 2

    return y_predicted

# Function to calculate the Planck spectrum for a given temperature
def planck_spectrum(nu, T):
    return (2 * h * nu**3) / (c**2) * (1 / (np.exp(h * nu / (k * T)) - 1))

# Function to calculate the ionizing flux for a given temperature
# The temperature here should be D*T/(1+z) where D is the Doppler factor, 1+z is the gravitational redshift and T is the rest frame wavelength 
def ionizing_flux_element(T,D,R,dr,dtheta):
    def integrand(nu):
        return planck_spectrum(nu, T)
    flux, _ = quad(integrand, 13.6 / h * 1.60218e-19, 20 * 13.6 / h * 1.60218e-19)
    return flux*R*dr*dtheta

@cuda.jit()
def accretion_disk_kernel(angle, R, M, times, gp_times, d_temp_curves, initial_phase, v_obs, r_obs, d_debug):
    idx_times, idx_angle, idx_R = cuda.grid(3)
    
    #starting_light_curve = cuda.shared.array(d_temp_curves, dtype=float64)
    if idx_times < times.size and idx_angle < angle.size and idx_R < R.size:
        j = idx_angle
        k = idx_R
        r = R[idx_R]
        theta = angle[idx_angle]
        gp_temp = d_temp_curves[:, idx_angle, idx_R]
        t = times[idx_times]

        theta_prime = de_evolve(float64(r), float64(theta), float64(t), float64(initial_phase), float64(M))

        positions = position_components(theta_prime, r) #returns (x,y), if check you need d_debug = positions[i]
        velocities = velocity_components(theta_prime, r, M, 0.)
        doppler = doppler_factor(v_obs, velocities, r_obs, positions)
        T = find_temperature(gp_times, gp_temp, t)
        d_debug[idx_times, idx_angle, idx_R] = compute_ionizing_flux(T,doppler,z, positions,dr,dtheta)

if __name__ == 'main_':
    # Defining constants
    h = 6.62607015e-34  # Planck's constant (Joule-seconds)
    c = 3e8  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (Joule/Kelvin)
    G = 6.67e-11  # Gravitational constant (m^3/kg/s^2)
    sigma_B = 1.28e-23  # Stefan-Boltzmann constant (W/m^2/K^4)
    M_sun = np.float64(2e30)  # Solar mass (kg)

    M = np.float64(1e8*M_sun)
    d_M = cuda.to_device(M)
    inc = np.radians(30.)
    chi = 0.8
    R_in = np.float64(1.5*R_ISCO(M, chi))
    f_edd = 0.5
    z = 0.1
    tau = 100.
    sigma = 0.5

    r = np.logspace(1.1*R_in,3.0*R_in, 20).astype(np.float64) #if r_1 = R_in Nan for high chi, check
    #r = np.linspace(100,200).astype(np.float64)
    angle = np.linspace(0., 2.*np.pi, 20).astype(np.float64)
    times = np.linspace(20., 40., 20).astype(np.float64)
    gp_times = np.linspace(0., 50., 1000).astype(np.float64)

    dr = r[1]-r[0]
    dtheta = angle[1]-angle[0]
    d_r = cuda.to_device(r)
    d_angle = cuda.to_device(angle)
    d_times = cuda.to_device(times)
    d_gp_times = cuda.to_device(gp_times)
    r_obs = cuda.to_device([5.0*R_in,5.0*R_in])
    #r_obs = cuda.to_device([100,100])
    v_obs = cuda.to_device([c/2,c/2])

    deb = np.zeros([times.size, angle.size, r.size])
    d_deb = cuda.to_device(deb)
    deb_def = np.zeros([times.size])
    d_deb_def = cuda.to_device(deb_def)

    start_time = tempo.time()
    temp_curves = generate_set_of_temperatures(gp_times, r, angle, M, f_edd, chi, R_in, tau, sigma)
    temp_curves = np.ascontiguousarray(temp_curves)
    d_temp_curves = cuda.to_device(np.ascontiguousarray(temp_curves))
    end_time = tempo.time()
    print(f'Expected shape: ({len(gp_times):.0f}, {len(angle):.0f}, {len(r):.0f})')
    print('Resulting shape', d_temp_curves.shape)
    print('Tempo per generare curve di temperatura:', end_time-start_time)

    threads_per_block =  (4)
    blocks_per_grid = (
      (d_times.size + (threads_per_block - 1)) // threads_per_block,
    )

    threads_per_block_1 =  (4,4,4)
    blocks_per_grid_1 = (
      (d_times.size + (threads_per_block_1[0] - 1)) // threads_per_block_1[0],
      (d_angle.size + (threads_per_block_1[1] - 1)) // threads_per_block_1[1],
      (d_r.size + (threads_per_block_1[2] - 1)) // threads_per_block_1[2]
    )

    start_time_gpu = tempo.time()
    accretion_disk_kernel[blocks_per_grid_1, threads_per_block_1](d_angle, d_r, M, d_times, d_gp_times, d_temp_curves,  0., v_obs, r_obs, d_deb)
    cuda.synchronize()
    sum_kernel[blocks_per_grid, threads_per_block](d_r, d_angle, d_times, d_deb, d_deb_def)
    cuda.synchronize()
    end_time_gpu_without_copying = tempo.time()