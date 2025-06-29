import numpy as np
import math
import celerite
from celerite import terms
import time as tempo
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

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
    y_predicted = np.interp(t, times, starting_light_curve)#(starting_light_curve[index1[-1]] + starting_light_curve[index2[0]]) / 2

    return y_predicted

def new_approximate_temperature(times, starting_light_curve, gp_times):
    """
    Interpolate a 3D light curve (time, angle, radius) over a new time grid.

    Args:
        times (array-like): Target times to interpolate to.
        starting_light_curve (ndarray): Shape (T, A, R), where T = len(gp_times)
        gp_times (array-like): Time points corresponding to the first axis of starting_light_curve.

    Returns:
        ndarray: Interpolated light curve values, shape (len(times), A, R)
    """
    times = np.asarray(times, dtype=float).flatten()
    gp_times = np.asarray(gp_times, dtype=float).flatten()
    starting_light_curve = np.asarray(starting_light_curve, dtype=float)

    # Interpolator over axis=0 (time), returns shape (len(times), angle, radius)
    interpolator = interp1d(gp_times, starting_light_curve, axis=0, bounds_error=False, fill_value="extrapolate")
    result = interpolator(times)

    return result

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

def new_ionizing_flux_element(T, R, dr, dtheta, num_points=1000):
    """
    Fully vectorized calculation of ionizing flux using discrete integration.

    Args:
        T (ndarray): Temperature array, shape (A, R, Z)
        R (float or ndarray): Radius scalar or array broadcastable to T
        dr (float): r[1]-r[0]
        dtheta (float): angle[1]-angle[0]
        num_points (int): Number of frequency samples

    Returns:
        ndarray: Ionizing flux at each T element, same shape as T
    """
    #T = np.asarray(T)
    #shape = T.shape
    #R = np.broadcast_to(R, shape)
    T = T[..., None]

    # Frequency range in Hz
    nu_min = (13.6 * 1.60218e-19) / h
    nu_max = 20 * nu_min
    nu = np.linspace(nu_min, nu_max, num_points)
    dnu = nu[1] - nu[0]

    # Vectorized Planck spectrum integration
    intensity = planck_spectrum(nu, T)  # shape (T.shape + (num_points,))
    flux_integrated = np.trapezoid(intensity, dx=dnu, axis=-1)  # integrate over frequency

    return flux_integrated * R * dr * dtheta

def accretion_disk(r, angle, times, r1, v1, temp_curves, gp_times):
    """
    Function to loop on r, angle, times to obtain the flux.
    Used to check if reshape/meshgrid versions are consistent
    with the results we want. Debugging purposes

    Args:
        r (array, float): radii of accretion disk grid
        angle (array, float): angles of accretion disk grid
        times (array, float): times over which flux is computed
        r1 (tuple): (x,y) position components of BLR element
        v1 (tuple): (vx, vy) velocity components of BLR element
        temp_curves (array): Shape (T, A, R), where T = len(gp_times); 
                            contains temperature curves for each element of the accretion disk 
                            for each time in gp_times
        gp_times (array): times over which computed temperature curves
    Returns:
        ndarray: Ionizing flux at each T element, same shape as T, A, R
    """
    D = np.zeros((len(times), len(angle), len(r)))
    angl = np.zeros((len(times), len(angle), len(r)))
    vx = np.zeros((len(times), len(angle), len(r)))
    vy = np.zeros((len(times), len(angle), len(r)))
    x = np.zeros((len(times), len(angle), len(r)))
    y = np.zeros((len(times), len(angle), len(r)))
    gamma_loop = np.zeros((2, len(times), len(angle), len(r)))
    T= np.zeros((len(times), len(angle), len(r)))
    flux = np.zeros((len(times), len(angle), len(r)))

    #check = np.zeros((2, len(times), len(angle), len(r)))

    for idx_time, t in enumerate(times):
        for idx_ang, ang in enumerate(angle):
            for idx_r, erre in enumerate(r):
                td = erre*(1+np.sin(inc)*np.cos(ang))
                kf = np.sqrt(G * M / (erre*erre*erre))
                
                ang_pre = (ang - (t -td)*kf + 0.) - (2*np.pi) * np.floor((ang - (t -td)*kf + 0.)/(2*np.pi)) 
                
                #angl.append(ang_pre)
                angl[idx_time, idx_ang, idx_r] = ang_pre
                v_kep = np.sqrt(G * M / erre)  # Keplerian velocity
                vx[idx_time, idx_ang, idx_r] = v_kep*np.cos(ang_pre)
                vy[idx_time, idx_ang, idx_r] = -v_kep*np.sin(ang_pre)
                x[idx_time, idx_ang, idx_r] = erre * np.sin(ang_pre)
                y[idx_time, idx_ang, idx_r] = erre*np.cos(ang_pre)

                r2 = np.array([erre * np.sin(ang_pre), erre * np.cos(ang_pre)]) 
                v2 = np.array([v_kep*np.cos(ang_pre), -v_kep*np.sin(ang_pre)])

                v2_para = np.dot(v2, v1) * v1 / (np.linalg.norm(v1)*np.linalg.norm(v1))  # Parallel component of v2 
                #D[idx_time, idx_ang, idx_r] = v2[0]
                #D[idx_time, idx_ang, idx_r] = np.dot(v2, v1)
                #check[:, idx_time, idx_ang, idx_r] = v2_para
                v2_perp = v2 - v2_para  # Perpendicular component of v2
                gamma_2 = 1 / np.sqrt(1 - np.linalg.norm(v2)**2 / c**2)  # Lorentz factor for v2
                beta = 1 / c * (v1 - v2 + 1 / gamma_2 * v2_perp) / (1 - np.dot(v1, v2) / c**2)
                               
                r_vers = (r2 - r1) / np.linalg.norm(r2 - r1)  # Unit vector in the direction of r2 - r1
                gamma_loop[:, idx_time, idx_ang, idx_r] = r_vers
                gamma = np.sqrt(1 / (1 - np.linalg.norm(beta)**2))
                dop = 1 / (gamma * (1 - np.dot(beta, r_vers)))
                #D.append(dop)  # Doppler factor
                D[idx_time, idx_ang, idx_r] = dop

                temp = approximate_temperature(t, temp_curves[:, idx_ang, idx_r], gp_times)
                #T.append(temp)
                T[idx_time, idx_ang, idx_r] = temp
                f = ionizing_flux_element(dop*temp/(1+z), dop, erre,dr,dtheta)
                #flux.append(f)
                flux[idx_time, idx_ang, idx_r] = f
    
    return angl, x,y, vx, vy, gamma_loop, D, T, flux

if __name__ == '__main__':
    np.random.seed(42)
    # Defining constants
    h = 6.62607015e-34  # Planck's constant (Joule-seconds)
    c = 3e8  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (Joule/Kelvin)
    G = 6.67e-11  # Gravitational constant (m^3/kg/s^2)
    sigma_B = 1.28e-23  # Stefan-Boltzmann constant (W/m^2/K^4)
    M_sun = np.float64(2e30)  # Solar mass (kg)

    M = np.float64(1e8*M_sun)               # Black hole mass
    inc = np.radians(30.)                   # Inclination
    chi = 0.8                               # Spin Parameter
    R_in = np.float64(1.5*R_ISCO(M, chi))   # R_in
    f_edd = 0.5                             # f_edd
    z = 0.1                                 # redshift
    tau = 100.*86400.                              # DRW tau
    sigma = 0.5  
    initial_phase = 0.

    time_dim = 10
    theta_dim = 14
    r_dim = 7
    """ Radii, angles and times definition: change linspace size to speed up computation """
    r = np.logspace(np.log10(1.1*R_in),np.log10(3.0*R_in), r_dim).astype(np.float64)

    #r = np.linspace(100,200).astype(np.float64)
    angle = np.linspace(0., 2.*np.pi, theta_dim).astype(np.float64)
    times = np.logspace(4., 5., time_dim).astype(np.float64)

    #times for temperature curves: ranges depend on times
    gp_times = np.logspace(3., 6., 1000).astype(np.float64)

    # dr, dtheta from linspaces
    dr = r[1]-r[0]
    dtheta = angle[1]-angle[0]

    """Properties of BLR element: will become 2D matrices with dimensions(2, 175x175)"""
    r1 = np.array([300.0*R_in,300.0*R_in])
    v1 = np.array([c/2,c/2]) #fai conto da r_obs

    """ STEP 1. Temperature Curves Generation"""
    start_time = tempo.time()
    temp_curves = generate_set_of_temperatures(gp_times, r, angle, M, f_edd, chi, R_in, tau, sigma)
    temp_curves = np.ascontiguousarray(temp_curves)
    end_time = tempo.time()
    print(f'Expected shape: ({len(gp_times):.0f}, {len(angle):.0f}, {len(r):.0f})')
    print('Resulting shape', temp_curves.shape)
    print('Tempo per generare curve di temperatura:', end_time-start_time)

    start_time_loop = tempo.perf_counter()
    ang_pre_loop, x_loop, y_loop, vx_loop, vy_loop, gamma_loop, dop_loop, temp_loop, flux_loop= accretion_disk(r, angle, times, r1, v1, temp_curves, gp_times)
    flux_def_loop = np.sum(np.sum(flux_loop, axis = 2), axis = 1)
    end_time_loop = tempo.perf_counter()
    
    r_reshaped = r.reshape(1,1,r_dim)
    time_reshaped = times.reshape(time_dim, 1, 1)
    theta_reshaped = angle.reshape(1, theta_dim, 1)
    
    start_time = tempo.perf_counter()

    td = r_reshaped*(1+np.sin(inc)*np.cos(theta_reshaped))
    td = np.tile(td, (time_dim, 1, 1))

    kf = np.sqrt(G * M / (r_reshaped*r_reshaped*r_reshaped))
    kf = np.tile(kf, (time_dim, theta_dim, 1))

    ang_pre = (theta_reshaped - (time_reshaped -td)*kf + initial_phase)
    ang_pre = ang_pre - (2*np.pi) * np.floor((theta_reshaped - (time_reshaped - td)*kf + 0.)/(2*np.pi))

    v_kep = np.sqrt(G * M / r_reshaped)
    v_kep = np.tile(v_kep, (time_dim, theta_dim, 1))

    r2 = np.array([r_reshaped*np.sin(ang_pre), r_reshaped*np.cos(ang_pre)])
    v2 = np.array([v_kep*np.cos(ang_pre), -v_kep*np.sin(ang_pre)])

    v2_dot_v1 = v2[0]*v1[0]+v2[1]*v1[1]
    r2_minus_r1_norm = np.sqrt((r2[0]-r1[0])*(r2[0]-r1[0])+(r2[1]-r1[1])*(r2[1]-r1[1]))
    v1_reshaped = v1.reshape(2,1,1,1)
    r1_reshaped = r1.reshape(2,1,1,1)
    
    v2_para = (v2_dot_v1* v1_reshaped) / np.linalg.norm(v1, axis = 0)**2

    v2_perp = v2 - v2_para  # Perpendicular component of v2
    gamma_2 = 1 / np.sqrt(1 - np.linalg.norm(v2, axis = 0)**2 / (c**2))  # Lorentz factor for v2
    beta = 1 / c * (v1_reshaped - v2 + 1 / gamma_2 * v2_perp) / (1 - v2_dot_v1 / (c*c))

    beta_squared_norm = beta[0]*beta[0]+beta[1]*beta[1]
    r_vers = (r2 - r1_reshaped) / r2_minus_r1_norm # Unit vector in the direction of r2 - r1
    gamma = np.sqrt(1 / (1 - beta_squared_norm))

    beta_dot_r_vers = beta[0]*r_vers[0] + beta[1]*r_vers[1]
    dop = 1 / (gamma * (1 - beta_dot_r_vers))     #np.dot(beta, r_vers)))    

    temp = new_approximate_temperature(time_reshaped, temp_curves, gp_times)
    f = new_ionizing_flux_element(dop*temp/(1+z), r_reshaped, dr, dtheta)
    f_def = np.sum(np.sum(f, axis = 2), axis = 1)

    end_time = tempo.perf_counter()

    print(end_time-start_time)


    print('Reshape:', end_time-start_time)
    print('Loop:', end_time_loop-start_time_loop)
    print('Ratio:', (end_time-start_time)/(end_time_loop-start_time_loop))


    choice_map = {
        'ang_pre': ang_pre,
        'D': dop,
        'T': temp,
        'DT': dop*temp,
        'f': f
    }

    choice_map_loop = {
        'ang_pre_loop': ang_pre_loop,
        'D_loop': dop_loop,
        'T_loop': temp_loop,
        'DT_loop': dop_loop*temp_loop,
        'f_loop': flux_loop
    }

    for name, values in choice_map.items():
        print('[testing] Plotting property '+ name)
        plt.figure()
        plt.plot(times, values[:, 0, 0], label = name+', element 0')
        plt.plot(times, values[:, 1, 1], label = name+', element 1')
        plt.title(name)
        
        loop_name = name + '_loop'
        if loop_name in choice_map_loop:
            loop_values = choice_map_loop[loop_name]
            plt.plot(times, loop_values[:, 0, 0], label=f'{name}, element 0 (loop)', linestyle='-', alpha=0.5)
            plt.plot(times, loop_values[:, 1, 1], label=f'{name}, element 1 (loop)', linestyle='--', alpha=0.5)

        plt.xlabel('t')
        plt.ylabel(name)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.legend()
        plt.savefig('test_cpu/'+name+'.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

