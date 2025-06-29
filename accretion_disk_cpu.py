import numpy as np
import math
import celerite
from celerite import terms
import time as tempo
from scipy.integrate import quad
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

def accretion_disk(r, angle, times, r1, v1, temp_curves, gp_times):
    D = []
    angl = []
    vx = []
    vy = []
    x = []
    y = []
    T = []
    flux = []

    D = np.zeros((len(times), len(angle), len(r)))
    angl = np.zeros((len(times), len(angle), len(r)))
    vx = np.zeros((len(times), len(angle), len(r)))
    vy = np.zeros((len(times), len(angle), len(r)))
    x = np.zeros((len(times), len(angle), len(r)))
    y = np.zeros((len(times), len(angle), len(r)))
    T= np.zeros((len(times), len(angle), len(r)))
    flux = np.zeros((len(times), len(angle), len(r)))

    for idx_time, t in enumerate(times):
        for idx_ang, ang in enumerate(angle):
            for idx_r, erre in enumerate(r):
                td = erre*(1+math.sin(inc)*math.cos(ang))
                kf = math.sqrt(G * M / erre**3)
                
                ang_pre = (ang - (t -td)*kf + 0.) - (2*math.pi) * math.floor((ang - (t -td)*kf + 0.)/(2*math.pi)) 
                #angl.append(ang_pre)
                angl[idx_time, idx_ang, idx_r] = ang_pre
                v_kep = math.sqrt(G * M / erre)  # Keplerian velocity
                """vx.append(v_kep*np.cos(ang_pre))
                vy.append(-v_kep*np.sin(ang_pre))
                x.append(erre * math.sin(ang_pre))
                y.append(erre*math.cos(ang_pre))"""
                vx[idx_time, idx_ang, idx_r] = v_kep*np.cos(ang_pre)
                vy[idx_time, idx_ang, idx_r] = -v_kep*np.sin(ang_pre)
                x[idx_time, idx_ang, idx_r] = erre * math.sin(ang_pre)
                y[idx_time, idx_ang, idx_r] = erre*math.cos(ang_pre)

                r2 = np.array([erre * math.sin(ang_pre), erre * math.cos(ang_pre)]) 
                v2 = np.array([v_kep*np.cos(ang_pre), -v_kep*np.sin(ang_pre)])
                
                v2_para = np.dot(v2, v1) * v1 / np.linalg.norm(v1)**2  # Parallel component of v2 
                v2_perp = v2 - v2_para  # Perpendicular component of v2
                gamma_2 = 1 / np.sqrt(1 - np.linalg.norm(v2)**2 / c**2)  # Lorentz factor for v2
                beta = 1 / c * (v1 - v2 + 1 / gamma_2 * v2_perp) / (1 - np.dot(v1, v2) / c**2)
                r_vers = (r2 - r1) / np.linalg.norm(r2 - r1)  # Unit vector in the direction of r2 - r1
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

    """angl = np.array(angl)
    angl = angl.reshape(len(times), len(angle), len(r))
    y = np.array(y)
    x = np.array(x)
    x = x.reshape(len(times), len(angle), len(r))
    y = np.array(y)
    y = y.reshape(len(times), len(angle), len(r))
    vx = np.array(vx)
    vx = vx.reshape(len(times), len(angle), len(r))
    vy = np.array(vy)
    vy = vy.reshape(len(times), len(angle), len(r))
    D = np.array(D)
    D = D.reshape(len(times), len(angle), len(r))
    T = np.array(T)
    T = T.reshape(len(times), len(angle), len(r))
    flux = np.array(flux)
    flux = flux.reshape(len(times), len(angle), len(r))"""
    return angl, x, y, vx, vy, D, T, flux

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
    sigma = 0.5                             # DRW sigma

    """ Radii, angles and times definition: change linspace size to speed up computation """
    r = np.logspace(np.log10(1.1*R_in),np.log10(3.0*R_in), 175).astype(np.float64)
    #r = np.linspace(100,200).astype(np.float64)
    angle = np.linspace(0., 2.*np.pi, 175).astype(np.float64)
    times = np.logspace(1., 5., 175).astype(np.float64)

    #times for temperature curves: ranges depend on times
    gp_times = np.logspace(0., 6., 1000).astype(np.float64)

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

    #plotting temperature curves
    ax = plt.gca()
    ax.plot(gp_times, temp_curves[:, 0,0], label = 'T curve, element 0')
    ax.plot(gp_times, temp_curves[:, 1,1], label = 'T curve, element 1')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Temperature [K]')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('test_cpu/temp_curves.png', dpi = 300, bbox_inches = 'tight')
    ax.clear()

    """2. Flux Generation with CPU"""
    start_time = tempo.time()
    angl, x, y, vx, vy, D, T, flux = accretion_disk(r, angle, times, r1, v1, temp_curves, gp_times)
    flux_def = np.array(np.sum(np.sum(flux, axis = 2), axis=1))
    end_time = tempo.time()
    print("Total elements in f_def:", flux_def.shape)
    print("Expected elements:", len(times))
    print('Tempo per CPU: ', end_time-start_time)

    """3. Testing for CPU"""
    check = 'f' 
    choice_map = {
        'angl': angl,
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'D': D,
        'T': T,
        'DT': D * T,
        'f': flux
    }

    for name, values in choice_map.items():
        print('[testing] Plotting property '+ name)
        plt.figure()
        plt.plot(times, values[:, 0, 0], label = name+', element 0')
        plt.plot(times, values[:, 1, 1], label = name+', element 1')
        plt.title(name)
        plt.xlabel('t')
        plt.ylabel(name)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('test_cpu/'+name+'.png', dpi = 300, bbox_inches = 'tight')
        plt.show()