import numpy as np
import matplotlib.pyplot as plt
from derivkit import ForecastKit
import sys
sys.path.insert(0, "/Users/bsherwin/Documents/CMBxLBG/lbg-desc-forecast/src")
from lbg_tools import TomographicBin
from lbg_desc_forecast.forecaster import Forecaster
from lbg_desc_forecast.cosmo_factories import MainCosmology
from lbg_desc_forecast.default_lbg import get_lbg_mappers


def fiducial_model(theta):
    '''Generates fiducial, unbiased data vector for given cosmological parameters'''
    Omega_M, sigma8, Omega_b, h, n_s, m_nu, bu, bg, br, bu_int, bg_int, br_int = theta 
    cosmo = MainCosmology(Omega_M=Omega_M, sigma8 = sigma8, Omega_b = Omega_b, h = h, n_s = n_s, m_nu = m_nu)
    
    # Use clean mappers (no dz, no stretch, and use Petri Y10 interloper fractions)
    mappers = get_lbg_mappers(10, contamination=0.1)  

    for mapper in mappers:
        mapper.dz = 0.0
        mapper.stretch = 1.0
        mapper.dz_interlopers = 0.0
        mapper.stretch_interlopers = 1.0
        if mapper.drop_band == "u":
            mapper.f_interlopers = 0.07 # Petri Y10
            mapper.g_bias = bu 
            mapper.g_bias_inter = bu_int
        elif mapper.drop_band == "g":
            mapper.f_interlopers = 0.10 # Petri Y10
            mapper.g_bias = bg
            mapper.g_bias_inter = bg_int
        elif mapper.drop_band == "r":
            mapper.f_interlopers = 0.03 # Petri Y10
            mapper.g_bias = br
            mapper.g_bias_inter = br_int

    # forceast and create unbiased signal, no noise
    forecaster = Forecaster(mappers, cosmo)
    return forecaster.create_signal(add_noise=False)

def get_biased_signal(theta, param_name, bias_val, band=None):
    """Generates a systematic biased data vector"""
    Omega_M, sigma8, Omega_b, h, n_s, m_nu, bu, bg, br, bu_int, bg_int, br_int = theta 
    cosmo = MainCosmology(Omega_M=Omega_M, sigma8 = sigma8, Omega_b = Omega_b, h = h, n_s = n_s, m_nu = m_nu)
    
    # Use clean mappers (no dz, no stretch, and use Petri Y10 interloper fractions)
    mappers = get_lbg_mappers(10, contamination=0.1)
    
    for mapper in mappers:
        mapper.dz = 0.0
        mapper.stretch = 1.0
        mapper.dz_interlopers = 0.0
        mapper.stretch_interlopers = 1.0
        if mapper.drop_band == "u":
            mapper.f_interlopers = 0.07 # Petri Y10 
            mapper.g_bias = bu
            mapper.g_bias_inter = bu_int
        elif mapper.drop_band == "g":
            mapper.f_interlopers = 0.10 # Petri Y10
            mapper.g_bias = bg
            mapper.g_bias_inter = bg_int
        elif mapper.drop_band == "r":
            mapper.f_interlopers = 0.03 # Petri Y10
            mapper.g_bias = br
            mapper.g_bias_inter = br_int
        
        # Apply a bias (ie, dz, stretch) to specified band (one at a time)
        if band is None or mapper.drop_band == band:
            setattr(mapper, param_name, bias_val)

    # create forecaster and signal with bias, no noise
    forecaster = Forecaster(mappers, cosmo)
    return forecaster.create_signal(add_noise=False)

if __name__ == "__main__":
    
    # 1. Generate covariance matrix using unbiased mappers and fiducial cosmology
    theta0 = np.array([0.3156, 0.831, 0.0491685, 0.6727, 0.9645, 0.06, 2.1574085950258355, 3.3278572844505083, 4.858283802099753, 1.49, 1.69, 1.79])  
    cosmo = MainCosmology(Omega_M=theta0[0], sigma8=theta0[1], Omega_b=theta0[2], h=theta0[3], n_s=theta0[4], m_nu=theta0[5])
    
    # Use clean mappers (no dz, no stretch, and use Petri Y10 interloper fractions) for cov calculation
    base_mappers = get_lbg_mappers(10, contamination=0.1)
    for mapper in base_mappers:
        mapper.dz = 0.0
        mapper.stretch = 1.0
        mapper.dz_interlopers = 0.0
        mapper.stretch_interlopers = 1.0
        if mapper.drop_band == "u":
            mapper.f_interlopers = 0.07 # Petri Y10
            mapper.g_bias = theta0[6]
            mapper.g_bias_inter = theta0[9]
        elif mapper.drop_band == "g":
            mapper.f_interlopers = 0.10 # Petri Y10
            mapper.g_bias = theta0[7]
            mapper.g_bias_inter = theta0[10]
        elif mapper.drop_band == "r":
            mapper.f_interlopers = 0.03 # Petri Y10
            mapper.g_bias = theta0[8]
            mapper.g_bias_inter = theta0[11]
    
    base_forecaster = Forecaster(base_mappers, cosmo)
    print("Generating covariance matrix...", flush=True)
    base_forecaster.create_cov()
    cov = base_forecaster.cov

    #np.save("data_cov_petri_y10_neutrino.npy", cov)

    #  2. Fisher matrix calculation
    print("Calculating Fisher Matrix...", flush=True)
    
    data_unbiased = fiducial_model(theta0)
    fk = ForecastKit(function=fiducial_model, theta0=theta0, cov=cov)
    fisher = fk.fisher()
    
    inv_fisher = np.linalg.inv(fisher)
    sigma_theta = np.sqrt(np.diag(inv_fisher))
    
    #np.save("neutrino_fisher_petri_y10.npy", fisher)
    
    
    # Define list of bias scenarios to test
    scenarios = [
        {'name': 'dz_u_0.01', 'param': 'dz', 'val': 0.01, 'band': 'u'},
        {'name': 'dz_g_0.01', 'param': 'dz', 'val': 0.01, 'band': 'g'},
        {'name': 'dz_r_0.01', 'param': 'dz', 'val': 0.01, 'band': 'r'},
        {'name': 'dz_u_0.02', 'param': 'dz', 'val': 0.02, 'band': 'u'},
        {'name': 'dz_g_0.02', 'param': 'dz', 'val': 0.02, 'band': 'g'},
        {'name': 'dz_r_0.02', 'param': 'dz', 'val': 0.02, 'band': 'r'},
        {'name': 'dz_u_0.03', 'param': 'dz', 'val': 0.03, 'band': 'u'},
        {'name': 'dz_g_0.03', 'param': 'dz', 'val': 0.03, 'band': 'g'},
        {'name': 'dz_r_0.03', 'param': 'dz', 'val': 0.03, 'band': 'r'},
        {'name': 'dz_u_0.04', 'param': 'dz', 'val': 0.04, 'band': 'u'},
        {'name': 'dz_g_0.04', 'param': 'dz', 'val': 0.04, 'band': 'g'},
        {'name': 'dz_r_0.04', 'param': 'dz', 'val': 0.04, 'band': 'r'},
        {'name': 'dz_u_0.05', 'param': 'dz', 'val': 0.05, 'band': 'u'},
        {'name': 'dz_g_0.05', 'param': 'dz', 'val': 0.05, 'band': 'g'},
        {'name': 'dz_r_0.05', 'param': 'dz', 'val': 0.05, 'band': 'r'},
        {'name': 'dz_u_0.1', 'param': 'dz', 'val': 0.1, 'band': 'u'},
        {'name': 'dz_g_0.1', 'param': 'dz', 'val': 0.1, 'band': 'g'},
        {'name': 'dz_r_0.1', 'param': 'dz', 'val': 0.1, 'band': 'r'},
        
        {'name': 'dz_u_int_0.01', 'param': 'dz_interlopers', 'val': 0.01, 'band': 'u'},
        {'name': 'dz_g_int_0.01', 'param': 'dz_interlopers', 'val': 0.01, 'band': 'g'},
        {'name': 'dz_r_int_0.01', 'param': 'dz_interlopers', 'val': 0.01, 'band': 'r'},
        {'name': 'dz_u_int_0.02', 'param': 'dz_interlopers', 'val': 0.02, 'band': 'u'},
        {'name': 'dz_g_int_0.02', 'param': 'dz_interlopers', 'val': 0.02, 'band': 'g'},
        {'name': 'dz_r_int_0.02', 'param': 'dz_interlopers', 'val': 0.02, 'band': 'r'},
        {'name': 'dz_u_int_0.03', 'param': 'dz_interlopers', 'val': 0.03, 'band': 'u'},
        {'name': 'dz_g_int_0.03', 'param': 'dz_interlopers', 'val': 0.03, 'band': 'g'},
        {'name': 'dz_r_int_0.03', 'param': 'dz_interlopers', 'val': 0.03, 'band': 'r'},
        {'name': 'dz_u_int_0.04', 'param': 'dz_interlopers', 'val': 0.04, 'band': 'u'},
        {'name': 'dz_g_int_0.04', 'param': 'dz_interlopers', 'val': 0.04, 'band': 'g'},
        {'name': 'dz_r_int_0.04', 'param': 'dz_interlopers', 'val': 0.04, 'band': 'r'},
        {'name': 'dz_u_int_0.05', 'param': 'dz_interlopers', 'val': 0.05, 'band': 'u'},
        {'name': 'dz_g_int_0.05', 'param': 'dz_interlopers', 'val': 0.05, 'band': 'g'},
        {'name': 'dz_r_int_0.05', 'param': 'dz_interlopers', 'val': 0.05, 'band': 'r'},
        {'name': 'dz_u_int_0.1', 'param': 'dz_interlopers', 'val': 0.1, 'band': 'u'},
        {'name': 'dz_g_int_0.1', 'param': 'dz_interlopers', 'val': 0.1, 'band': 'g'},
        {'name': 'dz_r_int_0.1', 'param': 'dz_interlopers', 'val': 0.1, 'band': 'r'},
        
        {'name': 'stretch_u_1.01', 'param': 'stretch', 'val': 1.01, 'band': 'u'},
        {'name': 'stretch_g_1.01', 'param': 'stretch', 'val': 1.01, 'band': 'g'},
        {'name': 'stretch_r_1.01', 'param': 'stretch', 'val': 1.01, 'band': 'r'},
        {'name': 'stretch_u_1.025', 'param': 'stretch', 'val': 1.025, 'band': 'u'},
        {'name': 'stretch_g_1.025', 'param': 'stretch', 'val': 1.025, 'band': 'g'},
        {'name': 'stretch_r_1.025', 'param': 'stretch', 'val': 1.025, 'band': 'r'},
        {'name': 'stretch_u_1.05', 'param': 'stretch', 'val': 1.05, 'band': 'u'},
        {'name': 'stretch_g_1.05', 'param': 'stretch', 'val': 1.05, 'band': 'g'},
        {'name': 'stretch_r_1.05', 'param': 'stretch', 'val': 1.05, 'band': 'r'},
        {'name': 'stretch_u_1.1', 'param': 'stretch', 'val': 1.1, 'band': 'u'},
        {'name': 'stretch_g_1.1', 'param': 'stretch', 'val': 1.1, 'band': 'g'},
        {'name': 'stretch_r_1.1', 'param': 'stretch', 'val': 1.1, 'band': 'r'},
        
        {'name': 'stretch_u_int_1.01', 'param': 'stretch_interlopers', 'val': 1.01, 'band': 'u'},
        {'name': 'stretch_g_int_1.01', 'param': 'stretch_interlopers', 'val': 1.01, 'band': 'g'},
        {'name': 'stretch_r_int_1.01', 'param': 'stretch_interlopers', 'val': 1.01, 'band': 'r'},
        {'name': 'stretch_u_int_1.025', 'param': 'stretch_interlopers', 'val': 1.025, 'band': 'u'},
        {'name': 'stretch_g_int_1.025', 'param': 'stretch_interlopers', 'val': 1.025, 'band': 'g'},
        {'name': 'stretch_r_int_1.025', 'param': 'stretch_interlopers', 'val': 1.025, 'band': 'r'},
        {'name': 'stretch_u_int_1.05', 'param': 'stretch_interlopers', 'val': 1.05, 'band': 'u'},
        {'name': 'stretch_g_int_1.05', 'param': 'stretch_interlopers', 'val': 1.05, 'band': 'g'},
        {'name': 'stretch_r_int_1.05', 'param': 'stretch_interlopers', 'val': 1.05, 'band': 'r'},
        {'name': 'stretch_u_int_1.1', 'param': 'stretch_interlopers', 'val': 1.1, 'band': 'u'},
        {'name': 'stretch_g_int_1.1', 'param': 'stretch_interlopers', 'val': 1.1, 'band': 'g'},
        {'name': 'stretch_r_int_1.1', 'param': 'stretch_interlopers', 'val': 1.1, 'band': 'r'},
        
        {'name': 'f_int_g_0.08', 'param': 'f_interlopers', 'val': 0.08, 'band': 'g'},
        {'name': 'f_int_g_0.09', 'param': 'f_interlopers', 'val': 0.09, 'band': 'g'},
        {'name': 'f_int_g_0.095', 'param': 'f_interlopers', 'val': 0.095, 'band': 'g'},
        {'name': 'f_int_g_0.105', 'param': 'f_interlopers', 'val': 0.105, 'band': 'g'},
        {'name': 'f_int_g_0.11', 'param': 'f_interlopers', 'val': 0.11, 'band': 'g'},
        {'name': 'f_int_g_0.12', 'param': 'f_interlopers', 'val': 0.12, 'band': 'g'},
        {'name': 'f_int_u_0.05', 'param': 'f_interlopers', 'val': 0.05, 'band': 'u'},
        {'name': 'f_int_u_0.06',	'param':'f_interlopers','val' : 0.06,	'band':'u'},
        {'name': 'f_int_u_0.065',	'param':'f_interlopers','val' : 0.065,	'band':'u'},
        {'name': 'f_int_u_0.075',	'param':'f_interlopers','val' : 0.075,	'band':'u'},
        {'name': 'f_int_u_0.08',	'param':'f_interlopers','val' : 0.08,	'band':'u'},
        {'name': 'f_int_u_0.09',	'param':'f_interlopers','val' : 0.09,	'band':'u'},
        {'name': 'f_int_r_0.01', 'param': 'f_interlopers', 'val': 0.01, 'band': 'r'},
        {'name': 'f_int_r_0.02', 'param': 'f_interlopers', 'val': 0.02, 'band': 'r'},
        {'name': 'f_int_r_0.025',	'param':'f_interlopers','val' : 0.025,	'band':'r'},
        {'name': 'f_int_r_0.035',	'param':'f_interlopers','val' : 0.035,	'band':'r'},
        {'name': 'f_int_r_0.04',	'param':'f_interlopers','val' : 0.04,	'band':'r'},
        {'name': 'f_int_r_0.05',	'param':'f_interlopers','val' : 0.05,	'band':'r'},  
    ]

    print(f"{'Scenario':<15} | {'Sig d_Om':<8} | {'Sig d_m_nu':<8}")
    print("-" * 40)

    # Loop through scenarios, calculate systematic bias shifts, and store results
    raw_shifts = {}
    sigma_shifts = {}
    for s in scenarios:
        
        # Create biased data vector for this scenario
        data_biased = get_biased_signal(theta0, s['param'], s['val'], s['band'])
        
        # Calculate shift in signal (delta_nu)
        dn = fk.delta_nu(data_unbiased=data_unbiased, data_biased=data_biased)
        
        # Project bias onto parameter shifts (delta_theta) with Fisher matrix
        _, delta_theta = fk.fisher_bias(fisher_matrix=fisher, delta_nu=dn)
        
        # Calculate sigma units
        delta_sigma = delta_theta / sigma_theta
        
        # Store result
        raw_shifts[s['name']] = delta_theta
        sigma_shifts[s['name']] = delta_sigma
        
        print(f"{s['name']:<15} | {delta_sigma[0]:.2e} | {delta_sigma[5]:.2e}")

    np.save("raw_shifts_neutrino_petri_y10.npy", raw_shifts)
    np.save("sigma_shifts_neutrino_petri_y10.npy", sigma_shifts)



