
import os, sys, math

import numpy as np
import matplotlib.pyplot as plt
import colorsys

##############################################################################################
# Verifies that the analytical formulas for the FON model albedo match the numerical integral
##############################################################################################

def fON_fujii(sigma,
              theta_i, phi_i,
              theta_o, phi_o):

    cosThetaI = np.cos(theta_i)
    sinThetaI = np.sin(theta_i)
    cosThetaO = np.cos(theta_o)
    sinThetaO = np.sin(theta_o)
    cosPhiI = np.cos(phi_i)
    sinPhiI = np.sin(phi_i)
    cosPhiO = np.cos(phi_o)
    sinPhiO = np.sin(phi_o)
    cosPhiDiff = cosPhiI*cosPhiO + sinPhiI*sinPhiO

    C = 0.5 - 2.0/(3.0*math.pi)
    A = 1.0 / (1.0 + C * sigma)
    B = sigma * A

    s = cosPhiDiff * sinThetaI * sinThetaO
    if s <= 0.0:
        tinv = 1.0
    else:
        tinv = 1.0/max(cosThetaO, cosThetaI)

    return (1.0 / math.pi) * (A + B * s * tinv)


# Angular ranges
Ntheta  = 16
Nphi    = 256
theta_array = np.linspace(0.0, 1.0, Ntheta)
phi_array   = np.linspace(0.0, math.pi * 2.0, Nphi)

# Compute albedo numerically
def albedo_numerical(sigma, theta_o, phi_o):

    #  Integrate over theta_i
    #  Do solid angle integral for:
    #    pho(theta_o, phi_o) = \int_0^{pi/2} dtheta_i sin(theta_i) cos(theta_i)
    #                          \int_0^{2pi}  dphi_i   fON(theta_i, phi_i, theta_o, phi_o)
    Ntheta_integral = 64
    Integrand = np.empty(Ntheta_integral)
    theta_array_integral = np.linspace(0.0, 1.0, Ntheta_integral)

    # evaluate Integrand for theta_i integral
    for n_theta_i in range(Ntheta_integral):

        theta_i = (math.pi/2.0) * theta_array_integral[n_theta_i]
        bsdf = np.empty(Nphi) # integrand for phi integral

        # do integral over phi_i
        for n_phi_i in range(Nphi):
            phi_i = phi_array[n_phi_i]
            bsdf[n_phi_i] = fON_fujii(sigma, theta_i, phi_i, theta_o, phi_o)

        phi_integral = np.trapz(bsdf, phi_array)
        Integrand[n_theta_i] = phi_integral * np.sin(theta_i) * np.cos(theta_i)

    # do integral over theta_i
    theta_integral = np.trapz(Integrand, (math.pi/2.0) * theta_array_integral)
    return theta_integral


# Compute albedo analytically, via exact formula
def albedo_analytical(r, theta_o):

    cosThetaO = np.cos(theta_o)
    sinThetaO = np.sin(theta_o)
    C = 0.5 - 2.0/(3.0*math.pi)
    A = 1.0 / (1.0 + C * r)
    B = r * A
    X = sinThetaO * (theta_o - sinThetaO*cosThetaO) + 2.0/3.0*(sinThetaO/cosThetaO)*(1.0 - sinThetaO**3.0) - 2.0*sinThetaO/3.0
    return A + B*X/math.pi

# Compute albedo analytically, via approximate formula
def albedo_approx(r, theta_o):

    mu = np.cos(theta_o)
    mucomp = 1.0 - mu
    GoverPi = 0.0571085289*mucomp + 0.491881867*mucomp**2.0 - 0.332181442*mucomp**3.0 + 0.0714429953*mucomp**4.0
    C = 0.5 - 2.0/(3.0*math.pi)
    return (1.0 + r*GoverPi) / (1.0 + C*r)


Nroughness = 11
roughnesses = np.linspace(0.0, 1.0, Nroughness)

# directional albedo rho(theta_o)
rhos_numerical  = np.empty(Ntheta)
for n_r in range(Nroughness):
    r = roughnesses[n_r]
    print('Running numerics for r = %f' % r)
    for n_theta_o in range(Ntheta):
        theta_o = (math.pi/2.0) * theta_array[n_theta_o]
        rhos_numerical[n_theta_o]  = albedo_numerical(r, theta_o, 0.0)
    label_str = r'$r$=%3.2f' % r
    grayscale = r / roughnesses[-1]
    Clin = colorsys.hsv_to_rgb(1, 0.9, grayscale)
    plt.plot(theta_array, rhos_numerical,  label=label_str, color=Clin, linestyle='', marker='.', markersize=4)

Ntheta_anal = int(Ntheta*16)
theta_array_analytical = np.linspace(0.0, 1.0, Ntheta_anal)
rhos_analytical = np.empty(Ntheta_anal)
for n_r in range(Nroughness):
    r = roughnesses[n_r]
    print('Running theory for r = %f' % r)
    for n_theta_o in range(Ntheta_anal):
        theta_o = (math.pi/2.0) * theta_array_analytical[n_theta_o]
        rhos_analytical[n_theta_o] = albedo_approx(r, theta_o) # albedo_analytical(r, theta_o)
    grayscale = r / roughnesses[-1]
    Clin = colorsys.hsv_to_rgb(1, 0.9, grayscale)
    plt.plot(theta_array_analytical, rhos_analytical, label='', color=Clin, linewidth=0.75)

plt.ylim(0, 1)
plt.ylim(0.5, 1.01)
plt.xlabel (r'Output angle $\theta_o$ / 90$^{\circ}$')
plt.ylabel (r'Albedo $E_\mathrm{F}(\theta_o)$')
plt.legend(loc="lower right")
plt.savefig('FON_albedo.pdf')
plt.show()