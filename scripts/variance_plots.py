
import os, sys, math

import numpy as np
import matplotlib.pyplot as plt
import colorsys
import sys
import random
import multiprocess
from tqdm import tqdm
from functools import partial


#############################################################################
# Reproduces the variance plots for importance sampling of the EON model
#############################################################################

# assume rho = 1
def f_lambert():
    return 1.0/math.pi

constant1_FON = 0.5     - 2.0 /(3.0 *math.pi)
constant2_FON = 2.0/3.0 - 28.0/(15.0*math.pi)

# assume rho = 1
def f_FON(roughness, wi_local, wo_local):

    sigma = roughness                                    # FON sigma prime
    mu_i = wi_local[2]                                   # input angle cos
    mu_o = wo_local[2]                                   # output angle cos
    s = np.dot(wi_local, wo_local) - mu_i * mu_o         # QON s term
    if s > 0.0:                                          # FON s/t
        sovertF = s / max(mu_i, mu_o)
    else:
        sovertF = s

    AF = 1.0 / (1.0 + constant1_FON * sigma)             # FON A coeff.
    return (1.0/math.pi) * AF * (1.0 + sigma * sovertF)  # single-scatter

def E_FON_analyt(mu, roughness):

    sigma = roughness;                                   # FON sigma prime
    AF = 1.0 / (1.0 + constant1_FON * sigma);            # FON A coeff.
    BF = sigma * AF;                                     # FON B coeff.
    Si = math.sqrt(1.0 - (mu * mu))
    G = Si * (math.acos(mu) - Si*mu) + (2.0/3.0) * ((Si/mu)*(1.0 - (Si*Si*Si)) - Si)
    return AF + (BF/math.pi) * G

def f_EON(roughness, wi_local, wo_local):

    sigma = roughness                                    # FON sigma prime
    mu_i = wi_local[2]                                   # input angle cos
    mu_o = wo_local[2]                                   # output angle cos
    s = np.dot(wi_local, wo_local) - mu_i * mu_o         # QON s term
    if s > 0.0:                                          # FON s/t
        sovertF = s / max(mu_i, mu_o)
    else:
        sovertF = s
    constant1_FON = 0.5 - 2.0/(3.0*math.pi)
    AF = 1.0 / (1.0 + constant1_FON * sigma)             # FON A coeff.
    f_ss = (1.0/math.pi) * AF * (1.0 + sigma * sovertF)  # single-scatter
    EFo = E_FON_analyt(mu_o, sigma)
    EFi = E_FON_analyt(mu_i, sigma)                      # FON wi albedo (analyt)
    avgEF = AF * (1.0 + constant2_FON * sigma)           # avg. albedo
    rho_ms = 1.0 #avgEF / (1.0 - rho*(1.0 - avgEF))
    eps = 1.0e-7
    f_ms = (rho_ms/math.pi) * max(eps, 1.0-EFo) * max(eps, 1.0-EFi) / max(eps, 1.0-avgEF)
    return f_ss + f_ms

def drawProgressBar(fraction, barLen=40):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * fraction):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("\t[ %s ] %.2f%%" % (progress, fraction * 100))
    if fraction==1.0: sys.stdout.write("\n")
    sys.stdout.flush()

def normalize(wi_local):
    mag_sqr = np.dot(wi_local, wi_local)
    return wi_local / math.sqrt(mag_sqr)

# uniform
def sample_uniform(wo_local, roughness, u1, u2):

    sin_theta = math.sqrt(1.0 - u1*u1)
    phi = 2.0 * math.pi * u2
    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = u1
    pdf = 1.0/(2.0*math.pi)
    return (np.array([x, y, z]), pdf)

# cosine weighted
def sample_cosine_weighted_hemisphere(wo_local, roughness, u1, u2):

    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - x*x - y*y))
    pdf = max(1.0e-7, abs(z) / math.pi)
    return (np.array([x, y, z]), pdf)


# LTC (d=0)
def ltc_terms_positive(x, y):

    ########################################################################################################################
    # LTC coefficients
    # Assume the LTC transformation matrix, applied to the cosine-hemisphere-sampled directions,
    # is of the form:
    #
    #                 [[a   0    b]
    #             M =  [0   c    0]
    #                  [0   0    1]]
    #
    # The corresponding inverse is given by:
    #
    #                    [[c       0       -b*c  ]
    #        M^{-1}  =    [0       a        0    ]   /  det(M)   where det(M) = c*a
    #                     [0       0       a*c  ]]
    #
    a = 1.0 + y*(0.303392 + (-0.518982 + 0.111709*x)*x + (-0.276266 + 0.335918*x)*y)
    b = y*(-1.16407 + 1.15859*x + (0.150815 - 0.150105*x)*y)/(-1.43545 + x*x*x)
    c = 1.0 + (0.20013 + (-0.506373 + 0.261777*x)*x)*y
    d = 0
    return (a, b, c, d)

def sample_ltc_positive(wo_local, roughness, u1, u2):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms_positive(cos_o, roughness)
    detM = c*(a - b*d)
    ########################################################################################################################

    # thus sample from LTC lobe
    sample = sample_cosine_weighted_hemisphere(wo_local, roughness, u1, u2)
    wi_cos_sample         = sample[0]                                    # wo
    cosine_hemisphere_pdf = sample[1]                                    # Do(wo)
    wi_ltc_sample = np.array([a*wi_cos_sample[0] + b*wi_cos_sample[2],
                              c*wi_cos_sample[1],
                              d*wi_cos_sample[0] + wi_cos_sample[2]])    # M wo
    length = 1.0 / math.sqrt(np.dot(wi_ltc_sample, wi_ltc_sample))       # ||M^-1 w|| = 1 / ||M wo||
    wi_ltc_sample = normalize(wi_ltc_sample)
    if wi_ltc_sample[2] < 0.0:
        print('should not happen')
        quit()

    inverse_determinant = 1.0/detM                                       # det(M^-1)
    jacobian = inverse_determinant / (length**3.0)                       # det(M^-1) / ||M^-1 w||^3
    pdf = cosine_hemisphere_pdf * jacobian                               # D(w), equation (1) from LTC paper

    return (wi_ltc_sample, pdf)

def positive_ltc_pdf(wo_local, wi_local, roughness):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms_positive(cos_o, roughness)
    detM = c*(a - b*d)

    ltc = np.array([[a, 0, b],
                    [0, c, 0],
                    [d, 0, 1]])
    ltcInv = np.linalg.inv(ltc)

    wo = ltcInv @ wi_local
    lensq = np.dot(wo, wo)

    vz = 1.0 / math.sqrt(d*d + 1.0)
    s = 0.5 * (1.0 + vz)
    pdfCosine = max(wo[2], 0.0) / (math.pi * s)

    det = 1.0 / detM
    pdf = pdfCosine * det / (lensq * lensq)

    return pdf


# LTC (more general, d != 0)
def ltc_terms(cos_o, roughness):

    ########################################################################################################################
    # LTC coefficients
    # Assume the LTC transformation matrix, applied to the cosine-hemisphere-sampled directions,
    # is of the form:
    #
    #                 [[a   0    b]
    #             M =  [0   c    0]
    #                  [d   0    1]]
    #
    # The corresponding inverse is given by:
    #
    #                    [[c       0       -b*c  ]
    #        M^{-1}  =    [0       a-b*d    0    ]   /  det(M)   where det(M) = c*(a - b*d)
    #                     [-c*d     0       a*c  ]]
    #
    mu = cos_o
    r = roughness
    a = 1.0 + r*(0.303392 + (-0.518982 + 0.111709*mu)*mu + (-0.276266 + 0.335918*mu)*r)
    b = r*(-1.16407 + 1.15859*mu + (0.150815 - 0.150105*mu)*r)/(-1.43545 + mu*mu*mu)
    c = 1.0 + (0.20013 + (-0.506373 + 0.261777*mu)*mu)*r
    d = ((0.540852 + (-1.01625 + 0.475392*mu)*mu)*r)/(-1.0743 + mu*(0.0725628 + mu))
    return (a, b, c, d)


def sample_ltc(wo_local, roughness, u1, u2):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms(cos_o, roughness)
    detM = c*(a - b*d)
    ########################################################################################################################

    # thus sample from LTC lobe
    sample = sample_cosine_weighted_hemisphere(wo_local, roughness, u1, u2)
    wi_cos_sample         = sample[0]                                    # wo
    cosine_hemisphere_pdf = sample[1]                                    # Do(wo)
    wi_ltc_sample = np.array([a*wi_cos_sample[0] + b*wi_cos_sample[2],
                              c*wi_cos_sample[1],
                              d*wi_cos_sample[0] + wi_cos_sample[2]])    # M wo
    length = 1.0 / math.sqrt(np.dot(wi_ltc_sample, wi_ltc_sample))       # ||M^-1 w|| = 1 / ||M wo||
    wi_ltc_sample = normalize(wi_ltc_sample)
    if wi_ltc_sample[2] < 0.0:
        return (wi_ltc_sample, 0.0)

    inverse_determinant = 1.0/detM                                       # det(M^-1)
    jacobian = inverse_determinant / (length**3.0)                       # det(M^-1) / ||M^-1 w||^3
    pdf = cosine_hemisphere_pdf * jacobian                               # D(w), equation (1) from LTC paper

    return (wi_ltc_sample, pdf)


def copysign(x):
    return 1.0 if x >= 0.0 else -1.0

def mix(a, b, t):
    return a + (b - a)*t

def sample_clipped_cosine(u1, u2, d):

    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    vz = 1.0 / math.sqrt(d*d + 1.0)
    s = 0.5 * (1.0 + vz)
    x = copysign(d) * mix(math.sqrt(1.0 - y*y), x, s)
    z = math.sqrt(max(1.0 - x*x - y*y, 0.0))
    return (np.array([x, y, z]), z / (math.pi * s))


# LTC clipped
def sample_clipped_ltc(wo_local, roughness, u1, u2):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms(cos_o, roughness)
    detM = c*(a - b*d)

    sample = sample_clipped_cosine(u1, u2, d)
    wi_cos_sample         = sample[0]                                    # wo
    cosine_hemisphere_pdf = sample[1]                                    # Do(wo)
    wi_ltc_sample = np.array([a*wi_cos_sample[0] + b*wi_cos_sample[2],
                              c*wi_cos_sample[1],
                              d*wi_cos_sample[0] + wi_cos_sample[2]])    # M wo
    length = 1.0 / math.sqrt(np.dot(wi_ltc_sample, wi_ltc_sample))       # ||M^-1 w|| = 1 / ||M wo||
    wi_ltc_sample = normalize(wi_ltc_sample)

    inverse_determinant = 1.0/detM                                       # det(M^-1)
    jacobian = inverse_determinant / (length**3.0)                       # det(M^-1) / ||M^-1 w||^3
    pdf = cosine_hemisphere_pdf * jacobian                               # D(w), equation (1) from LTC paper

    return (wi_ltc_sample, pdf)


def clipped_ltc_pdf(wo_local, wi_local, roughness):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms(cos_o, roughness)
    detM = c*(a - b*d)
    ltc = np.array([[a, 0, b],
                    [0, c, 0],
                    [d, 0, 1]])
    ltcInv = np.linalg.inv(ltc)
    wh = ltcInv @ wi_local
    lensq = np.dot(wh, wh)
    vz = 1.0 / math.sqrt(d*d + 1.0)
    s = 0.5 * (1.0 + vz)
    pdfCosine = max(wh[2], 0.0) / (math.pi * s)
    det = 1.0 / detM
    pdf = pdfCosine * det / (lensq * lensq)
    return pdf

def ltc_pdf(wo_local, wi_local, roughness):

    cos_o = wo_local[2]
    (a, b, c, d) = ltc_terms(cos_o, roughness)
    detM = c*(a - b*d)
    ltc = np.array([[a, 0, b],
                    [0, c, 0],
                    [d, 0, 1]])
    ltcInv = np.linalg.inv(ltc)
    wh = ltcInv @ wi_local
    lensq = np.dot(wh, wh)
    pdfCosine = max(wh[2], 0.0) / math.pi
    det = 1.0 / detM
    pdf = pdfCosine * det / (lensq * lensq)
    return pdf


# LTC clipped with MIS
def sample_clipped_ltc_mis(wo_local, roughness, u1, u2):

    mu = wo_local[2]
    probU = (roughness**0.1) * (0.162925 + mu*(-0.372058 + (0.538233 - 0.290822*mu)*mu))
    probC = 1.0 - probU
    pdfU = 1.0 / (2.0 * math.pi)

    if u1 <= probU:
        u1 = u1 / probU
        wPDF = sample_uniform(wo_local, roughness, u1, u2)
        pdfC = clipped_ltc_pdf(wo_local, wPDF[0], roughness)
    else:
        u1 = (u1 - probU) / probC
        wPDF = sample_clipped_ltc(wo_local, roughness, u1, u2)
        pdfC = wPDF[1]

    return (wPDF[0], probU*pdfU + probC*pdfC)



###############################################################
# sampling testbed
###############################################################

NUM_THREADS = 8

Nsamples = int(1.0e6)
Ntheta = 64
theta_o_array = math.pi/2.0 * np.linspace(0.0, 0.999, Ntheta)

f_varian = plt.figure()
ax_varian = f_varian.add_subplot(111)

f_weight = plt.figure()
ax_weight = f_weight.add_subplot(111)

f_max_weight = plt.figure()
ax_max_weight = f_max_weight.add_subplot(111)

Nroughnesses = 5
roughnesses = np.linspace(0.0, 1.0, Nroughnesses)

def body(SAMPLE_METHOD, roughness, n_theta):

    theta_o = theta_o_array[n_theta]
    wo_local = np.array([math.sin(theta_o), 0.0, math.cos(theta_o)])

    # Draw samples from PDF given incident angle and roughness
    Ewi  = 0.0 # expectation value of weight\
    Ewi2 = 0.0 # expectation value of weight^2
    max_weight = 0.0
    local_random = random.Random(n_theta)

    for n_sample in range(0, Nsamples):

        r1 = local_random.random()
        r2 = local_random.random()

        sample = SAMPLE_METHOD(wo_local, roughness, r1, r2)
        wi_local = sample[0]
        pdf      = sample[1]
        if pdf == 0.0:
            continue
        f = f_EON(roughness, wi_local, wo_local)
        mu_i = wi_local[2]
        xi = mu_i * f / pdf
        Ewi  += xi
        Ewi2 += xi*xi
        max_weight = max(max_weight, xi)

    Ewi  /= float(Nsamples)
    Ewi2 /= float(Nsamples)
    Var = Ewi2 - Ewi*Ewi # population variance of weight

    return (Ewi, Var, max_weight)


def draw_and_plot_samples(SAMPLE_METHOD, hue, sat, style, label_str):

    for n_r in reversed(range(Nroughnesses)):

        roughness = roughnesses[n_r]

        func = partial(body, SAMPLE_METHOD, roughness)
        pool = multiprocess.Pool(NUM_THREADS)
        results = list(tqdm(pool.imap(func, range(0, Ntheta)), total=Ntheta))
        del pool

        weight_array     = [t[0] for t in results]
        variance_array   = [t[1] for t in results]
        max_weight_array = [t[2] for t in results]

        legend_str = r'$r$=%3.2f (%s)' % (roughness, label_str)
        grayscale = 1.0 - 0.8*(1.0-roughness)
        Clin = colorsys.hsv_to_rgb(hue, grayscale, 0.2+0.8*grayscale)

        ax_varian.plot(   theta_o_array,  variance_array,   label=legend_str, marker='', linewidth=1.0, linestyle=style, color=Clin)
        ax_weight.plot(   theta_o_array,  weight_array,     label=legend_str, marker='', linewidth=1.0, linestyle=style, color=Clin)
        ax_max_weight.plot(theta_o_array, max_weight_array, label=legend_str, marker='', linewidth=1.0, linestyle=style, color=Clin)


if __name__ == '__main__':

    print('sample_cosine_weighted_hemisphere...')
    draw_and_plot_samples(sample_cosine_weighted_hemisphere, 0.0, 1.0, 'dashed', 'cos weighted')

    print('sample_clipped_ltc_mis...')
    draw_and_plot_samples(sample_clipped_ltc_mis, 0.333, 1.0, 'solid',  'CLTC + uniform, MIS')

    ax_varian.set_xlim(0.0, math.pi/2.0)
    ax_varian.set_yscale('log')
    ax_varian.set_xlabel (r'Output angle $\theta_o$')
    ax_varian.set_ylabel (r'Throughput weight variance')
    leg_v = f_varian.legend(loc="upper left", borderaxespad=9, fontsize="7")
    bb = leg_v.get_bbox_to_anchor().transformed(ax_varian.transAxes.inverted())
    xOffset = 0.02
    yOffset = 0.04
    bb.x0 += xOffset
    bb.x1 += xOffset
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg_v.set_bbox_to_anchor(bb, transform = ax_varian.transAxes)
    f_varian.savefig('importance_sampling_variance_vs_roughness.pdf')

    ax_weight.set_xlim(0.0, math.pi/2.0)
    ax_weight.set_xlim(0.0, 1.1)
    ax_weight.set_xlabel (r'Output angle $\theta_o$')
    ax_weight.set_ylabel (r'Throughput weight')
    leg_w = f_weight.legend(loc="lower left", borderaxespad=8, fontsize="7")
    bb = leg_w.get_bbox_to_anchor().transformed(ax_weight.transAxes.inverted())
    xOffset = 0.01
    yOffset = 0.04
    bb.x0 += xOffset
    bb.x1 += xOffset
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg_w.set_bbox_to_anchor(bb, transform = ax_weight.transAxes)
    f_weight.savefig('importance_sampling_weight_vs_roughness.pdf')

    ax_max_weight.set_xlim(0.0, math.pi/2.0)
    ax_max_weight.set_yscale('log')
    ax_max_weight.set_xlabel (r'Output angle $\theta_o$')
    ax_max_weight.set_ylabel (r'Throughput maximum weight')
    leg_w = f_max_weight.legend(loc="upper left", borderaxespad=8, fontsize="7")
    bb = leg_w.get_bbox_to_anchor().transformed(ax_max_weight.transAxes.inverted())
    xOffset = 0.02
    yOffset = 0.04
    bb.x0 += xOffset
    bb.x1 += xOffset
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg_w.set_bbox_to_anchor(bb, transform = ax_max_weight.transAxes)
    f_max_weight.savefig('importance_sampling_max_weight_vs_roughness.pdf')

    plt.show()


