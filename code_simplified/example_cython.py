import argus_cy # this is the compiled extension module
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# use Algorithm 2 to generate samples for different chi
x = np.linspace(1e-5, 0.999, num=100)
gen = argus_cy.argus_pinv()
for chi in [1e-6, 0.001, 0.05, 0.5, 3.5, 7]:
    r = gen.rvs(chi, size=10000)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

# give an example of how to sample for different chi in one go
# (varying parameter case)
chis = [0.1, 1.3, 3.5]
chi_arr = np.array(chis) * np.ones((1000, 3))
y = gen.rvs_varying(chi_arr)

# first column contains the rvs for chi=0.1, second for 1.3, third for 3.5
for i, chi in enumerate(chis):
    plt.hist(y[:, i], bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()

# another example of the varying parameter case
chis = [0.1, 2, 6]
r = gen.rvs_varying(chis*1000)
for i in range(3):
    plt.hist(r[i::3], bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chis[i]))
    plt.show()

# generate samples with Ratio-Of-Uniforms
for chi in [0.75, 2.1]:
    r = argus_cy.rvs_rou_shifted_gamma_cy(chi, size=10000)
    plt.hist(r, bins=20, density=True)
    plt.plot(x, stats.argus.pdf(x, chi))
    plt.show()
