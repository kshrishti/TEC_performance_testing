import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve

# Clearing MATLAB specific settings
plt.close('all')

# Symbolic variables
l, k1, a, g, Th, Tc, Vc, Vh, j2, Tinf, qh, h, C2, j1 = symbols('l k1 a g Th Tc Vc Vh j2 Tinf qh h C2 j1')

# Constants
ld = 1.524e-3
k1d = 1.612
ad = 1.941e-4
Tcd = 25 + 273.15
Thd = 25 + 273.15
Vcd = 0
Vhd = 0.058
gd = 8.422e+04
vals = [ld, k1d, ad, gd, Vhd]

C1 = j1**2 / g / k1 * l / 2
eq1 = -j1 / g * l + a * j1**2 / (g * k1) * l**2 / 2 - a * C1 * l - Vh
sol = solve(eq1.subs(list(zip([l, k1, a, g, Vh], vals))), j1)

x0 = 0.0
xf = ld
xn = 100
c = 0
j = [s for s in sol if s < 0][0]
C1d = j**2 / gd / k1d * ld / 2

x_vals = np.linspace(x0, xf, xn + 1)
Tv_vals = []
Vv_vals = []
xv_vals = []

for x in x_vals:
    c += 1
    Tv = -j**2 / gd / k1d * x**2 / 2 + C1d * x + Tcd
    Vv = -j / gd * x + ad * j**2 / gd / k1d * x**2 / 2 - ad * C1d * x
    Tv_vals.append(Tv)
    Vv_vals.append(Vv)
    xv_vals.append(x)

# Plotting
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(xv_vals, Vv_vals, linewidth=2, color='blue')
ax1.set_ylabel('ΔV[V]', color='blue')
ax1.set_xlabel('x')
ax1.tick_params(axis='y', labelcolor='blue')

ax2.plot(xv_vals, np.array(Tv_vals) - 273.15, linewidth=2, color='red')
ax2.set_ylabel('Tc[ºC]', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.xlim([0, ld])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()

plt.show()
