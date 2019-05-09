from scipy import linalg
import numpy as np
import matplotlib.cm as cm
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt
# %matplotlib inline

# == Set up the Gaussian prior density p == #
Σ = [[0.3**2, 0.0], [0.0, 0.3**2]]
Σ = np.matrix(Σ)
x_hat 	 = np.matrix([0.5, -0.5]).T
# == Define the matrices G and R from the equation y = G x + N(0, R) == #
G = [[1, 0], [0, 1]]
G = np.matrix(G)
R = 0.5 * Σ
# == The matrices A and Q == #
A = [[1.0, 0], [0, 1.0]]
A = np.matrix(A)
Q = 0.3 * Σ
# == The observed value of y == #
y = np.matrix([2.3, -1.9]).T

# == Set up grid for plotting == #
x_grid = np.linspace(-1.5, 2.9, 100)
y_grid = np.linspace(-3.1, 1.7, 100)
X, Y = np.meshgrid(x_grid, y_grid)


def gen_gaussian_plot_vals(μ, C):
    "Z values for plotting the bivariate Gaussian N(μ, C)"
    m_x, m_y = float(μ[0]), float(μ[1])
    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    s_xy = C[0, 1]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)

fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()
 
# Plot the figure
# Density 1
Z = gen_gaussian_plot_vals(x_hat, Σ)
cs1 = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs1, inline=1, fontsize=10)




# Density 2
#Kalman Gain
K = Σ * G.T * linalg.inv(G * Σ * G.T + R)
# Update the state estimate
x_hat_F = x_hat + K*(y - G * x_hat)
#update covariance estimation
Σ_F = Σ - K * G * Σ
Z_F = gen_gaussian_plot_vals(x_hat_F, Σ_F)
cs2 = ax.contour(X, Y, Z_F, 6, colors="black")
ax.clabel(cs2, inline=1, fontsize=10)

# Density 3
# Predict next state of the feature with the last state and predicted motion
#https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
new_x_hat = A * x_hat_F 
# print(new_x_hat)
#predict next covariance
new_Σ = A * Σ_F * A.T + Q
new_Z = gen_gaussian_plot_vals(new_x_hat, new_Σ)
cs3 = ax.contour(X, Y, new_Z, 6, colors="black")
ax.clabel(cs3, inline=1, fontsize=10)
ax.contourf(X, Y, new_Z, 6, alpha=0.6, cmap=cm.jet)
ax.text(float(y[0]), float(y[1]), "$y$", fontsize=20, color="black")

plt.show()
dt = 33.3e-3
#state update matrices
A = np.matrix( ((1, 0, dt, 0),(0, 1, 0, dt),(0, 0, 1, 0),(0, 0, 0, 1)) )
Q = np.matrix( (30, 68, 0, 0) ).transpose()
B = np.matrix( ((dt**2/2),(dt**2/2), dt, dt)).transpose()
C = np.matrix( ((1,0,0,0),(0,1,0,0)) ) #this is our measurement function C, that we apply to the state estimate Q to get our expect next/new measurement
Q_estimate = Q
u = .005 #define acceleration magnitude

marker_noise_mag = .1; #process noise: the variability in how fast the Hexbug is speeding up (stdv of acceleration: meters/sec^2)
tkn_x = 1;  #measurement noise in the horizontal direction (x axis).
tkn_y = 1;  #measurement noise in the horizontal direction (y axis).
Ez = np.matrix(((tkn_x,0),(0,tkn_y))) 
Ex = np.matrix( ((dt**4/4,0,dt**3/2,0),(0,dt**4/4,0,dt**3/2),(dt**3/2,0,dt**2,0),(0,dt**3/2,0,dt**2)) )*marker_noise_mag**2# Ex convert the process noise (stdv) into covariance matrix
P = Ex; # estimate of initial Hexbug position variance (covariance matrix)

# Predict next state of the Hexbug with the last state and predicted motion.
Q_estimate = A*Q_estimate + B*u;
# predic_state = [predic_state; Q_estimate(1)] ;
# predict next covariance
P = A*P*A.T + Ex;
# predic_var = [predic_var; P] ;
# predicted Ninja measurement covariance
# Kalman Gain
K = P*C.T*linalg.inv(C*P*C.T + Ez);
# Update the state estimate
x_avg = 32
y_avg = 70
Q_loc_meas = np.matrix( (x_avg, y_avg) ).transpose()
Q_estimate = Q_estimate + K * (Q_loc_meas - C*Q_estimate);
print(Q_estimate)
# update covariance estimation.
P = (np.identity(4) - K*C)*P;

import csv

float_list = [1.13, 0.25, 3.28]

# with open('ANN_0.csv', "w") as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerow(Ez)

outfile = open('./ANN_0.csv','w')
writer=csv.writer(outfile)
writer.writerow(float_list)
# writer.writerow(['SNo', 'States', 'Dist', 'Population'])
# writer.writerows(list_of_rows)
writer.writerow(float_list)
writer.writerow(float_list)
writer.writerow(float_list)
writer.writerow(float_list)