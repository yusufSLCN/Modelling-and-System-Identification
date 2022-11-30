"""
This is the template for coding problems in exercise sheet 5, task 3.

Everywhere you see "YOUR CODE", it means a playground for you :P

WARNING: do not rename variables as this will break the tests.

Have fun!
"""

import matplotlib.pyplot as plt
import numpy as np


# Load data
data = np.load("exercise5_task3_data.npz")
X = data['X'] # X is of shape (N_e, N_m)
Y = data['Y'] # Y is of shape (N_e, N_m)

# array of the alphas
alphas = [0, 1e-6, 1e-5, 1]

# order of the fitting polynomials
order = 7

# derive useful constants
N_alpha = len(alphas) #
N_theta = order + 1  # number of parameters
N_e = np.size(X, 0)  # number of students/experiments
N_m = np.size(X, 1)  # number of measurements per experiment

## x and Phi used for plotting the polynomials
x_plotting = np.linspace(0, 1, 200) 
Phi_plotting = np.ones((200, order+1))
for o in range(order+1):
    Phi_plotting[:, o] = x_plotting.T**o


## Single experiment
plt.figure(1)
plt.plot(X[0,:], Y[0,:], "x")
plt.xlim([0, 1])
plt.ylim([-7, 7])
plt.xlabel("X")
plt.ylabel("Y")

# (a) For α ∈ {0, 10−6, 10−5, 1}, fit a polynomial of order 7 to the data of the first experiment. Plot the data and the fitted polynomials.
######## YOUR CODE ########
# define the regressor matrix of the LLS fit
Phi = np.zeros((len(X[0]), N_theta))
for i, x in enumerate(X[0]):
    Phi[i] = np.power(x, np.arange(N_theta))

###########################

# empty arrays to store the results
thetas_single = np.zeros((N_alpha, N_theta))
thetas_single_norm = np.zeros((N_alpha, 1))
R_squared = np.zeros((N_alpha, 1))

# iterate the alphas
for i in range(N_alpha):
    alpha = alphas[i]

    ######## YOUR CODE ########
    # solve the LLS problem to find the parameters of the fit
    # theta = np.polyfit(X[0], Y[0], order)
    theta = np.linalg.inv(Phi.T @ Phi + alpha * np.identity(Phi.shape[1])) @ Phi.T @ Y[0]
    ###########################

    # save values
    thetas_single[i, :] = theta

    ######## YOUR CODE ########
    # (b) For experiment 1 and for each α, compute the L2-norm of the estimated parameters
    # compute the norm and the R squared value
    thetas_single_norm[i] = np.linalg.norm(theta, 2)
    ###########################

    ######## YOUR CODE ########
    # (c) To compare the goodness of fit, compute the R2 values for each of the three fits obtained for experiment 1.
    error = Y[0] - theta @ Phi.T
    R_squared[i] = 1 - np.dot(error, error)/ np.dot(Y[0], Y[0])
    print(f'{alpha=}, R_squared={R_squared[i][0]:.2f}, 2_norm={thetas_single_norm[i][0]:.2f}')
    ###########################

    # add the plot of the fit to the figure
    plt.plot(x_plotting, Phi_plotting@theta)

# add a legend
plt.legend(["data", "no regularization",
            "small regularization", "strong regularization",
            "very strong regularization"],
            loc="lower left")


### (d) ####
# empty arrays to store the results
thetas = np.zeros((N_alpha, N_e, N_theta))
thetas_mean = np.zeros((N_alpha, N_theta))

# titles of the subplots
titles = ("no regularization", "small regularization", "strong regularization", "very strong")

plt.figure(2,figsize=(12,6))

# iterate the alphas
for i in range(N_alpha):
    alpha = alphas[i]
    
    # (d) For each α and each experiment, fit a polynomial of order 7. For each α, plot the fitted polynomials in a subplot

    # create a subplot 
    plt.subplot(N_alpha, 2, 2*i+1)
    plt.title(titles[i])
    plt.ylim([-6, 6])
    plt.xlim([0, 1])
    
    # iterate the experiments
    for k in range(N_e):

        ######## YOUR CODE ########
        # define the regressor matrix of the LLS fit
        Phi = np.zeros((len(X[k]), N_theta))
        for meas, x in enumerate(X[k]):
            Phi[meas] = np.power(x, np.arange(N_theta))

        # solve the LLS problem to find the parameters of the fit
        theta = np.linalg.inv(Phi.T @ Phi + alpha * np.identity(Phi.shape[1])) @ Phi.T @ Y[k]
        ###########################
        
        # store the result
        thetas[i,k,:] = theta

        # plot the fit
        plt.plot(x_plotting, Phi_plotting@theta, "-");  
    

    # Compute the average parameter vector for each α and plot the polynomial obtained from the averaged parameter vector.
    plt.subplot(N_alpha, 2, 2*i+2)
    plt.title(titles[i])
    plt.ylim([-6, 6])
    plt.xlim([0, 1])

    ######## YOUR CODE ########
    # compute the average parameter vector
    theta_mean = np.mean(thetas[i], axis=0)
    ###########################
    
    # store the results
    thetas_mean[i,:] = theta_mean
    
    # plot the average fit
    plt.plot(x_plotting, Phi_plotting@theta_mean, "-")

plt.tight_layout()
plt.show()
