
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.stats import norm
from scipy.special import factorial, erf

# Parameters for the simulation
N = 2000  # Number of particles
L = 200  # The left boundary for initial positions
r = 0.01  # Resetting rate
D = 0.5  # Diffusion coefficient
t = 1000  # Total time of system evolution
a, b, c = 1, 0.5, 0.001  # Parameters for state proposal  
steps = int(2e9)  # Number of MCMC steps ''' increase this if you want to increase accuracy'''
measurement_interval = 100 * N  # Measurement interval  ( We want to avoid correlation )
start_measurement_step = 20 * 100 * N  # Start measuring after 20 * 100 * N steps (Take this amount of time to thermalize)

# Defining Importance sampling method
@njit
def biased_metropolis_mcmc_njit(N, L, r, D, t, a, b, c, theta, steps, measurement_interval, start_measurement_step):
    # Initialize states for each particle
    x0 = np.random.uniform(-L, 0, N)    
    tau = np.random.exponential(1 / r, N)
    
    delta_x = np.zeros(N) 
    for i in range(N):
        # Compute standard deviation for delta_x
        std_dev = np.sqrt(2 * D * min(tau[i], t))
        delta_x[i] = np.random.normal(0, std_dev)

    # Initial net flux Q
    Q = np.sum(x0 + delta_x > 0)

    # Vector to store measurements of Q
    measurements = []

    for step in range(steps):
        # Sample a particle index
        i = np.random.randint(N)
        
        # changing x0 as proposed
        if x0[i] < -(L-a):
            x0_new = x0[i] + np.random.uniform(-L-x0[i], a)
        elif x0[i] > -a:
            x0_new = x0[i] + np.random.uniform(-a, -x0[i])
        else:
            x0_new = x0[i] + np.random.uniform(-a, a)
        
        # changing tau and delta_x as proposed
        tau_new = tau[i] + np.random.uniform(-b, b)
        delta_x_new = delta_x[i] + np.random.uniform(-c, c)

        # Calculate the new Q
        Q_new = Q - (x0[i] + delta_x[i] > 0) + (x0_new + delta_x_new > 0)

        # Calculate acceptance probability
        delta_x_squared_old = (delta_x[i]**2) / (4 * D * min(tau[i], t))
        delta_x_squared_new = (delta_x_new**2) / (4 * D * min(tau_new, t))
        exp_component = (-delta_x_squared_new + delta_x_squared_old - r * (tau_new - tau[i]) - theta * (Q_new - Q))
        p_acc = np.exp(exp_component)
        p_acc = min(1, p_acc)

        # Accept or reject the move
        if np.random.random() < p_acc:
            x0[i], tau[i], delta_x[i] = x0_new, tau_new, delta_x_new
            Q = Q_new

        # Measure the net flux Q every measurement_interval steps after start_measurement_step
        if step >= start_measurement_step and (step + 1) % measurement_interval == 0:
            measurements.append(Q)

    return measurements




theta_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
# You can take theta, as much as you want and you can reduce gap for mor accuracy 
measurements = [] # Measurement for each theta

for theta in theta_values:
    measurements.append(biased_metropolis_mcmc_njit(N, L, r, D, t, a, b, c, theta, steps, measurement_interval, start_measurement_step))


#Final modified Data

C_theta=0  #Initial Value of C_theta
C_theta_values=[0]  
Prob=[]
QQ=[]
measurements_theta_i = measurements[0]
theta_i = theta_values[0]
bins=np.arange(0,np.max(measurements_theta_i)+2) 
counts_theta_i, bins_theta_i = np.histogram(measurements_theta_i, bins=bins, density=True)
# Initially counts_theta_i is taking value for theta = 0


for i in range(1, 10): 
    measurements_theta_ip1 = measurements[i] #calling very next data set
    theta_ip1 = theta_values[i] # theta of new data set
    
    counts_theta_ip1, bins_theta_ip1 = np.histogram(measurements_theta_ip1, bins=bins, density=True)
    beens = bins[:-1] # this is the value all Q available

    prob_theta_i = counts_theta_i    #Just changing the name(no need to)
    prob_theta_ip1 = counts_theta_ip1  

    # Filter out all bins where original and biased both probability is not zero (so i can calculate log), basically taking all overlap region
    valid_indices = (prob_theta_i > 0) & (prob_theta_ip1 > 0)
    filtered_prob_theta_i = prob_theta_i[valid_indices] # original probability
    filtered_prob_theta_ip1 = prob_theta_ip1[valid_indices] # Biased probability
    filtered_Q_overlap = beens[valid_indices] # Value of Q

    '''Theoretically , I can calculate C_theta now, but to reduce errors, i am taking central part (Width is half of the original one) of the overlap region
        to calculate C_theta'''
    
    half_length = len(filtered_Q_overlap) // 2
    start_index = len(filtered_Q_overlap) // 4
    end_index = start_index + half_length

    # These are the central part of overlapped area
    middle_filtered_Q_overlap = filtered_Q_overlap[start_index:end_index] # value of Q
    middle_filtered_prob_theta_i = filtered_prob_theta_i[start_index:end_index] # original Probability
    middle_filtered_prob_theta_ip1 = filtered_prob_theta_ip1[start_index:end_index] # Biased Probability

    # Calculating C_theta
    log_prob_theta_ip1 = np.log(middle_filtered_prob_theta_ip1) # log of Biased Probability
    log_prob_theta_i = np.log(middle_filtered_prob_theta_i) # log of Original Probability
    C_theta = np.mean(log_prob_theta_ip1 - log_prob_theta_i + theta_ip1 * middle_filtered_Q_overlap)
    C_theta_values.append(C_theta)

    # I just want to know where BIASED Probability is not zero
    biased_index_valids= (prob_theta_ip1 > 0)
    biased_P = prob_theta_ip1[biased_index_valids] # Biased Probability
    biased_Q = beens[biased_index_valids] # Q Value where Biased Probability is not zero 

    # I want to know where UNBIASED Probability is not zero
    unbiased_index_valids = (prob_theta_i>0)
    unbiased_p = prob_theta_i[unbiased_index_valids] # Unbiased Probability
    unbiased_Q = beens[unbiased_index_valids] # Q value where Unbiased Probability is not zero

    
    index_no = int((len(middle_filtered_Q_overlap)+1)/2) # Middle index of overlapped region 
    ''' I am dividing overlapped region by middle index . one half will be treated as biased and other half will be treated as unbiased probability distribution'''

    # This is the biased probability region, starting from middle of overlapped region
    new_index_valids = np.where(biased_Q == middle_filtered_Q_overlap[index_no])[0][0]
    biased_valid_Q = biased_Q[:new_index_valids]
    biased_valid_p = biased_P[:new_index_valids]
    # This is the unbiased probability region, starting from middle of overlapped region
    last_part_of_index_valids = np.where(unbiased_Q == middle_filtered_Q_overlap[index_no])[0][0]
    unbiased_valid_Q = unbiased_Q [last_part_of_index_valids:]
    unbiased_valid_p = unbiased_p [last_part_of_index_valids:]

       
    log_prob_theta = np.log(biased_valid_p) - C_theta + theta_ip1 * biased_valid_Q # Converting biased region to unbiased region
    log_prob_theta=np.append(log_prob_theta,np.log(unbiased_valid_p)) # Merging new unbiased region and old unbiased region
    total_Q_valid=biased_valid_Q
    total_Q_valid=np.append(total_Q_valid,unbiased_valid_Q)
    ''' sO, WE have managed to convert biased distribution to unbiased distribution'''

    counts_theta_i = np.exp(log_prob_theta) # Now, this whole new distribution will be treated as unbiased distribution in the next loop
    zeros_to_add=np.zeros(len(bins[:-1])-len(counts_theta_i))
    counts_theta_i=np.append(zeros_to_add,counts_theta_i) 


# Theoretical 
def calculate_P(Q, u):
    return np.exp(-u) * (u**Q) / factorial(Q)
# Calculate rho
rho = N / L 
# Calculate u
u = rho * np.sqrt(D/r) * erf(np.sqrt(r*t)) /2
# Calculate P for each Q
P_values = [calculate_P(Q, u) for Q in range (0,60) ]
log_p=np.log(P_values)
plt.plot(range(0,60), log_p, label='Theoretical',color='orange') # Plotting theorertical

plt.scatter(total_Q_valid,log_prob_theta,label='Simulation') # Plotting simulation
plt.xlabel('Q(flux)')
plt.ylabel('ln(P)')
plt.legend()
plt.show()



    
