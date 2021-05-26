import numpy as np
import matplotlib.pyplot as plt



### Define the grid
print("Creating the Grid world, represented as:\n")
grid = GridWorld()

return_curves = []
v_curves = []

for x in range(100):
    Policy, V_function, return_curve, sarsa_historical_v_funcs = grid.do_SARSA(0.4, 0.3, 4000)
    return_curves.append(return_curve)
    v_curves.append(sarsa_historical_v_funcs)

sarsa_historical_v_funcs = np.average(v_curves, axis=0)
average_sarsa_curve = np.average(return_curves, axis=0) 


print(f'The mean of the dataset is: {np.mean(average_sarsa_curve)}')
print(f'The standard deviation of the dataset is: {np.std(average_sarsa_curve)}')


plt.plot(average_sarsa_curve)
plt.ylabel('Total Discounted Return')
plt.xlabel('Number of Episodes')
plt.show()

# Plot policy for policy iteration
print("\n\nThe optimal policy using policy iteration is:\n\n {}".format(Policy))
print("\n\nIts graphical representation is:\n")
grid.draw_deterministic_policy(np.array([np.argmax(Policy[row,:]) for row in range(grid.state_size)]))

# Plot value function for policy iteration
print("The value of the optimal policy computed using policy iteration is:\n\n {}".format(V_function))
print("\n\nIts graphical representation is:\n")
grid.draw_value(V_function)



### Define the grid
print("Creating the Grid world, represented as:\n")
grid = GridWorld()

return_curves = []
v_curves = []

for x in range(100):
    Policy, V_function, return_curve, mc_historical_v_funcs = grid.do_monteraclo(0.4, 2000)
    return_curves.append(return_curve)
    v_curves.append(mc_historical_v_funcs)

average_mc_curve = np.average(return_curves, axis=0) 
mc_historical_v_funcs = np.average(v_curves, axis=0)
                       
plt.plot(average_mc_curve)
plt.ylabel('Total Discounted Return')
plt.xlabel('Number of Episodes')
plt.show()

print(f'The mean of the dataset is: {np.mean(average_mc_curve)}')
print(f'The standard deviation of the dataset is: {np.std(average_mc_curve)}')
    
    
# Plot policy for policy iteration
print("\n\nThe optimal policy using policy iteration is:\n\n {}".format(Policy))
print("\n\nIts graphical representation is:\n")
grid.draw_deterministic_policy(np.array([np.argmax(Policy[row,:]) for row in range(grid.state_size)]))

# Plot value function for policy iteration
print("The value of the optimal policy computed using policy iteration is:\n\n {}".format(V_function))
print("\n\nIts graphical representation is:\n")
grid.draw_value(V_function)




# Policy iteration algorithm
V_opt, pol_opt, epochs, iteration_historical_v_functions = grid.policy_iteration(0.4, 0.00001)

# Plot value function for policy iteration
print("The value of the optimal policy computed using policy iteration is:\n\n {}".format(V_opt))
print("\n\nIts graphical representation is:\n")
grid.draw_value(V_opt)



# calculate RMS error for both lists
sarsa_rms_error = []
mc_rms_error = []

for sarsa, mc in zip(sarsa_historical_v_funcs[:400], mc_historical_v_funcs[:400]):
    mc_error = iteration_historical_v_functions[-1] - mc
    sarsa_error = iteration_historical_v_functions[-1] - sarsa
    
    mc_error = mc_error**2
    sarsa_error = sarsa_error**2
    
    mc_sum = np.sum(mc_error)
    sarsa_sum = np.sum(sarsa_error)

    mc_sum = np.sqrt(mc_sum)
    sarsa_sum = np.sqrt(sarsa_sum)
    
    sarsa_rms_error.append(sarsa_sum)
    mc_rms_error.append(mc_sum)

plt.plot(sarsa_rms_error)
plt.plot(mc_rms_error)
plt.ylabel('RMS Error')
plt.xlabel('Number of Episodes')
plt.legend(['SARSA', 'Monte Carlo'], loc='upper right')

plt.show()


plt.plot(sarsa_rms_error, average_sarsa_curve[:400],'ro')
plt.plot(mc_rms_error, average_mc_curve[:400], 'bo')
plt.ylabel('Return')
plt.xlabel('RMS Error')
plt.legend(['SARSA', 'Monte Carlo'], loc='upper right')

plt.show()
