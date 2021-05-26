import numpy as np
import matplotlib.pyplot as plt
from tabulated_rl import GridWorld




# Run SARSA
# *********
sarsa_grid = GridWorld()
return_curves = []
v_curves = []

for x in range(100):
    Policy, V_function, return_curve, sarsa_historical_v_funcs = sarsa_grid.do_SARSA(0.4, 0.3, 500)
    return_curves.append(return_curve)
    v_curves.append(sarsa_historical_v_funcs)

sarsa_historical_v_funcs = np.average(v_curves, axis=0)
average_sarsa_curve = np.average(return_curves, axis=0) 
print(f'The mean of the dataset is: {np.mean(average_sarsa_curve)}')
print(f'The standard deviation of the dataset is: {np.std(average_sarsa_curve)}')

plt.plot(average_sarsa_curve)
plt.ylabel('Total Discounted Return')
plt.xlabel('Number of Episodes')
plt.savefig("Result_images/SARSA_results.png")
plt.close()

sarsa_grid.draw_deterministic_policy(np.array([np.argmax(Policy[row,:]) for row in range(sarsa_grid.state_size)]))
sarsa_grid.draw_value(V_function)



# Run Monte Carlo
# ***************
MC_grid = GridWorld()
return_curves = []
v_curves = []

for x in range(100):
    Policy, V_function, return_curve, mc_historical_v_funcs = MC_grid.do_monteraclo(0.4, 500)
    return_curves.append(return_curve)
    v_curves.append(mc_historical_v_funcs)

average_mc_curve = np.average(return_curves, axis=0) 
mc_historical_v_funcs = np.average(v_curves, axis=0)
                       
plt.plot(average_mc_curve)
plt.ylabel('Total Discounted Return')
plt.xlabel('Number of Episodes')
plt.savefig("Result_images/Monte_Carlo_results.png")
plt.close()

print(f'The mean of the dataset is: {np.mean(average_mc_curve)}')
print(f'The standard deviation of the dataset is: {np.std(average_mc_curve)}')
MC_grid.draw_deterministic_policy(np.array([np.argmax(Policy[row,:]) for row in range(MC_grid.state_size)]))
MC_grid.draw_value(V_function)



# Run Policy iteration
# ********************
optimisation_grid = GridWorld()
V_opt, pol_opt, epochs, iteration_historical_v_functions = optimisation_grid.policy_iteration(0.4, 0.00001)

# Plot value function for policy iteration
print("\n\nIts graphical representation of the optimal value function is:\n")
optimisation_grid.draw_value(V_opt)



# Evaluate the Algorithms
# ***********************
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
plt.savefig("Result_images/RMS_error_of_sarsa_and_mc.png")
plt.close()

plt.plot(sarsa_rms_error, average_sarsa_curve[:400],'ro')
plt.plot(mc_rms_error, average_mc_curve[:400], 'bo')
plt.ylabel('Return')
plt.xlabel('RMS Error')
plt.legend(['SARSA', 'Monte Carlo'], loc='upper right')
plt.savefig("Result_images/Return_vs_RMS_error_of_sarsa_and_mc.png")
plt.close()
