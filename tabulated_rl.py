import numpy as np
import matplotlib.pyplot as plt
import random



class GridWorld(object):
    def __init__(self):
        # Shape of the gridworld
        self.shape = (6,6)
        # Locations of the obstacles
        self.obstacle_locs = [(1,1),(3,1),(4,1),(4,2),(2,3),(2,5),(4,4)]
        # Locations for the absorbing states
        self.absorbing_locs = [(1,2),(4,3)]
        # Rewards for each of the absorbing states 
        self.special_rewards = [10,-100] # Corresponds to each of the absorbing_locs
        # Reward for all the other states
        self.default_reward = -1
        # Starting location
        self.starting_loc = (3,0)
        # Action names
        self.action_names = ['N','E','S','W'] # Action 0 is 'N', 1 is 'E' and so on
        # Number of actions
        self.action_size = len(self.action_names)
        # Randomizing action results: [1 0 0 0] to no Noise in the action results.
        self.action_randomizing_array = [0.75, 0.083, 0.083, 0.083]
    
        # Internal State
        # **************    
        # Get attributes defining the world
        state_size, T, R, absorbing, locs = self.build_grid_world()
        # Number of valid states in the gridworld (there are 22 of them - 5x5 grid minus obstacles)
        self.state_size = state_size
        # Transition operator (3D tensor)
        self.T = T # T[st+1, st, a] gives the probability that action a will 
                   # transition state st to state st+1
        # Reward function (3D tensor)
        self.R = R # R[st+1, st, a ] gives the reward for transitioning to state
                   # st+1 from state st with action a
        # Absorbing states
        self.absorbing = absorbing
        # The locations of the valid states 
        self.locs = locs # State 0 is at the location self.locs[0] and so on
        # Number of the starting state
        self.starting_state = self.loc_to_state(self.starting_loc, locs);
        # Locating the initial state
        self.initial = np.zeros((1,len(locs)));
        self.initial[0,self.starting_state] = 1
        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob]=1
        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1
        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]
        #Illustrating the grid world
        self.paint_maps()
  
    def get_transition_matrix(self):
        return self.T
    
    def get_reward_matrix(self):
        return self.R
    
    def value_iteration(self, discount:float = 0.9, threshold:float = 0.0001):
        """
        DOCSTRING
            The following algorithm performs value iteration to find the optimal policy in the grid world
        INPUTS
            Discount: the discount of future reward, gamma in the literature
            Threshold: below this change in the value function the algorithm will stop optimising
        OUTPUT
            optimal_policy: an array containing the optimal action for each state
            epochs: the number of iterations performed
        """
        # Transition and reward matrices, both are 3d tensors, c.f. internal state
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        epochs = 0
        delta = threshold # Setting value of delta to go through the first breaking condition
        V = np.zeros(self.state_size) # Initialise values at 0 for each state

        while delta >= threshold:
            epochs += 1 # Increment the epoch
            delta = 0 # Reinitialise delta value

            # For each state
            for state_idx in range(self.state_size):

                # If not an absorbing state
                if not(self.absorbing[0, state_idx]):
                  
                    # Store the previous value for that state
                    v = V[state_idx] 

                    # Compute Q value
                    Q = np.zeros(4) # Initialise with value 0
                    for state_idx_prime in range(self.state_size):
                        Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])
                
                    # Set the new value to the maximum of Q
                    V[state_idx]= np.max(Q) 

                    # Compute the new delta
                    delta = max(delta, np.abs(v - V[state_idx]))
        
        # When the loop is finished, fill in the optimal policy
        optimal_policy = np.zeros((self.state_size, self.action_size)) # Initialisation

        # For each state
        for state_idx in range(self.state_size):
             
            # Compute Q value
            Q = np.zeros(4)
            for state_idx_prime in range(self.state_size):
                Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])
            
            # The action that maximises the Q value gets probability 1
            optimal_policy[state_idx, np.argmax(Q)] = 1 

        return optimal_policy, epochs

    def policy_iteration(self, discount:float = 0.9, threshold: float = 0.0001):
        """
        DOCSTRING
            The following algorithm performs value iteration to find the optimal policy in the grid world
        INPUTS
            Discount: the discount of future reward, gamma in the literature
            Threshold: below this change in the value function the algorithm will stop optimising
        OUTPUT
            V: the final value function, an array contaning the discounted future reqard for each action in 
            each state
            optimal_policy: an array containing the optimal action for each state
            epochs: the number of iterations performed
            historical_Vs: value function over time for plotting
        """
        # Transition and reward matrices, both are 3d tensors, c.f. internal state
        T = self.get_transition_matrix()
        R = self.get_reward_matrix()
        
        # Initialisation
        policy = np.zeros((self.state_size, self.action_size)) # Vector of 0
        policy[:,0] = 1 # Initialise policy to choose action 1 systematically
        epochs = 0
        policy_stable = False # Condition to stop the main loop
        
        historical_Vs = []
        
        while not(policy_stable): 

            # Policy evaluation
            V, epochs_eval = self.policy_evaluation(policy, threshold, discount)
            historical_Vs.append(V)
            epochs += epochs_eval # Increment epoch

            # Set the boolean to True, it will be set to False later if the policy prove unstable
            policy_stable = True

            # Policy iteration
            for state_idx in range(policy.shape[0]):
                
                # If not an absorbing state
                if not(self.absorbing[0,state_idx]):
                    
                    # Store the old action
                    old_action = np.argmax(policy[state_idx,:])
                
                    # Compute Q value
                    Q = np.zeros(4) # Initialise with value 0
                    for state_idx_prime in range(policy.shape[0]):
                        Q += T[state_idx_prime,state_idx,:] * (R[state_idx_prime,state_idx, :] + discount * V[state_idx_prime])

                    # Compute corresponding policy
                    new_policy = np.zeros(4)
                    new_policy[np.argmax(Q)] = 1  # The action that maximises the Q value gets probability 1
                    policy[state_idx] = new_policy
                
                    # Check if the policy has converged
                    if old_action != np.argmax(policy[state_idx]):
                        policy_stable = False
            
        return V, policy, epochs, historical_Vs
                
    def policy_evaluation(self, policy, threshold: float, discount: float):
        """
        DOCSTRING
            The following algorithm performs policy iteration to find the optimal policy in the grid world
        INPUTS
            Discount: the discount of future reward, gamma in the literature
            Threshold: below this change in the value function the algorithm will stop optimising
        OUTPUT
            V: the final value function, an array contaning the discounted future reqard for each action in 
            each state
            epochs: the number of iterations performed
        """
        # Make sure delta is bigger than the threshold to start with
        delta= 2*threshold
        
        #Get the reward and transition matrices
        R = self.get_reward_matrix()
        T = self.get_transition_matrix()
        
        # The value is initialised at 0
        V = np.zeros(policy.shape[0])
        # Make a deep copy of the value array to hold the update during the evaluation
        Vnew = np.copy(V)
        
        epoch = 0
        # While the Value has not yet converged do:
        while delta>threshold:
            epoch += 1
            for state_idx in range(policy.shape[0]):
                # If it is one of the absorbing states, ignore
                if(self.absorbing[0,state_idx]):
                    continue   
                
                # Accumulator variable for the Value of a state
                tmpV = 0
                for action_idx in range(policy.shape[1]):
                    # Accumulator variable for the State-Action Value
                    tmpQ = 0
                    for state_idx_prime in range(policy.shape[0]):
                        tmpQ = tmpQ + T[state_idx_prime,state_idx,action_idx] * (R[state_idx_prime,state_idx, action_idx] + discount * V[state_idx_prime])
                    
                    tmpV += policy[state_idx,action_idx] * tmpQ
                    
                # Update the value of the state
                Vnew[state_idx] = tmpV
            
            # After updating the values of all states, update the delta
            delta =  max(abs(Vnew-V))
            V=np.copy(Vnew)
            
        return V, epoch

    
    # Internal Drawing Functions
    # **************************
    def draw_deterministic_policy(self, Policy):
        """Draw a deterministic policy"""
        # The policy needs to be a np array of 22 values between 0 and 3 with
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        #plt.hold('on')
        for state, action in enumerate(Policy):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
            action_arrow = arrows[action] # Take the corresponding action
            location = self.locs[state] # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    
        plt.savefig("Result_images/deterministic_policy.png")
        plt.close()
    
    def draw_value(self, Value):
        """Draw a policy value function, The value need to be a np array of 22 values"""
        plt.figure()
        
        plt.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
        for state, value in enumerate(Value):
            if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                continue
            location = self.locs[state] # Compute the value location on graph
            plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    
        plt.savefig("Result_images/value_function.png")
        plt.close()

    def draw_deterministic_policy_grid(self, Policy, title, n_columns, n_lines):
        """Draw a grid of deterministic policy, The policy needs to be an arrya of np array of 22 values between 0 and 3 with"""
        # 0 -> N, 1->E, 2->S, 3->W
        plt.figure(figsize=(20,8))
        for subplot in range (len(Policy)): # Go through all policies
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, action in enumerate(Policy[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
                  continue
              arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
              action_arrow = arrows[action] # Take the corresponding action
              location = self.locs[state] # Compute its location on graph
              plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.savefig("Result_images/deterministic_policy.png")
        plt.close()

    def draw_value_grid(self, Value, title, n_columns, n_lines):
        """Draw a grid of value function, The value need to be an array of np array of 22 values """
        plt.figure(figsize=(20,8))
        for subplot in range (len(Value)): # Go through all values
          ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
          ax.imshow(self.walls+self.rewarders +self.absorbers) # Create the graph of the grid
          for state, value in enumerate(Value[subplot]):
              if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
                  continue
              location = self.locs[state] # Compute the value location on graph
              plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
          ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
        plt.savefig("Result_images/grid_of_value_function.png")
        plt.close()
    
    
    # Internal Helper Functions
    # *************************
    def paint_maps(self) -> type(None):
        """Prints out three different versions of the grid world, one showing the obstacles, one
            showing the absorbing states and one showing the reward states"""
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(self.walls)
        plt.title('Obstacles')
        plt.subplot(1,3,2)
        plt.imshow(self.absorbers)
        plt.title('Absorbing states')
        plt.subplot(1,3,3)
        plt.imshow(self.rewarders)
        plt.title('Reward states')
        plt.savefig("Result_images/Grid_world_map.png")
        plt.close()
        
    def build_grid_world(self):
        """Get the locations of all the valid states, the neighbours of each state (by state number),
        and the absorbing states (array of 0's with ones in the absorbing states)"""
        locations, neighbours, absorbing = self.get_topology()
        
        # Get the number of states
        S = len(locations)
        
        # Initialise the transition matrix
        T = np.zeros((S,S,4))
        
        for action in range(4):
            for effect in range(4):
                # Randomize the outcome of taking an action
                outcome = (action+effect+1) % 4
                if outcome == 0:
                    outcome = 3
                else:
                    outcome -= 1

                # Fill the transition matrix:
                prob = self.action_randomizing_array[effect]
                for prior_state in range(S):
                    post_state = neighbours[prior_state, outcome]
                    post_state = int(post_state)
                    T[post_state,prior_state,action] = T[post_state,prior_state,action]+prob
                    
        # Build the reward matrix
        R = self.default_reward*np.ones((S,S,4))
        for i, sr in enumerate(self.special_rewards):
            post_state = self.loc_to_state(self.absorbing_locs[i],locations)
            R[post_state,:,:]= sr
        
        return S, T, R, absorbing, locations

    def get_topology(self):
        """
        DOCSTRING:
            Get the topology of the grid world: the states, neighboring states and absorbing states
        OUTPUT:
            locs: array containing valid locations (not an obstacle) in the grid contains the absorbing states.
                every valid location is in the format of (y_coordinate, x_coordinate)
            state_neighbours: is a number of valid states x number of actions sized matrix. The first element
                contains the successor states from state 0 if a movement was performed in the direction of
                [up, down, right, left]. if the number in the array is the state itself, that means that there
                was an obstacle in the way and as the result of the movement the agent "bounced back" from the
                wall and remaind in the same state
            absorbing: the indices of the absorbing states
        """
        height = self.shape[0]
        width = self.shape[1]
        index = 1 
        locations = []
        neighbour_locations = []
        
        for i in range(height):
            for j in range(width):
                # Get the locaiton of each state
                loc = (i,j)
                
                # And append it to the valid state locations if it is a valid state (ie not an obstacle)
                if(self.is_location(loc)):
                    locations.append(loc)
                    
                    # Get an array with the neighbours of each state by location (y_coordinate, x_coordinate)
                    # If a movement could not be made then the neighbour will be recorded as itself
                    # ie. ha nekimegyek egy falnak valamelyik iranyba akkor onmagam szomszedja vagyok abba az iranyba
                    local_neighbours = []
                    for direction in ['N','E','S', 'W']:
                        local_neighbours.append(self.get_neighbour(loc,direction))
                    
                    neighbour_locations.append(local_neighbours)
                
        # translate neighbour lists from locations to states
        num_states = len(locations)
        state_neighbours = np.zeros((num_states,4))
        
        # For every state
        for state in range(num_states):
            # For every direction, range four is used to index the neighbour locations array previously created,
            # index0 = up, index1 = right, index2 =down, index3 = left
            for direction in range(4):
                # Find neighbour location
                neighbour_state_location = neighbour_locations[state][direction]
                
                # Turn location into a state number
                number_of_the_state = self.loc_to_state(neighbour_state_location, locations)
      
                # Insert into neighbour matrix
                state_neighbours[state,direction] = number_of_the_state;
    
        # Translate absorbing locations into absorbing state indices
        absorbing = np.zeros((1,num_states))
        for a in self.absorbing_locs:
            absorbing_state = self.loc_to_state(a,locations)
            absorbing[0,absorbing_state] =1
        
        return locations, state_neighbours, absorbing 

    def loc_to_state(self, location:tuple, locations_list:list) -> type(int):
        """
        DOCSTRING:
            Converts a state's coordinate to the number of the state
        INPUT:
            location: (y-coordinate, x-coordinate)
            locations_list: a list of valid locations in the grid (not an absorbing state or obstacle)
        OUTPUT:
            the state number of the corresponding coordinate specified by the location
        
        The states are numbered in the following fashion:
            0  1  2  3  4  5
            6  X  7  8  9  10
            11 12 13 X  14  X
            15 X  16 17 18 19
            20 X  X  21 X  22
            23 24 25 26 27 28
        """
        #takes list of locations and gives index corresponding to input loc
        return locations_list.index(tuple(location))

    def is_location(self, location: tuple) -> type(bool):
        """
        DOCSTRING:
            Checks if a given state is a valid state or not, it is a valid state if it is within the grid and
            not an obstackle
        INPUT:
            location: (y_coordinate, x_coordinate) of the location you want to check
        OUTPUT:
            True if it is a valid location, False if it is not
        """
        # It is a valid location if it is in grid and not obstacle
        if(location[0]<0 or location[1]<0 or location[0]>self.shape[0]-1 or location[1]>self.shape[1]-1):
            return False
        elif(location in self.obstacle_locs):
            return False
        else:
             return True

    def get_neighbour(self, location:tuple, direction:str) -> type(tuple):
        """
        DOCSTRING:
            Returns the location of the new coordinate if a movement was successful, or the same coordinate
            if a movement could not be made. A movement cannot be made if the successor state is out of the 
            grid or an obstacle.
        INPUT:
            location: The current location of the agent in the form of (y_coordinate, x_coordinate)
            direction: the direction of the movement we want to perform
                'nr' -> up, 'ea' -> right, 'so' -> down, 'we' -> left
        OUTPUT:
            resulting coordinate if the movement could or could not be made (y_coordinate, x_coordinate)
        """
        #Find the valid neighbours (ie that are in the grif and not obstacle)
        i = location[0]
        j = location[1]
        
        nr = (i-1,j)
        ea = (i,j+1)
        so = (i+1,j)
        we = (i,j-1)
        
        # If the neighbour is a valid location, accept it, otherwise, stay put
        if(direction == 'N' and self.is_location(nr)):
            return nr
        elif(direction == 'E' and self.is_location(ea)):
            return ea
        elif(direction == 'S' and self.is_location(so)):
            return so
        elif(direction == 'W' and self.is_location(we)):
            return we
        else:
            #default is to return to the same location
            # (y_coordinate, x_coordinate)
            return location
    
    def return_reward_of_step(self, successor_state) -> type(int):
        """
        DOCSTRING:
            Returns the reward you receive for stepping into a specified state
        INPUT:
            successor_state: (y_coordinate, x_coordinate) if the state you entered
        OUTPUT:
            The reward you receive when you step into the provided successor state.
        """
        try:
            
            index = self.absorbing_locs.index(successor_state)
            return self.special_rewards[index]
        except:
            return self.default_reward
    
    def calculate_e_greedy_policy(self, q_function, iteration: int):
        """
        DOCSTRING:
            Calculates an epsilon greedy policy from a q function, based on the lower bit of slide 197
        INPUT:
            q_function: The q function of the grid
            iteration: iteration counter which sets the epsilon value, based on slide 198
        OUTPUT:
            The output is the Policy, which is a number_of_states times number_of_actions sized numpy array
        """
        epsilon = 1/iteration
        
        # initialise an empty policy
        Policy = np.zeros((self.state_size, self.action_size))
        
        for count, q_row in enumerate(q_function):
            
            # fill up the policy vector with e/action count
            for i in range(len(Policy[count])):
                Policy[count][i] = epsilon/self.action_size
            
            # Find the index of the biggest number in the q_row, if it occures multiple times just use the first one
            max_q_index = np.where(q_row == np.amax(q_row))
            max_q_index = max_q_index[0][0]
            
            # Update its value accordint to the e-greedy policy setting rule
            Policy[count][max_q_index] = 1 - epsilon + epsilon/self.action_size
        
        # Return the policy from that Q function
        return Policy
    
    def check_action_success(self, action: str) -> type(str):
        """
        DOCSTRING:
            Checks if my action will be executed or a different one, based on the probabilistic world
        INPUT:
            action: could be N, E, S, W, the action that will be checked for execution
        OUTPUT:
            The action that will be executed, the same action if the step was successful or a different one
            if the world made me change it
        """
        # Generate a random value between 0 and 100
        rand_val = random.uniform(0, 100)
        
        # Get the possible actions and remove the choosen one from it
        possible_actions = np.copy(self.action_names)
        possible_actions = possible_actions[possible_actions != action]
        
        # Remove my choosen action from the possible actions array
        # The case when my action was successful
        if rand_val < self.action_randomizing_array[0]*100:
            return action
        # The case when my action was not successful, just return a random action, since all of them have the
        # same probability
        else:
            return random.choice(possible_actions)
    
    def get_action(self, state, Policy) -> type(str):
        """
        DOCSTRING:
            Determines which action to take in a state with a given Policy and runs the check_action_success function
            on the action determined by the policy, to check if the environment "allows" us to take that policy
        INPUT:
            state: The state that you want to get the action for (y_coordinate, x_coordinate)
            Policy: The Policy for the grid
        OUTPUT:
            A string correcponding to the action (N, E, S or w)
        """
        # Get the index of the state that will be the same in locs, policy, q_function
        state_index = self.locs.index(state)
        
        # The corresponding policy row will contain the probabilities of taking action in the structure of
        # [N, E, S, W] (up, right, down, left)
        policy_row = Policy[state_index]
        
        # Random number to choose a step
        rand_val = random.uniform(0, 100)
        
        if rand_val < policy_row[0]*100:
            # Step will be North
            return 'N'
        elif rand_val > policy_row[0]*100 and rand_val <= (policy_row[0]*100 + policy_row[1]*100):
            # Step will be East
            return 'E'
        elif rand_val > (policy_row[0]*100 + policy_row[1]*100) and rand_val <= (policy_row[0]*100 + policy_row[1]*100 + policy_row[2]*100):
            # Step will be South
            return 'S'
        else:
            # Step will be West
            return 'W'

    def get_actioin_index(self, action: str) -> type(int):
        if action == 'N':
            return 0
        elif action == 'E':
            return 1
        elif action == 'S':
            return 2
        elif action == 'W':
            return 3
    
    def get_random_starting_state(self):
        """Select a random starting locationwhich is a valid starting state (not absorbing or obstacle)"""
        starting_locations = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 3), (1, 4), (1, 5), (2, 0), (2, 1), (2, 2), (2, 4), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5), (4, 0), (4, 5), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]    
        return random.choice(starting_locations)
    
    def create_trace(self, Policy, starting_state) -> type(list):
        """
        DOCSTRING:
            Create a trace with a provided policy, it starts for a random state and runs until it goes into
            an absorbing state.
        INPUT:
            Policy: the policy for the grid (state_size x action_size)
        OUTPUT:
            Trace: [[current_state_index, action_index, reward, successor_state_index],[...]..]
        """
        # Get a random starting state
        #current_state = self.get_random_starting_state()
        current_state = starting_state
        
        # Initialise successor state
        successor_state = (0, 0)
        
        # Initialise an empty trace
        trace = []
        
        # while the successor state is not an absorbing state
        while successor_state not in self.absorbing_locs:
            action = self.get_action(current_state, Policy)
            action_taken = self.check_action_success(action)
            successor_state = self.get_neighbour(current_state, action_taken)
            reward = self.return_reward_of_step(successor_state)
            
            # Convert the "homan readable" states and actions to indicies
            current_state_index = self.loc_to_state(current_state, self.locs)
            successor_state_index = self.loc_to_state(successor_state, self.locs)
            action_index = self.get_actioin_index(action)
            
            # Append [(state, action), reward, successor_state] to trace
            trace.append([current_state_index, action_index, reward, successor_state_index])
            
            # Move the current state to the successor state ie. make the step
            current_state = successor_state
        return trace

    def evaluate_trace(self, trace: list, discount_factor: float) -> type(list):
        """
        DOCSTRING
            Convert the traces to a list of returns based on the first visit principle
        INPUTS
            trace: the trace observed by the agent (state, action, reward) until terminal state
            discount_factor: the discount of future reward, gamma in the literature
        OUTPUTS
            returns_list: list containint the returns for Q values in the format of ([s, a], discounted reward r)
        """
        # Has the structure of [[(state, action index), discounted return],[]]
        unique_sa_pairs = []
        returns_list = []
        
        # Get the unique State action pairs in the trace
        for index, step in enumerate(trace):
            # Check if this state action pair is in the unique_sa_pairs
            #if yes, go to the next step in the trace
            if (step[0], step[1]) in unique_sa_pairs:
                continue
            # Append the state action pair to the unique sa pairs list
            else:
                # Append it to the unique State action pairs so it will not be counted again (First Visit MC)
                unique_sa_pairs.append((step[0], step[1]))
                
                # Calculate the discounted return from that given state
                gamma_multiplier = 1
                total_reward = 0
                for i in range(index, len(trace)):
                    if i == index:
                        total_reward += trace[i][2]
                    else:
                        total_reward += trace[i][2]*discount_factor**gamma_multiplier
                        gamma_multiplier += 1
                
                # Add [(state, action), total_discounted_return] to the returns_list
                returns_list.append([(step[0], step[1]), total_reward])
        return returns_list
    
    def running_average_calculator(self, old_average:float, new_value:float, iterator_k:int):
        """Function used to calculate running averages"""
        new_average = old_average+(1/iterator_k)*(new_value - old_average)
        return new_average
    
    def update_q_in_mc(self, old_q, q_counter_matrix, returns_list):
        """
        DOCSTRING
            Function to update the Q values in Monte Carlo Learning based on traces collected by the agent
        INPUTS
            old_q: an array containing the old Q values for the grid world
            q_counter_matrix: an array containing the number of updates for each Q value, so the running
                average can be calculated
            returns_list: a list containing the traces observed by the agent
        OUTPUT
            old_q: the updated Q values array
            q_counter_matrix: the updated list with the number of updates for each Q value
        """
        # create a new Q matrix from the returns list
        new_Q = np.zeros((self.state_size, self.action_size))
        new_Q = new_Q - 1000
        
        # Add the return of each state action pair with a running average
        for s_a_return in returns_list:
            new_Q[s_a_return[0][0]][s_a_return[0][1]] = s_a_return[1]
                
        # Add the two Q matricies together with the running average calculator, use q_counter matrix and
        # also update it
        for i in range(len(new_Q)):
            #for old_q_value, new_q_value, q_counter_value in zip(old_q_row, new_q_rowq_counter_matrix_row):
            for j in range(len(new_Q[i])):
                # If we do not have a new Q value for that state and action do not do anything
                if new_Q[i][j] == -1000:
                    continue
                # If we have a new value for this specific state action par in Q
                else:
                    # Update the value in the Q matrix with a running average
                    old_q[i][j] = self.running_average_calculator(old_q[i][j], new_Q[i][j], q_counter_matrix[i][j])
                    
                    # Update the q_counter_matrix at the index
                    q_counter_matrix[i][j] = q_counter_matrix[i][j] + 1
        
        # return the new Q matrix and q new Q counter matrix
        # Naming is shit, the old q is the new q
        return old_q, q_counter_matrix
    
    def calculate_V_from_Q(self, Q_function, Policy) -> type(list):
        """
        DOCSTRING
            Calculate the value function from the the Q matrix and the policy, the value of the state is the
            probability of taking the action in tha state with the current policy and the Q value for that
            state action pair
        INPUTS
            Q_function: an array containing the Q values for each state and action pair
            Policy: an array containg the policy for the grid world
        OUTPUT
            V_function: the value function of the grid world
        """
        V_function = []
        for Q_row, Policy_row in zip(Q_function, Policy):
            value = 0
            for q_value, policy_value in zip(Q_row, Policy_row):
                value += q_value*policy_value
            V_function.append(value)
        
        return V_function
    
    def get_total_return_from_trace_backwards(self, trace:list, discount_factor:float) -> type(float):
        """
        DOCSTRING
            The function calculates the total reward for a state action pair, calculated backwards for more
            accurate values and better results.
        INPUT
            trace: a list containing state, action, reward steps until a terminal state
            discount_factor: the discount of future reward, gamma in the literature
        OUTPUT
            total_reward: the total discounted reward for a state and action
        """
        total_reward = 0
        gamma_multiplier = 1
        # Reverse the order of the array to get backwards propagated return
        trace = np.flip(trace)
        
        for i in range(len(trace)):
            if i == 0:
                total_reward += trace[i][2]
            else:
                total_reward += trace[i][2]*discount_factor**gamma_multiplier
                gamma_multiplier += 1
        return total_reward
        
    
    # Reinforcement Learning Algorithms
    # *********************************
    def do_monteraclo(self, discount_factor:float, iteration_counter:int):
        """
        DOCSTRING
            The algorithm performs First Visit Monte Carlo Learning, with backwards calculated returns for
            better performance
        INPUTS
            discount_factor: the discount of future reward, gamma in the literature
            iteration_counter: the number of iterations to run
        OUTPUTS
            Policy: an array containing the final policy achieved by the algorithm
            V_function: an array containing the value function achieved by the algorithm
            returns_for_each_trace: a list of returns for each iterations for plotting
            historical_V_func_data: a lost containing all of the value functions for RMS error calculation and
                plotting later
        """
        iteration = 1
        # Create a random Q function (number of states x number of action matrix) initialised to 0
        Q_function = np.zeros((self.state_size, self.action_size))
        
        # Calculate the policy from that Q function with e-greedy algorithm
        Policy = self.calculate_e_greedy_policy(Q_function, iteration)
        
        # create an update counter for each element in the policy, it will be used in the update q in mc
        # function for the running average calculator, fill it with ones so for the first time the updated q will
        # the sum of the new and the old q
        Q_element_update_counter = np.zeros((self.state_size, self.action_size))
        Q_element_update_counter = Q_element_update_counter + 1
        
        # Array to store the returns from each episode for the plot
        returns_for_each_trace = []
        historical_V_func_data = []
        
        while iteration < iteration_counter:
            starting_state = self.get_random_starting_state()
            trace = self.create_trace(Policy, starting_state)
                    
            # Get the returns list which will be in the format of [[(state_number, action_number), discounted_total_return], ..]
            returns_list = self.evaluate_trace(trace, discount_factor)

            # Update Q based on the trace
            Q_function, Q_element_update_counter = self.update_q_in_mc(Q_function, Q_element_update_counter, returns_list)
           
            # Update the Policy based on the Updated Q
            Policy = self.calculate_e_greedy_policy(Q_function, iteration)
            iteration += 1
            
            # Get the discounted total return from the trace, counted backwards
            trace = self.create_trace(Policy, self.get_random_starting_state())
            total_reward_from_episode = self.get_total_return_from_trace_backwards(trace, discount_factor)
            returns_for_each_trace.append(total_reward_from_episode)
        
            # Calculate the value function V(s) from the Q(s,a) function
            V_function = self.calculate_V_from_Q(Q_function, Policy)
            historical_V_func_data.append(V_function)
        
        return Policy, V_function, returns_for_each_trace, historical_V_func_data
    
    def SARSA_Q_update(self, Q_function:list, state:int, action:int, learning_rate:float, reward:float, discount_factor:float, successor_state:int, successor_state_action:int) -> type(list):
        """
        DOCSTRING
            Update the Q function for the SARSA algorithm
        INPUTS
            Q_function: the previous Q function
            state: the index of the current state
            action: the index of the action taken
            learning_rate: used by the running average function
            reward: the reward for that state and action
            discount_factor: 
            successor_state: the index of the successor state
            successor_state_action: the index of the action taken in the successor state
        PUTPUT
            Q_function: the updated Q function (values for each state and action pair)
        """
        Q_function[state][action] = Q_function[state][action] + learning_rate*(reward + discount_factor*Q_function[successor_state][successor_state_action] - Q_function[state][action])
        return Q_function
    
    def do_SARSA(self, discount_factor:float, learning_rate:float, iteration_counter:int):
        """
        DOCSTRING
            Run the SARSA algorithm on the grid world to learn the optimal policy and value function
        INPUT
            discount_factor: the discount of future reward, gamma in the literature
            learning_rate: the learning rate for the algorithm, used by the running average function
            iteration_counter: the number of iterations to run
        OUTPUT
            Policy: an array containing the final policy achieved by the algorithm
            V_function: an array containing the value function achieved by the algorithm
            returns_for_each_trace: a list of returns for each iterations for plotting
            historical_V_func_data: a lost containing all of the value functions for RMS error calculation and
                plotting later
        """
        iteration = 1
        
        # Create a random Q function (number of states x number of action matrix) initialised to 0
        Q_function = np.zeros((self.state_size, self.action_size))
        
        # Calculate the policy from that Q function with e-greedy algorithm
        Policy = self.calculate_e_greedy_policy(Q_function, iteration)
        
        # Array to store the returns from each episode for the plot
        returns_for_each_trace = []
        historical_V_func_data = []
        
        # Run traces n number of times
        while iteration < iteration_counter:
            # Get random starting state
            current_state = self.get_random_starting_state()
            successor_state = (0, 0)
            
            action = self.get_action(current_state, Policy)
            action_taken = self.check_action_success(action)
            
            # Keep stepping until a terminal state is reached
            while successor_state not in self.absorbing_locs:
                # Get action, reward, successor state
                successor_state = self.get_neighbour(current_state, action_taken)
                reward = self.return_reward_of_step(successor_state)
            
                # Convert the "human readable" states and actions to indicies
                current_state_index = self.loc_to_state(current_state, self.locs)
                successor_state_index = self.loc_to_state(successor_state, self.locs)
                action_index = self.get_actioin_index(action)    
            
                # Get the action from the successor state
                successor_state_action = self.get_action(successor_state, Policy)
                successpr_state_action_taken = self.check_action_success(successor_state_action)
                successor_state_action_index = self.get_actioin_index(successor_state_action)
                
                # Update the Q function
                Q_function = self.SARSA_Q_update(Q_function, current_state_index, action_index, learning_rate, reward, discount_factor, successor_state_index, successor_state_action_index)
                learning_rate = learning_rate * 0.9999

                # Update the current state and successor state
                current_state = successor_state
                action = successpr_state_action_taken
                action_taken = self.check_action_success(action)
            
            trace = self.create_trace(Policy, self.get_random_starting_state())
            total_reward_from_episode = self.get_total_return_from_trace_backwards(trace, discount_factor)
            returns_for_each_trace.append(total_reward_from_episode)
            
            # Update the policy from the Q function
            Policy = self.calculate_e_greedy_policy(Q_function, iteration)
            iteration += 1
        
            # Calculate V_function
            V_function = self.calculate_V_from_Q(Q_function, Policy)
            historical_V_func_data.append(V_function)
        
        return Policy, V_function, returns_for_each_trace, historical_V_func_data
