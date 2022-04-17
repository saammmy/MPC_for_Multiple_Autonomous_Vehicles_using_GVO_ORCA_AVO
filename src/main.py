import autograd.numpy as np
from autograd import grad
from time import time
import matplotlib.pyplot as plt
import math


class Vehicle:
    
    def __init__(self, start, goal, max_v, max_phi, start_time):
    
        '''
        Start, goal and state contains:
            pos_x : Position in X-coordinate (m)
            pos_y : Position in Y-coordinate (m)
            theta : Orientation Angle (radians)
            delta : Steering Angle (radians)
        These are the states of the Vehicle
        '''
        self.start = start
        self.goal = goal
        self.state = start
        self.virtual_state = self.state
        '''
        # Control Input = [v, phi]
        # v : Velocity (m/s)
        # phi : Steering Rate (rad/s) 
        '''
        self.virtual_input = [0.0,0.0]
        self.control_input = [0.0,0.0]

        # Length of the Vehicle
        self.length = 1

        # Start Time of the Robot
        # self.start_time = start_time
        self.time = start_time
        self.prediction_horizon = 10
        self.control_horizon = 1
        self.sampling_time = 0.25
        self.max_v = max_v
        self.max_phi = max_phi

        # History of the Vehicle
        self.state_history = []
        self.state_history.append(self.start)
        self.time_history = []
        self.time_history.append(0)
        self.control_input_history = []
        self.control_input_history.append([0,0])

    
    def bicycle_model(self,v,phi, update = False):
        '''
        STATE MODEL
        This funciton gives the next state based on control inputs
        '''
        # v and phi are given by the MPC
        pos_x = self.state[0]
        pos_y = self.state[1]
        theta = self.state[2] 
        delta = self.state[3]

        x_dot = v*math.cos(theta)
        y_dot = v*math.sin(theta)
        theta_dot = (v*math.tan(delta))/self.length
        delta_dot = phi

        # Updating the States based on Kinematic Bicycle Model
        if update:
            self.state[0] = self.state[0] + x_dot*self.sampling_time
            self.state[1] = self.state[1] + y_dot*self.sampling_time
            self.state[2] = self.state[2] + theta_dot*self.sampling_time
            self.state[3] = self.state[3] + delta_dot*self.sampling_time

            # Storing the states and control inputs for plotting
            self.state_history.append(self.state)
            self.control_input_history.append([v,phi])
            self.time += self.sampling_time
            self.time_history.append(self.time)
        else:
            self.virtual_state[0] = self.virtual_state[0] + x_dot*self.sampling_time
            self.virtual_state[1] = self.virtual_state[1] + y_dot*self.sampling_time
            self.virtual_state[2] = self.virtual_state[2] + theta_dot*self.sampling_time
            self.virtual_state[3] = self.virtual_state[3] + delta_dot*self.sampling_time
        
        

    def goal_reach_cost(self):
        # State cost to Goal
        cost_xy = (self.virtual_state[0]-self.goal[0])**2 + (self.virtual_state[1]-self.goal[1])**2
        cost_theta = (self.virtual_state[2]-self.goal[2])**2
        cost_delta = (self.virtual_state[3]-self.goal[3])**2

        # Weights
        W1 = 1
        W2 = 1
        W3 = 1

        goal_reach_cost = W1*cost_xy + W2*cost_theta + W3*cost_delta

        return goal_reach_cost

    def smoothness_cost(self):
        cost_v = ((self.control_input[0] - self.virtual_input[0])/self.sampling_time)
        cost_phi = ((self.control_input[1] - self.virtual_input[1])/self.sampling_time)

        # Weights
        W1 = 1
        W2 = 1

        smoothness_cost = W1*cost_v + W2*cost_phi 

        return smoothness_cost

    def constraint_cost(self):
        lamda_v = max(0,self.virtual_input[0] - self.max_v) + max(0,-self.virtual_input[0] - self.max_v)
        lamda_phi = max(0,self.virtual_input[1] - self.max_phi) + max(0,-self.virtual_input[1] - self.max_phi)
        lamda_ws = 0 #Both x and y workspace constraints
        lambda_safety = 0

        # Weights
        W1 = 1
        W2 = 1
        W3 = 1
        W4 = 1

        constraint_cost = W1*lamda_v + W2*lamda_phi + W3*lamda_ws + W4*lambda_safety
        
        return constraint_cost

    def total_cost(self, input):
        total_cost = 0
        self.virtual_state = self.state
        self.virtual_input = input

        # Calculating cost across the prediction horizon
        for _ in range(self.prediction_horizon):
            total_cost += self.goal_reach_cost() + self.smoothness_cost() + self.constraint_cost()
            # print("Me here")
            # self.bicycle_model(self.virtual_input[0],self.virtual_input[1])
            # self.bicycle_model(input[0],input[1])

        return total_cost

    def optimizer(self, iteration = 25, learning_rate = 0.005, decay = 0.9, eps = 1e-8):
        
        gradient = grad(self.total_cost)
        mean_square_gradient = [0, 0]
        virtual_input = self.control_input

        for _ in range(iteration):
            cost_gradient = gradient(virtual_input)
            print(cost_gradient)
            mean_square_gradient = (0.9*mean_square_gradient) + 0.1 * (cost_gradient**2)
            self.virtual_input = self.virtual_input - (learning_rate/((mean_square_gradient)**0.5+eps))*cost_gradient

                
if __name__== '__main__':

    Start_A = [0.0, 0.0, 0.0, 0.0]
    Goal_A = [500.0, 500.0, math.pi/2, 0.0]
    start_time = time()
    VehicleA = Vehicle(Start_A,Goal_A,25.0,5.0,start_time)
    print(VehicleA.total_cost(Start_A))

    while((VehicleA.state[0] - VehicleA.goal[0])**2 > 10 or (VehicleA.state[1] - VehicleA.goal[1])**2 > 10 ):
        VehicleA.optimizer()
        VehicleA.control_input = VehicleA.virtual_input
        VehicleA.bicycle_model(VehicleA.control_input[0], VehicleA.control_input[1], True)

    VehicleA.control_input_history = np.asarray(VehicleA.control_input_history)
    VehicleA.state_history = np.asarray(VehicleA.state_history)
    
    plt.figure(1)
    plt.title('Velocity')
    plt.plot(VehicleA.time_history,VehicleA.control_input_history[0,:])

    plt.figure(2)
    plt.title('Stearing Rate')
    plt.plot(VehicleA.time_history,VehicleA.control_input_history[1,:])

    plt.figure(3)
    plt.title('X Position')
    plt.plot(VehicleA.time_history,VehicleA.state_history[0,:])

    plt.figure(4)
    plt.title('Y Position')
    plt.plot(VehicleA.time_history,VehicleA.state_history[1,:])

    plt.figure(5)
    plt.title('Theta')
    plt.plot(VehicleA.time_history,VehicleA.state_history[2,:])

    plt.figure(6)
    plt.title('Delta')
    plt.plot(VehicleA.time_history,VehicleA.state_history[3,:])












