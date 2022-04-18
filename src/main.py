from turtle import color
import autograd.numpy as np
from autograd import grad
from time import time
from matplotlib import pyplot as plt


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
        x=start[0]
        y=start[1]
        theta=start[2]
        delta=start[3]
        
        self.start = [x, y, theta, delta]

        self.goal = goal

        self.state = [x, y, theta, delta]
        self.virtual_state = [x, y, theta, delta]
        '''
        # Control Input = [v, phi]
        # v : Velocity (m/s)
        # phi : Steering Rate (rad/s) 
        '''
        self.control_input = [0.0,0.0]

        # Length of the Vehicle
        self.length = 0.1
        self.size = 1

        # Obstacles
        self.offset = 2
        self.obstacles = [[5,3,0.5],[5,7,0.5]]

        # Start Time of the Robot
        # self.start_time = start_time
        self.time = start_time
        self.prediction_horizon = 10
        self.control_horizon = 1.0
        self.sampling_time = 0.25
        self.max_v = max_v
        self.max_phi = max_phi

        # History of the Vehicle
        self.state_history = []
        self.state_history.append([x, y, theta, delta])
        self.time_history = []
        self.time_history.append(start_time)
        self.control_input_history = []
        self.control_input_history.append([0.0,0.0])

    
    def bicycle_model(self,v,phi, update = False):
        '''
        STATE MODEL
        This funciton gives the next state based on control inputs
        '''
        if update:
            state = self.state
        else:
            state = self.virtual_state
        
        # v and phi are given by the MPC
        pos_x = state[0]
        pos_y = state[1]
        theta = state[2] 
        delta = state[3]

        # print("V=",v)
        # print("theta=",theta)
        x_dot = v*np.cos(theta)
        y_dot = v*np.sin(theta)
        theta_dot = (v*np.tan(delta))/self.length
        delta_dot = phi

        state[0] = state[0] + x_dot*self.sampling_time
        state[1] = state[1] + y_dot*self.sampling_time
        state[2] = state[2] + theta_dot*self.sampling_time
        state[3] = state[3] + delta_dot*self.sampling_time 

        # Updating the States based on Kinematic Bicycle Model
        if update:
            self.state = state
            x=self.state[0]
            y=self.state[1]
            theta=self.state[2]
            delta=self.state[3]
            self.state_history.append([x, y, theta, delta])
            self.control_input_history.append([v,phi])
            self.time += self.sampling_time
            self.time_history.append(self.time)
            
        else:
            self.virtual_state = state   
        

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

    

    def smoothness_cost(self, virtual_input):
        cost_v = ((self.control_input[0] - virtual_input[0])/self.sampling_time)
        cost_phi = ((self.control_input[1] - virtual_input[1])/self.sampling_time)

        # Weights
        W1 = 1
        W2 = 1

        smoothness_cost = W1*cost_v + W2*cost_phi 

        return smoothness_cost

    def constraint_cost(self, virtual_input):
        lamda_v = max(0,virtual_input[0] - self.max_v) + max(0,-virtual_input[0] - self.max_v)
        lamda_phi = max(0,virtual_input[1] - self.max_phi) + max(0,-virtual_input[1] - self.max_phi)
        lamda_ws = 0 #Both x and y workspace constraints
        
        lambda_safety = 0

        for i in range(len(self.obstacles)):
            obs_x = self.obstacles[i][0]
            obs_y = self.obstacles[i][1]
            dist = (((self.virtual_state[0]-obs_x)**2 + (self.virtual_state[1]-obs_y)**2)**0.5)
            if dist < self.offset:
                lambda_safety += max(0,-dist + self.obstacles[i][2] + self.size/2)
                   
            
        # Weights
        W1 = 1
        W2 = 1
        W3 = 1
        W4 = 1

        constraint_cost = W1*lamda_v + W2*lamda_phi + W3*lamda_ws + W4*lambda_safety
        
        return constraint_cost

    def static_obstacle_cost(self):
        static_obstacle_cost = 0
        for i in range(len(self.obstacles)):
            obs_x = self.obstacles[i][0]
            obs_y = self.obstacles[i][1]

            dist = (((self.virtual_state[0]-obs_x)**2 + (self.virtual_state[1]-obs_y)**2)**0.5)

            if dist < self.offset:
                static_obstacle_cost += 1/dist
    
        return static_obstacle_cost       

    def total_cost(self, virtual_input):
        total_cost = 0
        x=self.state[0]
        y=self.state[1]
        theta=self.state[2]
        delta=self.state[3]
    
        self.virtual_state = [x,y,theta,delta]
        
        # Calculating cost across the prediction horizon
        for _ in range(self.prediction_horizon):
            W1=1
            W2=1
            W3=1
            W4=100
            total_cost += self.goal_reach_cost()*W1 + self.smoothness_cost(virtual_input)*W2 + self.constraint_cost(virtual_input)*W3 + self.static_obstacle_cost()*W4
            self.bicycle_model(virtual_input[0],virtual_input[1])
            # print("Me here")
        
        return total_cost

    def optimizer(self, iteration = 25, learning_rate = 0.005, decay = 0.9, eps = 1e-8):
        
        gradient = grad(self.total_cost)
        mean_square_gradient = np.asarray([0.0, 0.0])
        virtual_input = self.control_input

        for _ in range(iteration):
            cost_gradient = np.asarray(gradient(virtual_input))
            mean_square_gradient = (0.9*mean_square_gradient) + 0.1 * (cost_gradient**2)
            virtual_input = virtual_input - (learning_rate/((mean_square_gradient)**0.5+eps))*cost_gradient
        self.control_input = virtual_input
                
if __name__== '__main__':

    Start_A = [0.0, 0.0, 0.0, 0.0]
    Goal_A = [10.0, 10.0, np.pi/4, 0.0]
    start_time = 0 #time()
    VehicleA = Vehicle(Start_A,Goal_A, 25.0, 0.5,start_time)
    # print(VehicleA.total_cost(Start_A))
    fig = plt.figure(1)
    board = plt.axes(xlim=(-5,15),ylim=(-5,12.5))

    while(((VehicleA.state[0] - VehicleA.goal[0])**2 > 1) or ((VehicleA.state[1] - VehicleA.goal[1])**2 > 1)):
        VehicleA.optimizer()
        VehicleA.bicycle_model(VehicleA.control_input[0], VehicleA.control_input[1], True)
        
        for i in range(len(VehicleA.obstacles)):
            obstacle = plt.Circle((VehicleA.obstacles[i][0], VehicleA.obstacles[i][1]),VehicleA.obstacles[i][2], color='orange')
            board.add_patch(obstacle)

        vehicle_a = plt.Circle((VehicleA.state[0], VehicleA.state[1]),0.5, color='blue')
        board.add_patch(vehicle_a)
        
        plt.draw()
        plt.pause(VehicleA.sampling_time*0.1)
        vehicle_a.remove()
        plt.scatter(VehicleA.state[0], VehicleA.state[1])
        # print("Time=",VehicleA.time)
       

    control_input_history = np.array(VehicleA.control_input_history)
    state_history = np.array(VehicleA.state_history)
    time_history = np.array(VehicleA.time_history)

    # print(control_input_history)
    # print(state_history)
    # print(VehicleA.start)
   
    plt.figure(2)
    plt.title('Velocity')
    plt.plot(time_history,control_input_history[:,0])
    
    plt.figure(3)
    plt.title('Stearing Rate')
    plt.plot(time_history,control_input_history[:,1])

    plt.figure(4)
    plt.title('X Position')
    plt.plot(time_history,state_history[:,0])

    plt.figure(5)
    plt.title('Y Position')
    plt.plot(time_history,state_history[:,1])

    plt.figure(6)
    plt.title('Theta')
    plt.plot(time_history,state_history[:,2])

    plt.figure(7)
    plt.title('Delta')
    plt.plot(time_history,state_history[:,3])

    plt.show()











