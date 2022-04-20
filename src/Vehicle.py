from turtle import pos
from PIL import Image
import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Vehicle:

    def __init__(self, id, start, goal, inital_control_input, max_v, max_phi, start_time, sampling_time, prediction_horizon, Obstacles, offset = [100, 250], COLOR = "#f1c40f", path = None,  ZOOM = 0.05):
    
        # Vehicle Details (Dimesions Inspired by Lamborgini Aventador)
        self.length = 2.70  # Wheelbase
        self.size = 4.78    # Vehicle Size (Diameter)
        self.type = "Vehicle"
        self.id = id
        # Load Image from Path
        if path!=None:
            self.image = Image.open(path)
        else:
            self.image = None

        self.COLOR = COLOR  #Color of the Vehicle Path/ Sensing Region
        self.ZOOM = ZOOM    #Zoom of the Marker (Photo)

        '''
        Start, goal and state contains:
            pos_x : Position in X-coordinate (m)
            pos_y : Position in Y-coordinate (m)
            theta : Orientation Angle (radians)
            delta : Steering Angle (radians)
        These are the states of the Vehicle
        '''
        x_start=start[0]
        y_start=start[1]
        theta_start=start[2]*np.pi/180
        delta_start=start[3]*np.pi/180
        
        self.start = [x_start, y_start, theta_start, delta_start]

        x_goal=goal[0]
        y_goal=goal[1]
        theta_goal=goal[2]*np.pi/180
        delta_goal=goal[3]*np.pi/180
        self.goal = [x_goal, y_goal, theta_goal, delta_goal]

        self.state = [x_start, y_start, theta_start, delta_start]
        self.virtual_state = [x_start, y_start, theta_start, delta_start]

        '''
        # Control Input = [v, phi]
        # v : Velocity (m/s)
        # phi : Steering Rate (rad/s) 
        '''
        initial_v = inital_control_input[0]
        inital_phi = inital_control_input[1]*np.pi/180
        self.control_input = [initial_v, inital_phi]

        # Constraints:
        self.max_v = max_v
        self.max_phi = max_phi

        # Obstacles
        self.Obstacles = Obstacles
        # Sensing range of the obstacles for the vehicle
        self.static_offset = offset[0]
        self.dynamic_offset = offset[1]
        
        # Simulation Parameters
        self.time = start_time
        self.prediction_horizon = prediction_horizon
        self.control_horizon = 1.0
        self.sampling_time = sampling_time

        # History of the Vehicle
        self.state_history = [[x_start, y_start, theta_start, delta_start]]
        self.state_history.append([x_start, y_start, theta_start, delta_start])
        self.time_history = [start_time]
        self.time_history.append(start_time)
        self.control_input_history = [[initial_v, inital_phi]]
        self.control_input_history.append([initial_v, inital_phi])


    def distance(self, x1, x2):
        return np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)


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
        W3 = 0

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

    def constraint_cost(self, virtual_input, W_v = 1, W_phi = 1, W_ws = 1, W_safety = 5):
        lamda_v = max(0,virtual_input[0] - self.max_v) + max(0,-virtual_input[0] - self.max_v)
        lamda_phi = max(0,virtual_input[1] - self.max_phi) + max(0,-virtual_input[1] - self.max_phi)
        lamda_ws = 0 #Both x and y workspace constraints
        
        lambda_safety = 0
        for obstacle in self.Obstacles:
            if obstacle.type != "Vehicle":
                obs_x = obstacle.parameters[0]
                obs_y = obstacle.parameters[1]
                radius = obstacle.parameters[2]
            else:
                if self.id < obstacle.id: #This condition ensures that the vehicle doesnt know the updated state of other vehicle
                    other_vehicle_state = obstacle.state_history[-2]
                else:
                    other_vehicle_state = obstacle.state_history[-1]
                obs_x = other_vehicle_state[0]
                obs_y = other_vehicle_state[1]
                radius = obstacle.size/2

            dist = self.distance([obs_x,obs_y],[self.state[0],self.state[1]])
            dist_virtual = self.distance([obs_x,obs_y],[self.virtual_state[0],self.virtual_state[1]])
            if obstacle.type == 'Static':
                if dist < self.static_offset:
                    lambda_safety += max(0,-dist_virtual + radius + self.size/2)
            elif obstacle.type == 'Dynamic': 
                if dist < self.dynamic_offset:
                    lambda_safety += max(0,-dist_virtual + radius + self.size/2)
            elif obstacle.type == 'Vehicle':
                if dist < self.dynamic_offset:
                    lambda_safety += max(0,-dist_virtual + radius + self.size/2)

        constraint_cost = W_v*lamda_v + W_phi*lamda_phi + W_ws*lamda_ws + W_safety*lambda_safety
        
        return constraint_cost
    
    def obstacle_cost(self, W_static = 1000, W_dynamic = 1500):
        obstacle_cost = 0
        for obstacle in self.Obstacles:
            if obstacle.type != "Vehicle":
                obs_x = obstacle.parameters[0]
                obs_y = obstacle.parameters[1]
            else:
                if self.id < obstacle.id: #This condition ensures that the vehicle doesnt know the updated state of other vehicle
                    other_vehicle_state = obstacle.state_history[-2]
                else:
                    other_vehicle_state = obstacle.state_history[-1]
                obs_x = other_vehicle_state[0]
                obs_y = other_vehicle_state[1]
                # print("X=",obs_x)
                # print("Y=",obs_y)

            dist = self.distance([obs_x,obs_y],[self.state[0],self.state[1]])
            dist_virtual = self.distance([obs_x,obs_y],[self.virtual_state[0],self.virtual_state[1]])
            # if dist_virtual<10e-4:
            #     print("virtual distance = ",dist_virtual)
            #     print("X", obs_x)
            #     print("Y", obs_y)
            #     print(obstacle.type)
            #     print(obstacle.id)
            #     print(self.id)
            #     print("------")

            if obstacle.type == 'Static':
                if dist < self.static_offset:
                    obstacle_cost += 1/dist_virtual * W_static
            elif obstacle.type == 'Dynamic':
                if dist < self.dynamic_offset:
                    obstacle_cost += 1/dist_virtual * W_dynamic
            elif obstacle.type == 'Vehicle':
                if dist < self.dynamic_offset:
                    obstacle_cost += 1/dist_virtual * W_dynamic
        
        return obstacle_cost


    def total_cost(self, virtual_input):
        total_cost = 0
        x=self.state[0]
        y=self.state[1]
        theta=self.state[2]
        delta=self.state[3]
    
        self.virtual_state = [x,y,theta,delta]
        
        # Calculating cost across the prediction horizon
        for i in range(self.prediction_horizon):
            # Weights
            W_goal_reach=0.1
            W_smoothness=1
            W_constraint=10
            W_static=1000
            W_dynamic=2000

            # Adding a Terminal Cost for the final prediction horizon
            if i == self.prediction_horizon-1:
                terminal_cost = self.goal_reach_cost()*W_goal_reach*5
            else:
                terminal_cost = 0

            total_cost += self.goal_reach_cost()*W_goal_reach + self.smoothness_cost(virtual_input)*W_smoothness + self.constraint_cost(virtual_input)*W_constraint + self.obstacle_cost(W_static, W_dynamic) + terminal_cost
            
            # Update the virtual state to next prediction horizon
            self.bicycle_model(virtual_input[0],virtual_input[1])
        
        return total_cost

    def optimizer(self, iteration = 20, learning_rate = 0.005, decay = 0.9, eps = 1e-8):
        # Using Autograd to find the gradient
        gradient = grad(self.total_cost)
        mean_square_gradient = np.asarray([0.0, 0.0])
        virtual_input = self.control_input #Assigning the virtual input

        # Performing Optimization using RMS Prop
        for _ in range(iteration):
            cost_gradient = np.asarray(gradient(virtual_input))
            mean_square_gradient = (0.9*mean_square_gradient) + 0.1 * (cost_gradient**2)
            virtual_input = virtual_input - (learning_rate/((mean_square_gradient)**0.5+eps))*cost_gradient
        
        # Assigning the optimized virtual input to vehicle
        self.control_input = virtual_input
    

    def plot(self, ax):
        # blue: #003b77
        # red: #f50116
        # yellow: #f1c40f
        # green: #79a824
        # purple: #8e44ad
        if self.image == None:
            vehicle = plt.Circle((self.state[0], self.state[1]),self.size/2, facecolor=self.COLOR, edgecolor='black')
            ax.add_artist(vehicle)
            vehicle_photo = None
        else:
            vehicle = plt.Circle((self.state[0], self.state[1]),self.size/2, facecolor="None", edgecolor='black', linestyle='--')
            ax.add_artist(vehicle)
            img= self.image
            img= img.rotate(self.state[2]*180/np.pi,expand=1)
            vehicle_photo = AnnotationBbox(OffsetImage(img, zoom= self.ZOOM), (self.state[0], self.state[1]), frameon=False)
            ax.add_artist(vehicle_photo)

        dynamic_region = plt.Circle((self.state[0], self.state[1]),self.dynamic_offset/2, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.1)
        static_region = plt.Circle((self.state[0], self.state[1]),self.static_offset/2, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.3)
        ax.add_artist(static_region)
        ax.add_artist(dynamic_region)
         
        arrow = plt.arrow(self.state[0], self.state[1], 4*np.cos(self.state[2]), 4*np.sin(self.state[2]), width = 0.5, facecolor= self.COLOR, edgecolor='black')
        plt.plot(np.array(self.state_history)[:,0], np.array(self.state_history)[:,1], color = self.COLOR)

        plt.plot(self.start[0],self.start[1], marker='*',  markersize=20, color = self.COLOR, markeredgecolor='black')
        plt.plot(self.goal[0],self.goal[1], marker='D',  markersize=20, color = self.COLOR, markeredgecolor='black')

        return vehicle, arrow, vehicle_photo, static_region, dynamic_region
