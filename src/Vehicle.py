
from PIL import Image
import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class Vehicle:

    def __init__(self, id, start, goal_list, inital_control_input, max_v, max_phi, max_delta, start_time, sampling_time, prediction_horizon, Obstacles, offset = [100, 250], VO_Type = "VO", COLOR = "#f1c40f", COLOR_NAME = 'Yellow', path = None,  ZOOM = 0.05):
     
        # Vehicle Details (Dimesions Inspired by Lamborgini Aventador)
        self.length = 2.70  # Wheelbase
        self.size = 4.78    # Vehicle Size (Diameter)
        self.type = "Vehicle"
        self.id = id
        self.collided = False
        self.Reach = False

        # Load Image from Path
        if path!=None:
            self.image = Image.open(path)
        else:
            self.image = None

        self.COLOR = COLOR  #Color of the Vehicle Path/ Sensing Region
        self.COLOR_NAME = COLOR_NAME
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

        self.goal_list = goal_list
        goal = self.goal_list.pop(0)
        
        x_goal = goal[0]
        y_goal = goal[1]
        theta_goal = goal[2]*np.pi/180
        delta_goal = goal[3]*np.pi/180
        self.goal = [x_goal, y_goal, theta_goal, delta_goal]

        if self.goal_list:
            self.goal_bound =  self.size * 2
        else:
            self.goal_bound = self.size
        

        self.state = [x_start, y_start, theta_start, delta_start]
        self.virtual_state = [x_start, y_start, theta_start, delta_start]

        '''
        # Control Input = [v, phi]
        # v : Velocity (km/h)
        # phi : Steering Rate (rad/s) 
        '''
        initial_v = inital_control_input[0] * 5/18
        inital_phi = inital_control_input[1]*np.pi/180
        self.control_input = [initial_v, inital_phi]

        # Constraints:
        self.max_v = max_v *5/18
        self.max_phi = max_phi * np.pi/180
        self.max_delta = max_delta * np.pi/180
        self.max_acc = 8
        self.max_dacc = 4

        # Obstacles
        self.Obstacles = Obstacles
        # Sensing range of the obstacles for the vehicle
        self.static_offset = offset[0]
        self.dynamic_offset = offset[1]
        self.VO_Type = VO_Type
        self.tau_vo = prediction_horizon + 5 # Here tau = 2*sampling_time (secs)

        # Simulation Parameters
        self.time = start_time
        self.prediction_horizon = prediction_horizon
        self.control_horizon = 1.0
        self.sampling_time = sampling_time

        # Path Parameters
        self.global_length = self.distance([x_start, y_start],[x_goal, y_goal])
        self.local_length = 0
        # History of the Vehicle
        self.state_history = [[x_start, y_start, theta_start, delta_start]]
        self.state_history.append([x_start, y_start, theta_start, delta_start])

        self.virtual_state_history = [[x_start, y_start, theta_start, delta_start]]

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
            pos_x = self.state[0]
            pos_y = self.state[1]
            theta = self.state[2]
            delta = self.state[3]
        else:
            pos_x = self.virtual_state[0]
            pos_y = self.virtual_state[1]
            theta = self.virtual_state[2]
            delta = self.virtual_state[3]
        
        # v and phi are given by the MPC
        state = [pos_x, pos_y, theta, delta]

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
            # Assign the State New Values
            self.state = state
            x=self.state[0]
            y=self.state[1]
            theta=self.state[2]
            delta=self.state[3]

            # Local Length
            self.local_length += self.distance([self.state[0], self.state[1]],[self.state_history[-1][0], self.state_history[-1][1]])
            
            # Update Time and History
            self.state_history.append([x, y, theta, delta])
            self.control_input_history.append([v,phi])
            self.time += self.sampling_time
            self.time_history.append(self.time)
        else:
            self.virtual_state = state
            x=self.virtual_state[0]
            y=self.virtual_state[1]
            theta=self.virtual_state[2]
            delta=self.virtual_state[3]
            self.virtual_state_history.append([x, y, theta, delta])
        

    def goal_reach_cost(self):
        # State cost to Goal
        cost_xy = self.distance(self.virtual_state,self.goal)**2
        cost_theta = (self.virtual_state[2]-self.goal[2])**2
        cost_delta = (self.virtual_state[3]-self.goal[3])**2

        # Weights
        W1 = 1
        W2 = 1
        W3 = 0

        goal_reach_cost = W1*cost_xy + W2*cost_theta + W3*cost_delta
        # print("Goal Reach Cost",goal_reach_cost)
        return goal_reach_cost

    
    def smoothness_cost(self, virtual_input):
        cost_v = ((self.control_input[0] - virtual_input[0])**2/self.sampling_time)
        cost_phi = ((self.control_input[1] - virtual_input[1])**2/self.sampling_time)

        # Weights
        W1 = 1
        W2 = 0.1

        smoothness_cost = W1*cost_v + W2*cost_phi 

        return smoothness_cost

    def constraint_cost(self, virtual_input, W_v = 10, W_phi = 1, W_delta=100, W_ws = 1, W_safety = 10):
        lamda_v = max(0,virtual_input[0] - self.max_v) + max(0,-virtual_input[0] - self.max_v)
        lamda_phi = max(0,virtual_input[1] - self.max_phi) + max(0,-virtual_input[1] - self.max_phi)
        lamda_delta = max(0,virtual_input[1] - self.max_delta) + max(0,-virtual_input[1] - self.max_delta)
        lamda_ws = 0 #Both x and y workspace constraints

        constraint_cost = W_v*lamda_v + W_phi*lamda_phi + W_delta*lamda_delta + W_ws*lamda_ws # + W_safety*lambda_safety
        
        return constraint_cost
    
    def obstacle_cost(self, W_static = 10000, W_dynamic = 10000, W_safety = 100):
        # W_dynamic = 0
        obstacle_cost = 0
        lambda_safety = 0
        for obstacle in self.Obstacles:
            if obstacle.type != "Vehicle":
                obs_x = obstacle.parameters[0]
                obs_y = obstacle.parameters[1]
                radius = obstacle.parameters[2]
            else:
                if self.id < obstacle.id: #This condition ensures that the vehicle doesnt know the updated state of other vehicle
                    other_vehicle_state = obstacle.state_history[-1][0:3]
                else:
                    other_vehicle_state = obstacle.state_history[-2][0:3]
                obs_x = other_vehicle_state[0]
                obs_y = other_vehicle_state[1]
                radius = obstacle.size/2

            dist = self.distance([obs_x,obs_y],[self.state[0],self.state[1]])
            dist_virtual = self.distance([obs_x,obs_y],[self.virtual_state[0],self.virtual_state[1]])
            if (-dist + (radius + self.size/2)>0 and not obstacle.collided):
                if obstacle.type != "Vehicle":
                    print("Collision Has Occured between Vehicle {} and {} Obstacle {}".format(self.COLOR_NAME, obstacle.type, obstacle.id))
                else:
                    print("Collision Has Occured between Vehicle {} and {} {}".format(self.COLOR_NAME, obstacle.type, obstacle.COLOR_NAME))
                obstacle.collided = True
                plt.plot(self.state[0],self.state[1],"-x", markersize=14, markeredgewidth=3, color = 'black')

            if obstacle.type == 'Static':
                if dist < self.static_offset:
                    obstacle_cost += 1/dist_virtual * W_static
                    lambda_safety += max(0,-dist_virtual + (radius + self.size/2)*1.25) * W_static
            else:
                if dist < self.dynamic_offset:
                    obstacle_cost += 1/dist_virtual * W_dynamic
                    lambda_safety += max(0,-dist_virtual + (radius + self.size/2)*1.25) * W_dynamic
        
        obstacle_cost += lambda_safety
        # print("Obstacle Cost",obstacle_cost*1000)

        
        return obstacle_cost

    def velocity_obstacle_cost (self, virtual_input, acceleration, W_dynamic_VO, W_safety_VO):
        
        velocity_obstacle_cost = 0

        for i in range(self.tau_vo - self.prediction_horizon):
            self.bicycle_model(virtual_input[0] + acceleration[0] * self.sampling_time * (i+self.prediction_horizon) ,virtual_input[1] + acceleration[1] * self.sampling_time * (i+self.prediction_horizon))

        for obstacle in self.Obstacles:
            if obstacle.type != "Static":
                if obstacle.type == "Dynamic":
                    obs_x = obstacle.parameters[0]
                    obs_y = obstacle.parameters[1]
                    radius = obstacle.parameters[2]
                    obs_v = obstacle.parameters[3]
                    obs_angle = obstacle.parameters[4]
                    if self.VO_Type == "RVO" or self.VO_Type == "RAVO":
                        obs_acceleration = obstacle.parameters[5]
                    else:
                        obs_acceleration = 0
                elif obstacle.type == "Vehicle":
                    if self.id < obstacle.id: #This condition ensures that the vehicle doesnt know the updated state of other vehicle
                        other_vehicle_state = obstacle.state_history[-1][0:3] #Only taking x,y & theta and not the steering angle
                        other_vehicle_velocity = obstacle.control_input_history[-1][0] #Only Taking Velocity and not the steering rate
                        if self.VO_Type == "RVO" or self.VO_Type == "RAVO":
                            if len(obstacle.control_input_history) > 2: 
                                other_vehicle_velocity_prev = obstacle.control_input_history[-2][0]
                            else:
                                other_vehicle_velocity_prev = other_vehicle_velocity
                        else:
                            other_vehicle_velocity_prev = other_vehicle_velocity
                    else:
                        other_vehicle_state = obstacle.state_history[-2][0:3]
                        other_vehicle_velocity = obstacle.control_input_history[-2][0]
                        if self.VO_Type == "RVO" or self.VO_Type == "RAVO":
                            if len(obstacle.control_input_history) > 2: 
                                other_vehicle_velocity_prev = obstacle.control_input_history[-3][0]
                            else:
                                other_vehicle_velocity_prev = other_vehicle_velocity
                        else:
                            other_vehicle_velocity_prev = other_vehicle_velocity

                    obs_x = other_vehicle_state[0]
                    obs_y = other_vehicle_state[1]
                    radius = obstacle.size/2
                    obs_v = other_vehicle_velocity
                    obs_angle = other_vehicle_state[2]
                    obs_acceleration =  (other_vehicle_velocity - other_vehicle_velocity_prev)/obstacle.sampling_time
                
                dist = self.distance([obs_x,obs_y],[self.state[0],self.state[1]])

                if dist < self.dynamic_offset:
                    virtual_obstacle = []
                    # Assuming Holonomic Model for the Obstacle
                    for i in range(self.tau_vo):
                        obs_v += obs_acceleration * obstacle.sampling_time
                        obs_x = obs_x + obs_v * np.cos(obs_angle) * self.sampling_time
                        obs_y = obs_y + obs_v * np.sin(obs_angle) * self.sampling_time
                        virtual_obstacle.append([obs_x, obs_y])

                    dist_X = (np.array(self.virtual_state_history)[:self.tau_vo,0] - np.array(virtual_obstacle)[:,0])**2
                    dist_Y = (np.array(self.virtual_state_history)[:self.tau_vo,1] - np.array(virtual_obstacle)[:,1])**2

                    dist_virtual = (np.sqrt(dist_X + dist_Y))
                    lambda_safety = np.sum(np.maximum(0,-dist_virtual + (radius + self.size/2)*1.2))

                    velocity_obstacle_cost += np.sum(1/dist_virtual) * W_dynamic_VO + lambda_safety * W_safety_VO

        return velocity_obstacle_cost

    def total_cost(self, virtual_input):
        total_cost = 0
        x=self.state[0]
        y=self.state[1]
        theta=self.state[2]
        delta=self.state[3]
    
        self.virtual_state = [x,y,theta,delta]
        self.virtual_state_history = []

        acceleration = [0,0]
        curr_virtual_input = [0,0]
        if self.VO_Type == "AVO" or self.VO_Type == "RAVO":
            acceleration[0] = (virtual_input[0] - self.control_input[0])/self.sampling_time
            if acceleration[0] > self.max_acc:
                acceleration[0] = self.max_acc
            elif acceleration[0] < -self.max_dacc:
                acceleration[0] = -self.max_dacc
 
            acceleration[1] = (virtual_input[1] - self.control_input[1])/self.sampling_time

        curr_virtual_input[0] = virtual_input[0]
        curr_virtual_input[1] = virtual_input[1]
          
        # Calculating cost across the prediction horizon
        for i in range(self.prediction_horizon):
            # Weights
            W_goal_reach = 5
            W_smoothness = 1000
            W_constraint = 10000
            W_static = 8000

            W_safety = 0 #10000
            W_dynamic = 0 #25000

            W_safety_VO = 100000
            W_dynamic_VO = 8000

            decceleration = 7

            # Adding a Terminal Cost for the final prediction horizon
            if i == self.prediction_horizon-1:
                terminal_cost = (self.goal_reach_cost()+(self.virtual_state[2]-self.goal[2])**2)*W_goal_reach*4
            else:
                terminal_cost = 0
            
            velocity_cost = 0
            if not self.goal_list:
                if self.distance(self.virtual_state,self.goal) < 30:
                        velocity_cost += curr_virtual_input[0]**2 * decceleration * 150
            
            velocity_cost += curr_virtual_input[0]**2 * decceleration

            total_cost += velocity_cost + self.goal_reach_cost() * W_goal_reach + self.smoothness_cost(virtual_input)*W_smoothness + self.constraint_cost(curr_virtual_input) * W_constraint + self.obstacle_cost(W_static, W_dynamic, W_safety) + terminal_cost
            curr_virtual_input[0] = virtual_input[0] + acceleration[0] * self.sampling_time * i
            curr_virtual_input[1] = virtual_input[1] + acceleration[1] * self.sampling_time * i
           
            # Update the virtual state to next prediction horizon
            self.bicycle_model(curr_virtual_input[0], curr_virtual_input[1])
        
        total_cost += self.velocity_obstacle_cost(virtual_input, acceleration, W_dynamic_VO, W_safety_VO)
                
        return total_cost

    def optimizer(self, iteration = 20, learning_rate = 0.005, decay = 0.9, eps = 1e-8):
        # Using Autograd to find the gradient
        gradient = grad(self.total_cost)
        mean_square_gradient = np.asarray([0.0, 0.0])
        virtual_input = self.control_input #Assigning the virtual input

        # Performing Optimization using RMS Prop
        for _ in range(iteration):
            cost_gradient = np.asarray(gradient(virtual_input))
            mean_square_gradient = (decay * mean_square_gradient) + (1-decay) * (cost_gradient**2)
            virtual_input = virtual_input - (learning_rate/((mean_square_gradient)**0.5+eps))*cost_gradient
        
        # Assigning the optimized virtual input to vehicle
        if self.VO_Type == "RVO" or self.VO_Type == "RAVO":
            self.control_input = (self.control_input + virtual_input)/2
        else:
            self.control_input = virtual_input
          

    def plot(self, ax, virtual_state_flag = True):
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
            
        if virtual_state_flag == True and self.Reach == False:
            virtual_states = plt.scatter(np.array(self.virtual_state_history)[0:self.prediction_horizon,0]._value, np.array(self.virtual_state_history)[0:self.prediction_horizon,1]._value, edgecolor = 'black' ,color = self.COLOR)
            dynamic_region = plt.Circle((self.state[0], self.state[1]),self.dynamic_offset, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.2)
            static_region = plt.Circle((self.state[0], self.state[1]),self.static_offset, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.2)
        else:
            virtual_states = plt.scatter(self.state[0], self.state[1], color = self.COLOR)
            dynamic_region = plt.Circle((self.state[0], self.state[1]),0, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.2)
            static_region = plt.Circle((self.state[0], self.state[1]),0, facecolor= self.COLOR, edgecolor='black', linestyle=':', alpha = 0.2)

        ax.add_artist(static_region)
        ax.add_artist(dynamic_region)
         
        arrow = plt.arrow(self.state[0], self.state[1], 4*np.cos(self.state[2]), 4*np.sin(self.state[2]), width = 0.6, facecolor= self.COLOR, edgecolor='black')
        plt.plot(np.array(self.state_history)[:,0], np.array(self.state_history)[:,1], color = self.COLOR)

        return vehicle, virtual_states, arrow, vehicle_photo, static_region, dynamic_region

    def global_plot(self,ax):

        plt.plot(self.start[0],self.start[1], marker='*',  markersize=25, color = self.COLOR, markeredgecolor='black')
        plt.plot([self.start[0], self.goal[0]], [self.start[1], self.goal[1]], '--', color = self.COLOR, alpha = 0.5 )
        if len(self.goal_list)>0:
            plt.plot([self.goal_list[0][0], self.goal[0]], [self.goal_list[0][1], self.goal[1]], '--', marker = 's', markersize=10, color = self.COLOR, markeredgecolor='black', alpha = 0.5 )
            plt.plot(np.array(self.goal_list)[:,0], np.array(self.goal_list)[:,1], '--', marker = 's',  markersize=10, color = self.COLOR, markeredgecolor='black', alpha = 0.5)
            plt.plot(self.goal_list[-1][0],self.goal_list[-1][1], marker='D',  markersize=20, color = self.COLOR, markeredgecolor='black')
        else:
            plt.plot(self.goal[0],self.goal[1], marker='D',  markersize=20, color = self.COLOR, markeredgecolor='black')
        
