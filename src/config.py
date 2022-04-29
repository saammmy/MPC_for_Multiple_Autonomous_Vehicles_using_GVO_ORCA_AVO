from Obstacle import *
from Vehicle import *

# Experiment No
Experiment_No = 1
# Simulation Parameters:
sampling_time = 0.2
prediction_horizon = 15

'''
Define the Obstacles (Read Obstacle Class for Better understanding)
Sample:
Obstacle_"Name" = Obstacle  (id, 
                            "Obstacle Type",
                            ["X Position", "Y Position", "Radius", "Velocity(Km/h)", "Angle(Deg)", Acceleration (m/s^2)], 
                            "Sampling Time", 
                            "Image Path",
                            "Color of Obstacle",
                            "Zoom for the Photo")
'''

Obstacle_1 = Obstacle(1, "Static", [120, 60, 2, 0.0, 0.0, 0.0], sampling_time)
Obstacle_2 = Obstacle(2, "Dynamic", [100.0, 10.0, 2.5, 30, 135.0, 1.0], sampling_time, "../Object_Photos/police.png") 

#Form a List (Do not forget to add Obstacles in this list and also add them to the Vehicles Parameters
Obstacles = [Obstacle_1, Obstacle_2]

'''
Define the Vehicle (Read Vehicle Class for Better understanding)
Sample:
Vehicle_"Name" = Vehicle("id",
                        "Start States", 
                        "Goal States",
                        "Initial Control Input",
                        "Velocity Limit(Km/h)", "Steering Rate Limit (Deg/s)", "Steering Angle Limit (Deg)",
                        "Start Time",
                        "Sampling Time",
                        "Prediction Horizon",
                        ["Obstacle 1","Obstacle 2", ... ] 
                        ["Static Offset Region", "Dynamic Offset Region"]
                        "Velocity Obstacle Type",
                        "Color of Path/Vehicle"
                        "Image Path"
                        "Zoom for the Photo")
'''
# Define Start, Goal States and Control Input 
# (States include x,y position, orientation, steering angle)
# (Inputs Include velocity and steering angle)

Start_1 = [20.0, 10.0, 30.0, 0.0]
Goal_1 = [[40.0, 20.0, 30.0, 0.0], [200.0, 100.0, 30.0, 0.0]]
inital_control_input_1 = [30.0,0.0]

start_time_1 = 0

start_time_1 = 0
Vehicle_Yellow = Vehicle(1, 
                    Start_1,Goal_1, 
                    inital_control_input_1, 
                    120.0, 10, 30,
                    start_time_1, 
                    sampling_time, 
                    prediction_horizon, 
                    [Obstacle_1, Obstacle_2], 
                    [40, 80],
                    "VO", 
                    "#f1c40f",
                    "Yellow", 
                    "../Object_Photos/aventador_y.png", 
                    ZOOM=0.01)
    

Start_2 = [220.0, 110.0, 215.0, 0.0]
Goal_2 = [[190.0, 95.0, 215.0, 0.0], [10, 5, 215.0, 0.0]]
inital_control_input_2 = [30.0,0.0]

start_time_2 = 0

Vehicle_Red = Vehicle(2, 
                    Start_2,Goal_2, 
                    inital_control_input_2, 
                    120.0, 10, 30,
                    start_time_2, 
                    sampling_time, 
                    prediction_horizon, 
                    [Obstacle_1, Obstacle_2], 
                    [40, 80],
                    "VO", 
                    "red",
                    "Red",
                    "../Object_Photos/ferrari_2.png", 
                    ZOOM=0.03)



#Form a List of Vehicles (Do not forget to add all Vehicles in this list)
Vehicles = [Vehicle_Yellow, Vehicle_Red]