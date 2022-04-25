
from matplotlib import pyplot as plt
from Obstacle import *
from Vehicle import *
import plot

def in_bound_region(Vehicle):
    '''
    Check if the Vehicle has reached within the Goal Bounds
    '''
    if Vehicle.distance(Vehicle.state,Vehicle.goal) < vehicle.goal_bound:
        if vehicle.goal_list:
            # print("Goal Update",Vehicle.distance(Vehicle.state,Vehicle.goal))
            new_goal = vehicle.goal_list.pop(0)
            vehicle.global_length += vehicle.distance([new_goal[0], new_goal[1]], [vehicle.goal[0], vehicle.goal[1]])
            vehicle.goal = new_goal
            vehicle.goal[2] *= np.pi/180
            if not vehicle.goal_list:
                vehicle.goal_bound = vehicle.size
        else:
            if Vehicle.Reach == False:
                vehicle.local_length += vehicle.goal_bound
            Vehicle.Reach = True

    return Vehicle.Reach


def end_loop(Vehicles):
    '''
    Takes in the list of Vehicle Objects. 
    Returns True if All vehicles are in bound region
    Returns False if vehicles >=1 hasn't reached goal

    Can be expanded to add collision detection
    '''
    flag = True
    for vehicle in Vehicles:
        if not in_bound_region(vehicle):
            flag = False

    return flag

if __name__== '__main__':
    
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

    # Obstacle_1 = Obstacle(1, 
    #                      "Static", 
    #                      [1000, 50, 2, 0, 0], 
    #                      sampling_time) 
    #                     #  "../Object_Photos/police.png"
                        #  )
    Obstacle_1 = Obstacle(1, "Static", [130, 70, 7.5, 0.0, 0.0, 0.0], sampling_time)
    # Obstacle_2 = Obstacle(2, "Static", [210.0, 50.0, 2.0, 0.0, 0.0, 0.0], sampling_time)

    # Obstacle_2 = Obstacle(1, "Static", [112.5, 22.5, 2.0, 0, 0, 0], sampling_time)
    # Obstacle_1 = Obstacle(1, "Dynamic", [20.0, 0.0, 2.5, 22, 45, 0], sampling_time, "../Object_Photos/taxi.png") 
    # Obstacle_3 = Obstacle(3, "Dynamic", [60.0, 140.0, 2.5, 0, 270, 5], sampling_time, "../Object_Photos/police.png") 
    Obstacle_3 = Obstacle(3, "Dynamic", [40.0, 50.0, 2.5, 68, 90, 0], sampling_time, "../Object_Photos/police.png") 
    Obstacle_4 = Obstacle(4, "Dynamic", [70.0, 120.0, 2.5, 50, 180, 4], sampling_time, "../Object_Photos/taxi.png") 
    Obstacle_5 = Obstacle(5, "Dynamic", [70.0, 100.0, 2.5, 15, 0, 3], sampling_time, "../Object_Photos/police.png")
    Obstacle_6 = Obstacle(6, "Dynamic", [60.0, 70.0, 2.5, 45, 270, 5], sampling_time, "../Object_Photos/taxi.png")
    Obstacle_7 = Obstacle(7, "Dynamic", [200.0, 90.0, 2.5, 9, 270, 3], sampling_time, "../Object_Photos/police.png")
    Obstacle_8 = Obstacle(8, "Dynamic", [220.0, 50.0, 2.5, 10, 90, 3], sampling_time, "../Object_Photos/taxi.png") 
    # Obstacle_2 = Obstacle(1, "Dynamic", [170.0, 45.0, 2.0, 10, 180, 3], sampling_time, "../Object_Photos/police.png")
    # Obstacle_1 = Obstacle(1, "Dynamic", [0.0, 0.0, 2.0, 31, 45], sampling_time, "../Object_Photos/police.png")
    # Obstacle_2 = Obstacle(2, "Dynamic", [200.0, 0.0, 2.0, 30.0, 155], sampling_time, "../Object_Photos/taxi.png")                        
    # Obstacle_2 = Obstacle(2, "Static", [110, 25, 2, 0, 0], sampling_time)
    # Obstacle_3 = Obstacle(3, [50, 50, 2, 0, 0])
    # Obstacle_4 = Obstacle(4, [ 50, 50, 2, 10, 270], "Dynamic", "../Object_Photos/police.png")
    # Obstacle_5 = Obstacle(5, [ 25, 50, 2, 10, -90], "Dynamic", "../Object_Photos/police.png")
    # Obstacles = []
    # for i in range(10):bstacle_1
    #     for j in range(1):
    #         Obstacles.append(Obstacle(i+j, "Static", [i*5+50, j*5+50, 1, 0, 0], sampling_time))

    #Form a List (Do not forget to add Obstacles in this list)
    Obstacles = [Obstacle_1, Obstacle_3, Obstacle_4, Obstacle_5, Obstacle_6, Obstacle_7, Obstacle_8]

    '''
    Define the Vehicle (Read Vehicle Class for Better understanding)
    Sample:
    Vehicle_"Name" = Vehicle("id",
                            "Start States", 
                            "Goal States",
                            "Initial Control Input",
                            "Velocity Limit(Km/h)",
                            "Steering Rate Limit (Deg/s)",
                            "Start Time",
                            "Sampling Time",
                            "Prediction Horizon 
                            ["Obstacle 1","Obstacle 2", ... ] 
                            ["Static Offset Region", "Dynamic Offset Region"]
                            "Color of Path/Vehicle"
                            "Image Path"
                            "Zoom for the Photo")
    '''
    # Define Start, Goal States and Control Input 
    # (States include x,y position, orientation, steering angle)
    # (Inputs Include velocity and steering angle)
    
    Start_1 = [10.0, 110.0, 0.0, 0.0]
    Goal_1 = [[75.0, 110.0, 0.0, 0.0], [185.0, 110.0, 0.0, 0.0], [210.0, 90.0, -90.0, 0.0], [210.0, 10.0, -90.0, 0.0]] #[90.0 ,45.0, 0.0, 0.0]
    inital_control_input_1 = [30.0,0.0]

    # Start_1 = [0.0, 200.0, 0.0, 0.0]
    # Goal_1 = [[400.0, 200.0, 0.0, 0.0]] #, [135.0, 0.0, 0.0, 0.0], [180.0, 0.0, 0.0, 0.0]] #[90.0 ,45.0, 0.0, 0.0]
    # inital_control_input_1 = [20.0,0.0]

    start_time_1 = 0

    start_time_1 = 0
    Vehicle_Yellow = Vehicle(1, 
                        Start_1,Goal_1, 
                        inital_control_input_1, 
                        120.0, 10, 30,
                        start_time_1, 
                        sampling_time, 
                        prediction_horizon, 
                        [Obstacle_1, Obstacle_3], 
                        [40, 80],
                        "AVO", 
                        "#f1c40f",
                        "Yellow", 
                        "../Object_Photos/aventador_y.png", 
                        ZOOM=0.01)
       
    
    Start_2 = [240.0, 30, 180.0, 0.0]
    Goal_2 = [[185.0, 30, 180.0, 0.0], [130.0, 30, 180.0, 0.0], [100.0, 50, 135.0, 0.0], [100.0, 90.0, 90.0, 0.0],[130.0, 110.0, 0.0, 0.0], [185.0, 110.0, 0.0, 0.0], [250.0, 110.0, 0.0, 0.0]] #  [75.0, 110.0, 135.0, 0.0], [50.0, 130.0, 135.0, 0.0], [50.0, 145.0, 90, 0.0]]
    inital_control_input_2 = [30.0,0.0]

    start_time_2 = 0

    Vehicle_Red = Vehicle(2, 
                        Start_2,Goal_2, 
                        inital_control_input_2, 
                        120.0, 10, 30,
                        start_time_2, 
                        sampling_time, 
                        prediction_horizon, 
                        [Obstacle_1], 
                        [40, 80],
                        "VO", 
                        "red",
                        "Red",
                        "../Object_Photos/ferrari_2.png", 
                        ZOOM=0.03)

    Start_3 = [10.0, 30, 0.0, 0.0]
    Goal_3 = [[75.0, 30, 0.0, 0.0], [185.0, 30, 0.0, 0.0], [250, 30.0, 0.0, 0.0]]
    inital_control_input_3 = [35.0,0.0]

    start_time_3 = 0

    Vehicle_Light_Blue = Vehicle(3, 
                        Start_3,Goal_3, 
                        inital_control_input_3, 
                        120.0, 10, 30,
                        start_time_3, 
                        sampling_time, 
                        prediction_horizon, 
                        [Obstacle_1], 
                        [40,80],
                        "AVO",
                        "#3498db",
                        "Light_Blue",
                        "../Object_Photos/ferrari_3.png", 
                        ZOOM=0.01)

    
    Start_4 = [220.0, 140.0, 225.0, 0.0]#[210.0, 145.0, 270.0, 0.0]
    Goal_4 = [[210.0, 130.0, 225.0, 0.0], [185.0, 110.0, 270.0, 0.0], [160.0, 90.0, -45.0, 0.0], [160.0, 50, 135.0, 0.0], [130.0, 30, 180.0, 0.0], [75.0, 30, 180.0, 0.0], [50.0, 10.0, 225.0, 0.0]]
    inital_control_input_4 = [20.0,0.0]

    start_time_4 = 0

    Vehicle_Green = Vehicle(4, 
                        Start_4,Goal_4, 
                        inital_control_input_4, 
                        120.0, 10, 30,
                        start_time_4, 
                        sampling_time, 
                        prediction_horizon, 
                        [Obstacle_1, Obstacle_5], 
                        [40, 80],
                        "VO",
                        "#79a824",
                        "Green",
                        "../Object_Photos/aventador_g.png", 
                        ZOOM=0.015)

    # Start_5 = [400.0, 200.0, 180.0, 0.0]
    # Goal_5 = [[0.0, 200.0, 180.0, 0.0]]
    # inital_control_input_5 = [20.0,0.0]

    # start_time_5 = 0

    # Vehicle_Grey = Vehicle(5, 
    #                     Start_5,Goal_5, 
    #                     inital_control_input_5, 
    #                     120.0, 10, 30,
    #                     start_time_5, 
    #                     sampling_time, 
    #                     prediction_horizon, 
    #                     [], 
    #                     [80, 120],
    #                     "AVO",
    #                     "#808080",
    #                     "Grey") 
    #                     # "../Object_Photos/ferrari_4.png", 
    #                     # ZOOM=0.025)
    
    # Start_6 = [341.42, 341.42, 225.0, 0.0]
    # Goal_6 = [[58.58, 58.58, 225.0, 0.0]]
    # inital_control_input_6 = [20.0,0.0]

    # start_time_6 = 0

    # Vehicle_Light_Blue = Vehicle(6, 
    #                     Start_6,Goal_6, 
    #                     inital_control_input_6, 
    #                     120.0, 10, 30,
    #                     start_time_5, 
    #                     sampling_time, 
    #                     prediction_horizon, 
    #                     [], 
    #                     [80, 120],
    #                     "AVO",
    #                     "#3498db",
    #                     "Light_Blue") 
    #                     # "../Object_Photos/ferrari_1.png", 
    #                     # ZOOM=0.025)
    
    # Start_7 = [200.0, 400.0, 270.0, 0.0]
    # Goal_7 = [[200.0, 0.0, 270.0, 0.0]]
    # inital_control_input_7 = [20.0,0.0]

    # start_time_7 = 0

    # Vehicle_Dark_Red = Vehicle(7, 
    #                     Start_7,Goal_7, 
    #                     inital_control_input_7, 
    #                     120.0, 10, 30,
    #                     start_time_7, 
    #                     sampling_time, 
    #                     prediction_horizon, 
    #                     [], 
    #                     [80, 120],
    #                     "AVO",
    #                     "#922b21",
    #                     "Dark_Red") 
    #                     # "../Object_Photos/ferrari_3.png", 
    #                     # ZOOM=0.025)

    # Start_8 = [58.58, 341.42, 315.0, 0.0]
    # Goal_8 = [[341.42, 58.58, 315.0, 0.0]]
    # inital_control_input_8 = [20.0,0.0]

    # start_time_8 = 0

    # Vehicle_Police = Vehicle(8, 
    #                     Start_8,Goal_8, 
    #                     inital_control_input_8, 
    #                     120.0, 10, 30,
    #                     start_time_8, 
    #                     sampling_time, 
    #                     prediction_horizon, 
    #                     [], 
    #                     [80, 120],
    #                     "AVO",
    #                     "black",
    #                     "Black") 
    #                     # "../Object_Photos/police.png", 
    #                     # ZOOM=0.025)

    #Form a List (Do not forget to add all Vehicles in this list)
    Vehicles = [Vehicle_Yellow, Vehicle_Red, Vehicle_Light_Blue, Vehicle_Green] #, Vehicle_Green, Vehicle_Grey, Vehicle_Dark_Red, Vehicle_Light_Blue, Vehicle_Police] #, Vehicle_2, Vehicle_3] #, Vehicle_3, Vehicle_4]#, Vehicle_C]

    # Define Plot Limits
    fig, ax = plt.subplots()
    fig.canvas.manager.full_screen_toggle()
    xlim = np.array([-18,278])
    ylim = np.array([-5, 155])
    # ylim = xlim * (14.5/26.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ws_limits = [[0,260], [0,140]]

    plot.plot_map(ax)

    for vehicle in Vehicles:
        vehicle.global_plot(ax)
        for other_vehicle in Vehicles:
            if vehicle.id != other_vehicle.id:
                vehicle.Obstacles.append(other_vehicle)
    
    # Plot an background image on the graph
    # img = cv2.imread('road.jpg')		# this is read in BGR format
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)		# this converts it into RGB
    # ax.imshow(img, extent=[-10, 110,-5, 55])

    # Plot the Static Obstacles
    for obstacle in Obstacles:
        if obstacle.type == 'Static':
            obstacle.plot(ax)

    # Simulation Runs Till All Robots Do Not Reach Goal Configuration 
    while(not end_loop(Vehicles)): #not end_loop(Vehicles)):

        for vehicle in Vehicles:
            if not in_bound_region(vehicle):
                vehicle.optimizer()
                vehicle.bicycle_model(vehicle.control_input[0], vehicle.control_input[1], True)
                
        for obstacle in Obstacles:
            if obstacle.type == 'Dynamic':
                if obstacle.parameters[0] < ws_limits[0][1] and obstacle.parameters[0] > ws_limits[0][0] and obstacle.parameters[1] > ws_limits[1][0] and obstacle.parameters[1] < ws_limits[1][1]: 
                    obstacle.Model()

        plot.plot_simulation(Vehicles, Obstacles, ax)
    
    for vehicle in Vehicles:
        print("Vehicle {}:".format(vehicle.COLOR_NAME))
        print(" Global Path Length = {} m".format(vehicle.global_length))
        print(" Local Path Length = {} m".format(vehicle.local_length))
        print(" Time Taken = {} secs".format(vehicle.time))

    plot.plot_simulation(Vehicles, Obstacles, ax, True, False, Experiment_No)
    plot.save_graph(Vehicles, Experiment_No)











