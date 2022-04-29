
from matplotlib import pyplot as plt
from Obstacle import *
from Vehicle import *
from config import *
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
    
    # Define Plot Limits
    fig, ax = plt.subplots()
    fig.canvas.manager.full_screen_toggle()
    xlim = np.array([-18,278])
    ylim = np.array([-5, 155])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ws_limits = [[0,260], [0,140]]

    # Uncomment this for the Example of Big Simulation
    # plot.plot_map(ax)

    for vehicle in Vehicles:
        vehicle.global_plot(ax)
        for other_vehicle in Vehicles:
            if vehicle.id != other_vehicle.id:
                vehicle.Obstacles.append(other_vehicle)

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











