from matplotlib import pyplot as plt
import numpy as np

def plot_simulation(Vehicles, Obstacles, ax, save = False, virtual_state_flag = True, Experiment_No = 1):
    VEHICLE_PLOT, VIRTUAL_STATE_PLOT, ARROW_PLOT, VEHICLE_PHOTO_PLOT, STATIC_REGION_PLOT, DYNAMIC_REGION_PLOT = ([] for i in range(6))
    OBSTACLE_FIG, OBSTACLE_PHOTO = ([] for i in range(2))

    for obstacle in Obstacles:
        if obstacle.type != 'Static':
            obstacle_fig, obstacle_photo = obstacle.plot(ax)
            OBSTACLE_FIG.append(obstacle_fig)
            OBSTACLE_PHOTO.append(obstacle_photo)
    
    for vehicle in Vehicles:
        vehicle_plot, virtual_state_plot, arrow_plot, vehicle_photo_plot, static_region_plot, dynamic_region_plot = vehicle.plot(ax, virtual_state_flag)
        VEHICLE_PLOT.append(vehicle_plot)
        VIRTUAL_STATE_PLOT.append(virtual_state_plot)
        ARROW_PLOT.append(arrow_plot)
        VEHICLE_PHOTO_PLOT.append(vehicle_photo_plot)
        STATIC_REGION_PLOT.append(static_region_plot)
        DYNAMIC_REGION_PLOT.append(dynamic_region_plot)

    plt.draw()

    if save == True:
        plt.pause(5)
        plt.savefig('../Graphs/Experiment_{}/Simulation_Result.png'.format(Experiment_No), bbox_inches='tight')
    else:
        plt.pause(Vehicles[0].sampling_time*0.01)


    for i in range(len(VEHICLE_PLOT)):
        VEHICLE_PLOT[i].remove()
        VIRTUAL_STATE_PLOT[i].remove()
        ARROW_PLOT[i].remove()
        STATIC_REGION_PLOT[i].remove()
        DYNAMIC_REGION_PLOT[i].remove()
        if VEHICLE_PHOTO_PLOT[i] != None:
            VEHICLE_PHOTO_PLOT[i].remove()

    for i in range(len(OBSTACLE_FIG)):
        OBSTACLE_FIG[i].remove()
        if OBSTACLE_PHOTO[i] != None:
            OBSTACLE_PHOTO[i].remove()
    

def save_graph(Vehicles, Experiment_No):
    for i in range(len(Vehicles)):
        vehicle = Vehicles[i]  
        control_input_history = np.array(vehicle.control_input_history)
        state_history = np.array(vehicle.state_history)
        time_history = np.array(vehicle.time_history)
        residual_state_history = np.array(vehicle.state_history)-np.array(vehicle.goal)
            
        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Velocity")
        plt.ylabel('Velocity(Km/hr)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,control_input_history[:,0]*18/5)
        plt.axhline(y=vehicle.max_v*18/5, color='r', linestyle='--')
        plt.axhline(y=-vehicle.max_v*18/5, color='r', linestyle='--')
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Velocity.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')
        
        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Steering Rate")
        plt.ylabel('Steering Rate (deg/s)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,control_input_history[:,1]*180/np.pi)
        plt.axhline(y=vehicle.max_phi*180/np.pi, color='r', linestyle='--')
        plt.axhline(y=-vehicle.max_phi*180/np.pi, color='r', linestyle='--')
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Steering_Rate.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')

        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Distance To Goal")
        plt.ylabel('Residual Position (m)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,(residual_state_history[:,0]**2+residual_state_history[:,1]**2)**0.5)
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Distance_To_Goal.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')

        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Residual Orientation")
        plt.ylabel('Residual Theta (deg)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,residual_state_history[:,2]*180/np.pi)
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Residual_Orientation.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')

        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Orientation Angle")
        plt.ylabel('Theta (deg)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,state_history[:,2]*180/np.pi)
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Orientation_Angle.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')

        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Steering Angle")
        plt.ylabel('Delta (deg)')
        plt.xlabel('Time (sec)')
        plt.plot(time_history,state_history[:,3]*180/np.pi)
        plt.axhline(y=vehicle.max_delta*180/np.pi, color='r', linestyle='--')
        plt.axhline(y=-vehicle.max_delta*180/np.pi, color='r', linestyle='--')
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Steering_Angle.png'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')

        plt.figure()
        plt.suptitle('Vehicle {}'.format(vehicle.COLOR_NAME),fontweight="bold")
        plt.title("Distance To Obstacles")
        plt.ylabel('Distance (m)')
        plt.xlabel('Time (sec)')
        for j in range(len(vehicle.Obstacles)):
            obstacle=vehicle.Obstacles[j]
            if obstacle.type == "Static":
                residual_history = np.array(vehicle.state_history)[:,0:2]-np.array(obstacle.history)[0,0:2]
                plt.plot(time_history,(residual_history[:,0]**2 + residual_history[:,1]**2)**0.5 - (vehicle.size/2 + obstacle.parameters[2]), label = 'Obstacle {} (Static)'.format(j+1))
            elif obstacle.type == "Dynamic":
                size = min(len(vehicle.state_history),len(obstacle.history))
                residual_history = np.array(vehicle.state_history)[:size,0:2]-np.array(obstacle.history)[:size,0:2]
                plt.plot(time_history[0:size],(residual_history[:size,0]**2+residual_history[:,1]**2)**0.5 - (vehicle.size/2 + obstacle.parameters[2]), label = 'Obstacle {} (Dynamic)'.format(j+1))
            elif obstacle.type == "Vehicle":
                size = min(len(vehicle.state_history),len(obstacle.state_history))
                residual_history = np.array(vehicle.state_history)[0:size,0:2]-np.array(obstacle.state_history)[0:size,0:2]
                plt.plot(time_history[0:size],(residual_history[:,0]**2+residual_history[:,1]**2)**0.5 - (vehicle.size + obstacle.size)/2, label = 'Obstacle {} (Vehicle)'.format(j+1))
        plt.legend()
        plt.axhline(y=0, color='r', linestyle='--')
        plt.savefig('../Graphs/Experiment_{}/Vehicle_{}/Vehicle_{}_Distance_To_Obstacles (m)'.format(Experiment_No,vehicle.COLOR_NAME,vehicle.COLOR_NAME), bbox_inches='tight')
        
def plot_map(ax):
    map = plt.Rectangle((0,0), 260, 150, facecolor = "None", edgecolor = 'black')
    box1 = plt.Rectangle((0,0), 30, 10, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box2 = plt.Rectangle((0,50), 30, 40, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box3 = plt.Rectangle((0,130), 30, 20, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box4 = plt.Rectangle((70,130), 40, 20, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box5 = plt.Rectangle((150,130), 40, 20, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box6 = plt.Rectangle((230,130), 30, 20, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box7 = plt.Rectangle((70,0), 40, 10, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box8 = plt.Rectangle((150,0), 40, 10, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box9 = plt.Rectangle((230,0), 30, 10, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box10 = plt.Rectangle((230,50), 30, 40, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box11 = plt.Rectangle((70,50), 10, 40, facecolor = "None", edgecolor = 'black', hatch = "/" )
    box12 = plt.Rectangle((180,50), 10, 40, facecolor = "None", edgecolor = 'black', hatch = "/" )
    round_about = plt.Circle((130,70), 15,facecolor = "None", edgecolor = 'black', hatch = "/" )

    ax.add_artist(map)
    ax.add_artist(box1)
    ax.add_artist(box2)
    ax.add_artist(box3)
    ax.add_artist(box4)
    ax.add_artist(box5)
    ax.add_artist(box6)
    ax.add_artist(box7)
    ax.add_artist(box8)
    ax.add_artist(box9)
    ax.add_artist(box10)
    ax.add_artist(box11)
    ax.add_artist(box12)
    ax.add_artist(round_about)

        