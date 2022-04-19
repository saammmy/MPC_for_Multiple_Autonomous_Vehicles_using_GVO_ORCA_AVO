from matplotlib import pyplot as plt
import numpy as np

def plot_simulation(Vehicles, Obstacles, ax):
    VEHICLE_PLOT, ARROW_PLOT, VEHICLE_PHOTO_PLOT, STATIC_REGION_PLOT, DYNAMIC_REGION_PLOT = ([] for i in range(5))
    OBSTACLE_FIG, OBSTACLE_PHOTO = ([] for i in range(2))

    for obstacle in Vehicles[0].Obstacles:
        if obstacle.type != 'Static':
            obstacle_fig, obstacle_photo = obstacle.plot(ax)
            OBSTACLE_FIG.append(obstacle_fig)
            OBSTACLE_PHOTO.append(obstacle_photo)
    
    for vehicle in Vehicles:
        vehicle_plot, arrow_plot, vehicle_photo_plot, static_region_plot, dynamic_region_plot = vehicle.plot(ax)
        VEHICLE_PLOT.append(vehicle_plot)
        ARROW_PLOT.append(arrow_plot)
        VEHICLE_PHOTO_PLOT.append(vehicle_photo_plot)
        STATIC_REGION_PLOT.append(static_region_plot)
        DYNAMIC_REGION_PLOT.append(dynamic_region_plot)

    plt.draw()
    plt.pause(Vehicles[0].sampling_time*0.1)

    for i in range(len(VEHICLE_PLOT)):
        VEHICLE_PLOT[i].remove()
        ARROW_PLOT[i].remove()
        STATIC_REGION_PLOT[i].remove()
        DYNAMIC_REGION_PLOT[i].remove()
        if VEHICLE_PHOTO_PLOT[i] != None:
            VEHICLE_PHOTO_PLOT[i].remove()

    for i in range(len(OBSTACLE_FIG)):
        OBSTACLE_FIG[i].remove()
        if OBSTACLE_PHOTO[i] != None:
            OBSTACLE_PHOTO[i].remove()
    

def save_graph(Vehicles, Obstacles):
    
    for i in range(len(Vehicles)):
        vehicle = Vehicles[i]  
        control_input_history = np.array(vehicle.control_input_history)
        state_history = np.array(vehicle.state_history)
        time_history = np.array(vehicle.time_history)
        residual_state_history = np.array(vehicle.state_history)-np.array(vehicle.goal)
            
        plt.figure()
        plt.title('Vehicle Velocity')
        plt.plot(time_history,control_input_history[:,0])
        plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Vehicle_{}_Velocity.png'.format(i+1,i+1), bbox_inches='tight')
        
        plt.figure()
        plt.title('Steering Rate (Degree/s)')
        plt.plot(time_history,control_input_history[:,1]*180/np.pi)
        plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Vehicle_{}_Steering_Rate.png'.format(i+1,i+1), bbox_inches='tight')

        plt.figure()
        plt.title('XY Position Residual(m)')
        plt.plot(time_history,(residual_state_history[:,0]**2+residual_state_history[:,1]**2)**0.5)
        plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Vehicle_{}_XY_Residual.png'.format(i+1,i+1), bbox_inches='tight')

        plt.figure()
        plt.title('Theta Residual(Degree)')
        plt.plot(time_history,residual_state_history[:,2]*180/np.pi)
        plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Vehicle_{}_Theta_Residual.png'.format(i+1,i+1), bbox_inches='tight')

        plt.figure()
        plt.title('Delta Residual (Degree)')
        plt.plot(time_history,residual_state_history[:,3]*180/np.pi)
        plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Vehicle_{}_Delta_Residual.png'.format(i+1,i+1), bbox_inches='tight')

        for j in range(len(Obstacles)):
            obstacle=Obstacles[j]
            if obstacle.type == "Static":
                residual_history = np.array(vehicle.state_history)[:,0:2]-np.array(obstacle.history)[0,0:2]
                plt.figure()
                plt.title('Obstacle(Static) {} and Vehicle {} Residual(m)'.format(j+1,i+1))
                plt.plot(time_history,(residual_history[:,0]**2+residual_history[:,1]**2)**0.5)
                plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Obstacle_{}(Static)_&_Vehicle_{}_Residual.png'.format(i+1,j+1,i+1), bbox_inches='tight')
            else:
                residual_history = np.array(vehicle.state_history)[:,0:2]-np.array(obstacle.history)[0:len(vehicle.state_history),0:2]
                plt.figure()
                plt.title('Obstacle(Dynamic) {} and Vehicle {} Residual(m)'.format(j+1,i+1))
                plt.plot(time_history,(residual_history[:,0]**2+residual_history[:,1]**2)**0.5)
                plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Obstacle_{}(Static)_&_Vehicle_{}_Residual.png'.format(i+1,j+1,i+1), bbox_inches='tight')
                





        # plt.figure()
        # plt.title('X Position')
        # plt.plot(time_history,state_history[:,0])
        # plt.savefig('../Graphs/Experiment_1/Vehicle_{}/X_Position.png'.format(i+1), bbox_inches='tight')

        # plt.figure()
        # plt.title('Y Position')
        # plt.plot(time_history,state_history[:,1])
        # plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Y_Position.png'.format(i+1), bbox_inches='tight')

        # plt.figure()
        # plt.title('Theta (Degree)')
        # plt.plot(time_history,state_history[:,2]*180/np.pi)
        # plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Theta.png'.format(i+1), bbox_inches='tight')

        # plt.figure()
        # plt.title('Delta (Degree)')
        # plt.plot(time_history,state_history[:,3]*180/np.pi)
        # plt.savefig('../Graphs/Experiment_1/Vehicle_{}/Delta.png'.format(i+1), bbox_inches='tight')