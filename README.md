# MPC for Multiple Autonomous Vehicles (GVO, ORCA, AVO)


https://user-images.githubusercontent.com/51902286/166077232-4cb77f23-5569-4a4d-8dc6-5f8ab0432a41.mp4

## INTRODUCTION:
------------------------
This repository demonstrates the implementation of a Model Predictive Controller for the path planning of Multiple Autonomous Vehicles and incorporates the GVO, ORCA and AVO methods for dynamic obstacle avoidance.\
The implemention is done in Python Environment.\

The demonstration for the project and various experiments can be found here:\
https://youtube.com/playlist?list=PLXBGAyISD6AmZxxhjcWZdMaxmb_QGNCOQ

## REQUIREMENTS:
------------------------
The following Libraries are required:
- PIL
- autograd
- matplotlib
- scipy
 
## CONTENTS:
------------------------
The repository consists of the following files:
- src:
    - main.py: This is the main file that is used for simulations. Run it using the command "python main.py"
    - config.py: Here the vehicles and obstacles are defined.
    - Vehicle.py: This contains the Vehicle Class which helps us define a Vehicle. The class also contains the kinematic model and the MPC controller for the vehicle.
    - Obstacle.py: This contains the Obstacle Class which defines the obstacles and its parameters.
    - plot.py: This helps generate the plots for simulation and also plots the graph of various parameters.

- Graphs:
    - Here the graphs for the experiments are stored. Please ensure there is a folder present to store the results.
    - Folder name should be: "Experiment_"Number"".

- bject_Photos:
    - Here the images of the vehicles and obstacles are stored. Feel free to add more photos.

Feel free to update the config.py file to create your desired simulations and suggest changes to make in order to make the code more robust.


