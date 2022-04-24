
from PIL import Image
import autograd.numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.misc import face

class Obstacle:
    def __init__(self, id, obstacle_type, parameters, sampling_time, path = None, COLOR = "#c0392b", ZOOM = 0.052, collided = False):
        
        # Obstacle Details
        self.id = id
        self.type = obstacle_type # Static or Dynamic?
        self.collided = collided
        # Description of the Obstacle 
        x_pos = parameters[0] 
        y_pos = parameters[1]
        radius = parameters[2]
        velocity = parameters[3] * 5/18
        angle = parameters[4] * np.pi/180
        acceleration = parameters[5]

        self.parameters = [x_pos, y_pos, radius, velocity, angle, acceleration]
        self.sampling_time = sampling_time

        # Load Image from Path
        if path!=None:
            self.image = Image.open(path)
        else:
            self.image = None
        if self.type == "Static":
            self.COLOR = COLOR      # Default Color for Static = Red
        else:
            self.COLOR = "#8e44ad"  # Default Color For Dynamic = Purple
        self.ZOOM = ZOOM

        # Store History
        self.history = []
        self.history.append([x_pos, y_pos, radius, velocity, angle, acceleration])

        
    def Model(self):
        self.parameters[3] += self.parameters[5] * self.sampling_time
        self.parameters[0] += self.parameters[3]*np.cos(self.parameters[4]) * self.sampling_time
        self.parameters[1] += self.parameters[3]*np.sin(self.parameters[4]) * self.sampling_time
        
        x_pos = self.parameters[0] 
        y_pos = self.parameters[1]
        radius = self.parameters[2]
        velocity = self.parameters[3]
        angle = self.parameters[4]
        acceleration = self.parameters[5]

        self.history.append([x_pos, y_pos, radius, velocity, angle, acceleration])
    
    def plot(self, ax):
        if self.image == None:
            obstacle = plt.Circle((self.parameters[0], self.parameters[1]),self.parameters[2], facecolor = self.COLOR, edgecolor='black')
            ax.add_artist(obstacle)
            obstacle_photo = None
        else:
            obstacle = plt.Circle((self.parameters[0], self.parameters[1]),self.parameters[2], facecolor='None', edgecolor='black')
            ax.add_artist(obstacle)
            img = self.image.rotate(self.parameters[4]*180/np.pi,expand=1)
            obstacle_photo = AnnotationBbox(OffsetImage(img, zoom= self.ZOOM), (self.parameters[0], self.parameters[1]), frameon=False)
            ax.add_artist(obstacle_photo)
        
        return obstacle, obstacle_photo