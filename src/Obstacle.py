
from PIL import Image
import autograd.numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class Obstacle:
    def __init__(self, parameters, obstacle_type='Static', path = None, COLOR = "#c0392b", ZOOM = 0.07):
        self.type = obstacle_type # Static or Dynamic?
        
        # Description of the Obstacle 
        x_pos = parameters[0] 
        y_pos = parameters[1]
        radius = parameters[2]
        velocity = parameters[3]
        angle = parameters[4]*np.pi/180

        self.parameters = [x_pos, y_pos, radius, velocity, angle]
        self.sampling_time = 0.2

        # Load Image from Path
        if path!=None:
            self.image = Image.open(path)
        else:
            self.image = None
        if self.type == "Static":
            self.COLOR = COLOR
        else:
            self.COLOR = "#8e44ad" 
        self.ZOOM = ZOOM

        # Store History
        self.history = []
        self.history.append([x_pos, y_pos, radius, velocity, angle])

        
    def Model(self):
        self.parameters[0] += self.parameters[3]*np.cos(self.parameters[4]) * self.sampling_time
        self.parameters[1] += self.parameters[3]*np.sin(self.parameters[4]) * self.sampling_time
        
        x_pos = self.parameters[0] 
        y_pos = self.parameters[1]
        radius = self.parameters[2]
        velocity = self.parameters[3]
        angle = self.parameters[4]

        self.history.append([x_pos, y_pos, radius, velocity, angle])
    
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