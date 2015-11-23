# Map exists which all robot particles operate in
# Particles each have a motion model and a measurement model

# Need to sample:
#   Motion model for particle (given location of particle, map)
#           Motion model (in this case) comes from log + noise. 
#   Measurement model for particle (given location, map)
#           True measurements come from log

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def mcl_update(particle_list, msg):
    # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
    # 'type' is 1.0 for laser scan, 0.0 for odometry-only
    
    # FIRST: Update locations and weights of particles
    for particle in particle_list:
        particle.sample_motion(msg)  # Update location
        if msg[0] > 0.1: # Only update weights on laser scan
            particle.measurement_likelihood(msg)  # Update weight

    # SECOND: Re-sample particles in proportion to particle weight
    if msg[0] > 0.1: # Only re-sample particles after laser-scan update
        new_particle_list = [] 
        particle_list_weights = [p.weight for p in particle_list]
        new_particle_list = sample_list_by_weight(particle_list, particle_list_weights)
        return new_particle_list
    else: # No laser scan - just return same list of particles with updated weights
        return particle_list


def sample_list_by_weight(list_to_sample, list_element_weights, randomize_order=True):
    new_sampled_list = []
    array_element_weights = np.array(list_element_weights)
    normed_weights = array_element_weights / array_element_weights.sum()
    list_idx_choices = np.random.multinomial(len(normed_weights), normed_weights)
    # list_idx_choices of form [0, 0, 2, 0, 1, 0, 4] for length 7 list_to_sample
    for idx, count in enumerate(list_idx_choices):
        while count > 0:
            if count == 1:
                new_sampled_list.append(list_to_sample[idx])
            else: # Need to add copies to new list, not just identical references! 
                new_sampled_list.append(copy.copy(list_to_sample[idx]))
            count -= 1
    if randomize_order:   # Not required, but nice for random order
        np.random.shuffle(new_sampled_list)
    return new_sampled_list


class occupancy_map():
    def __init__(self, map_filename):
        self.map_filename = map_filename
        self.load_map(self.map_filename)
        
    def load_map(self, map_filename):
        gmap = pd.read_csv(map_filename, sep=' ', header=None,
                   skiprows=list(range(7)), )
        gmap.drop(800, axis=1, inplace=True) # Drop garbage values
        self.values = gmap.values


class laser_sensor():
    """Defines laser sensor with specific meaasurement model"""
    def __init__(self):
        #TODO:
        pass


class robot_particle():
    
    def __init__(self, global_map, laser_sensor):
        self.weight = 1.0 # Default initial weight
        self.laser_sensor = laser_sensor
        self.global_map = global_map
        self.init_pose()

    def init_pose(self):
        """particle poses stored in GLOBAL map frame"""
        # Ensure particles start in valid locations - re-sample if not
        self.prev_log_pose = None
        theta_initial = np.random.uniform(-2*np.pi,2*np.pi)
        self.relative_pose = None

        valid_pose = False
        while not valid_pose:
            x_initial, y_initial = np.random.uniform(0,8000,2)
            self.pose = np.array([x_initial,y_initial,theta_initial])
            valid_pose = self.position_valid()

    def measurement_likelihood(self, msg):
        """Returns a new particle weight 
        High if actual measurement matches model"""
        #TODO: Implemente real weighting
        self.weight = 1.0
        return self.weight
    
    def sample_motion(self, msg):
        """Returns a new (sampled) x,y position for next timestep"""
        # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
        msg_pose = msg[2:5] # three elements: x, y, theta
        if self.prev_log_pose is None: # First iteration
            self.prev_log_pose = msg_pose

        #TODO: Add stochasticity here
        self.pose = new_pose_from_log_delta(self.prev_log_pose,
                                            msg_pose,
                                            self.pose)

        self.prev_log_pose = msg_pose # Save previous log pose for delta
        return self.pose

    def position_valid(self):
        x_pos = self.pose[0]
        y_pos = self.pose[1]
        nearest_xindex = x_pos//10
        nearest_yindex = y_pos //10
        # High map values = clear space ( > ~0.8), low values = obstacle
        if self.global_map.values[nearest_xindex, nearest_yindex] > 0.8:
            return True
        else:
            return False


def raycast_bresenham(x,y,theta, global_map,
                      threshold_val = 0.5, max_dist = 1000):
     """Brensenham line algorithm
     Input: x,y in cm, theta in radians, 
            global_map with 800x800 10-cm occupancy grid

     Ref: https://mail.scipy.org/pipermail/scipy-user/2009-September/022601.html"""
     
     # Cast rays within 800x800 map (10cm * 800 X 10cm * 800)

     #TODO: Implement with x,y in range 0~800 - will be much faster.
     x0 = x
     y0 = y
     x2 = x + int(max_dist * np.cos(theta))
     y2 = y + int(max_dist * np.sin(theta))
     # Short-circuit if inside wall
     if global_map.values[x//10,y//10] < threshold_val :
        return x, y, 0
     steep = 0
     #coords = []
     dx = abs(x2 - x)
     if (x2 - x) > 0: sx = 1
     else: sx = -1
     dy = abs(y2 - y)
     if (y2 - y) > 0: sy = 1
     else: sy = -1
     if dy > dx:
         steep = 1
         x,y = y,x
         dx,dy = dy,dx
         sx,sy = sy,sx
     d = (2 * dy) - dx
     try:
         for i in range(0,dx):
             if steep: # X and Y have been swapped  #coords.append((y,x))
                if global_map.values[y//10, x//10] < threshold_val:
                    dist = np.sqrt((y - x0)**2 + (x - y0)**2)
                    return y, x, min(dist, max_dist)
             else: #coords.append((x,y))
                if global_map.values[x//10, y//10] < threshold_val:
                    dist = np.sqrt((x - x0)**2 + (y - y0)**2)
                    return x, y, min(dist, max_dist)
             while d >= 0:
                 y = y + sy
                 d = d - (2 * dx)
             x = x + sx
             d = d + (2 * dy)
         if steep:
             return y, x, max_dist
         else:
             return x, y, max_dist
     except IndexError: # Out of range
        dist = np.sqrt((y - x0)**2 + (x - y0)**2)
        return y, x, min(dist, max_dist)

def new_pose_from_log_delta(old_log_pose, new_log_pose, current_pose):
    """Transforms movement from message frame to particle frame"""
    log_delta_x = new_log_pose[0] - old_log_pose[0]
    log_delta_theta = new_log_pose[2] - old_log_pose[2]
    # Fwd motion in log frame == Fwd motion in particle frame
    cos_theta = np.cos(new_log_pose[2])
    if cos_theta > 0.1:
        fwd_motion = log_delta_x / cos_theta 
    else:  # Avoid numerical instability with divide by ~0
        log_delta_y = new_log_pose[1] - old_log_pose[1]
        fwd_motion = log_delta_y / np.sin(new_log_pose[2])

    new_current_theta = current_pose[2] + log_delta_theta
    new_current_x = current_pose[0] + fwd_motion * np.cos(new_current_theta)
    new_current_y = current_pose[1] + fwd_motion * np.sin(new_current_theta)
    return np.array([new_current_x, new_current_y, new_current_theta])


def load_log(filepath):
    """Log comes in two types:
    Type O (remapped to 0.0):  
    x y theta - coordinates of the robot in standard odometry frame
    ts - timestamp of odometry reading (0 at start of run)

    Type L  (remapped to 1.0):
    x y theta - coodinates of the robot in standard odometry frame when
    laser reading was taken (interpolated)
    xl yl thetal - coordinates of the *laser* in standard odometry frame
    when the laser reading was taken (interpolated)
    1 .. 180 - 180 range readings of laser in cm.  The 180 readings span
    180 degrees *STARTING FROM THE RIGHT AND GOING LEFT*  Just like angles,
    the laser readings are in counterclockwise order.
    ts - timestamp of laser reading
    """
 
    raw_df = pd.read_csv(filepath, sep=' ',header=None)
    # Extract and label odometry data
    odometry = raw_df[raw_df[0] == 'O'][list(range(5))]
    odometry.columns = ["type", "x", "y", "theta", "ts"]
    odometry.set_index('ts', inplace=True)
    # Extract and label laser scan data
    scans = raw_df[raw_df[0] == 'L']
    scans.columns = ['type', 'x', 'y', 'theta', 'xl', 'yl', 'thetal'] +\
                    [n+1 for n in range(180)] + ['ts']
    scans.set_index('ts', inplace=True)
    # Join and sort logs
    full_log = pd.concat([scans, odometry])
    full_log.sort_index(inplace=True)
    reordered_data = full_log.reset_index()[['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal'] + list(range(1,181))]
    # Remap laser -> 1.0,  odometry -> 0.0 to align datatype to float
    reordered_data['type'] = reordered_data['type'].map({'L':1, 'O':0})
    return reordered_data


def draw_map_state(gmap, particle_list=None, ax=None, title="Wean Hall Map",
                   rotate=True):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if rotate:
        values = gmap.values.T
        ax.set_ylim(2500,7500)
        ax.set_xlim(0,8000)
    else:
        values = gmap.values
        ax.set_ylim(0,8000)
        ax.set_xlim(2500,7500)

    ax.imshow(values, cmap=plt.cm.gray, interpolation='nearest',
              origin='lower', extent=(0,8000,0,8000), aspect='equal')
    ax.set_title(title)
    # Move left and bottom spines outward by 10 points
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if particle_list is not None:
        for particle in particle_list:
            plot_particle(particle, ax) # Scale to 1/10th scale map
    return ax



def plot_particle(particle, ax=None, pass_pose=False):
    if ax is None:
        ax = plt.gca()
    if pass_pose:
        x, y, theta = particle
    else:
        x, y, theta = particle.pose

    # 25cm is distance to actual sensor
    xt = x + 25*np.cos(theta)
    yt = y + 25*np.sin(theta)
    circle = patches.CirclePolygon((x,y),facecolor='none', edgecolor='b',
                                   radius=25, resolution=20)
    ax.add_artist(circle)  
    ax.plot([x, xt], [y, yt], color='b')
    return ax