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
import copy

def mcl_update(particle_list, msg):
    # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
    
    # FIRST: Update locations and weights of particles
    for particle in particle_list:
        particle.sample_motion(msg)  # Update location
        if msg[0] == 'L': # Only update weights on laser scan
            particle.measurement_likelihood(msg)  # Update weight

    # SECOND: Re-sample particles in proportion to particle weight
    if msg[0] == 'L': # Only re-sample particles after laser-scan update
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
        #TODO:
        # Ensure particles start in valid locations - re-sample if not
        self.prev_log_pose = None
        self.pose = (-600,750,0)
        pass

    def measurement_likelihood(self, msg):
        """Returns a new particle weight 
        High if actual measurement matches model"""
        #TODO: Implemente real weighting
        self.weight = 1.0
        return self.weight
    
    def sample_motion(self, msg):
        msg_pose = msg[2], msg[3], msg[4]
        if self.prev_log_pose is None: # First iteration
            self.prev_log_pose = self.pose

        """Returns a new (sampled) x,y position for next timestep"""
        # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
        #TODO:
        new_particle_location = (0,0,0)
        self.pose = new_particle_location
        return self.pose
    

def load_log(filepath):
    """Log comes in two types:
    Type O:  
    x y theta - coordinates of the robot in standard odometry frame
    ts - timestamp of odometry reading (0 at start of run)

    Type L:
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
    return reordered_data


def sample_pose_uniform(xmin, xmax, ymin, ymax):
    """Returns a single pose x,y,theta sampled uniformly"""
    #TODO: return pose
    pass


def draw_map_state(gmap, particle_list=None, title="Wean Hall Map"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(gmap.values, cmap=plt.cm.gray, interpolation='nearest')
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
    ax.set_xlim(250,750)

    #TODO: Draw particles

    plt.show()