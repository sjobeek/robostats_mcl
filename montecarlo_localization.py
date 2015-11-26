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
from scipy.spatial import distance


def mcl_update(particle_list, msg, min_particles = 300):
    # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
    # 'type' is 1.0 for laser scan, 0.0 for odometry-only
    
    # FIRST: Update locations and weights of particles
    for particle in particle_list:
        # Update location
        particle.sample_motion(msg)  
        if msg[0] > 0.1: # Only update weights on laser scan
            particle.update_measurement_likelihood(msg)  # Update weight

    # SECOND: Re-sample particles in proportion to particle weight
    if msg[0] > 0.1: # Only re-sample particles after laser-scan update
        particle_list_weights = [p.weight if p.position_valid() else 0.0
                                 for p in particle_list]
        # Renormalize weights if getting very small
        if sum(particle_list_weights) < 1:
            for p in particle_list:
                p.weight *= 100
        new_particle_list = sample_list_by_weight(particle_list, particle_list_weights)
        return new_particle_list
    else: # No laser scan - just filter particles for validity
        new_particle_list = [] 
        for p in particle_list:
            if p.position_valid():
                new_particle_list.append(p)
        while len(new_particle_list) < min_particles:
            # Add back in new duplicate particles
            duplicate_index = np.random.choice(range(len(new_particle_list)))
            new_particle_list.append(copy.copy(new_particle_list[duplicate_index]))
        return new_particle_list


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
    def __init__(self, map_filename, range_filename='./data/range_array_120bin.npy'):
        self.map_filename = map_filename
        self.range_filename = range_filename
        self.load_map()
        
    def load_map(self):
        gmap = pd.read_csv(self.map_filename, sep=' ', header=None,
                   skiprows=list(range(7)), )
        gmap.drop(800, axis=1, inplace=True) # Drop garbage values
        self.values = gmap.values
        self.range_array = np.load(self.range_filename)

    def ranges_180(self, x_cm, y_cm, theta_rads, n_buckets=120):
        x_loc = x_cm//10
        y_loc = y_cm//10
        return np.array([self.range_array[x_loc,y_loc,bucket] for bucket in 
                         theta_to_bucket_ids(theta_rads, n_buckets=n_buckets)])


def rads_to_bucket_id(rads, n_buckets=120):
    return int(((rads / (2*np.pi)) * n_buckets) % n_buckets)

def theta_to_bucket_ids(theta_rads, n_buckets=120):
    """Returns the bucket ids corresponding to 
    -90 deg. to +90 deg around robot heading.
    All parametrs use radians."""
    start_theta = theta_rads - np.pi/2
    start_bucket_id = rads_to_bucket_id(start_theta)
    return [(start_bucket_id + i) % n_buckets 
            for i in range(n_buckets//2)]


class laser_sensor():
    """Defines laser sensor with specific meaasurement model"""
    def __init__(self):
        #TODO:
        pass


class robot_particle():
    
    def __init__(self, global_map, laser_sensor,
                 sigma_fwd_pct=.2, sigma_theta_pct=.05):
        """sigma_[x,y,theta]_pct  represents stddev of movement over 1 unit as percentage
            e.g., if true movement in X = 20cm, and sigma_x_pct=.1, then stddev = 2cm"""
        self.weight = 1.0 # Default initial weight
        self.laser_sensor = laser_sensor
        self.global_map = global_map
        self.sigma_fwd_pct = sigma_fwd_pct
        self.sigma_theta_pct = sigma_theta_pct
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

    def update_measurement_likelihood(self, laser_msg):
        """Returns a new particle weight 
        High if actual measurement matches model"""
        #TODO: Implemente real weighting
        msg_range_indicies = list(range(8,188))
        actual_measurement = laser_msg[msg_range_indicies]
        subsampled_measurements = actual_measurement[::3] #each third
        #Expected measurements in 60 3-degree buckets, covering -90 to 90 degrees
        expected_measurement = self.global_map.ranges_180(*self.pose)
        # Cosine distance = close to 0 if identical. Close to 1 if far apart.
        # Doesn't work well.
        # TODO: Implement actual laser measurement model here 
        dist = distance.cosine(subsampled_measurements, expected_measurement)
        self.weight = self.weight * (1 - dist)
        return dist
    
    def sample_motion(self, msg):
        """Returns a new (sampled) x,y position for next timestep"""
        # msg: ['type','ts', 'x', 'y', 'theta', 'xl','yl', 'thetal', r1~r180]
        msg_pose = msg[2:5] # three elements: x, y, theta
        if self.prev_log_pose is None: # First iteration
            self.prev_log_pose = msg_pose

        # Includes stochastic error to change in pose, scaled to magnitude of change
        self.new_pose_from_log_delta(msg_pose)

        self.prev_log_pose = msg_pose # Save previous log pose for delta
        return self.pose

    def new_pose_from_log_delta(self, new_log_pose):
        """Transforms movement from message frame to particle frame,
        adding error from self.sigma_fwd_pct and self.sigma_theta_pct"""
        log_delta_x = new_log_pose[0] - self.prev_log_pose[0]
        log_delta_y = new_log_pose[1] - self.prev_log_pose[1]
        log_delta_theta = new_log_pose[2] - self.prev_log_pose[2]
        # Fwd motion in log frame == Fwd motion in particle framea
        #TODO: Change to x^2 + y^2 ??
        cos_theta = np.cos(new_log_pose[2])
        if cos_theta > 0.1:
            fwd_motion = log_delta_x / cos_theta 
        else:  # Avoid numerical instability with divide by ~0
            fwd_motion = log_delta_y / np.sin(new_log_pose[2])

        # Calculate and add stochastic theta and forward error
        new_theta_error = log_delta_theta * self.sigma_theta_pct * np.random.normal()
        new_current_theta = self.pose[2] + log_delta_theta + new_theta_error
        # Wrap radians to enforce range 0 to 2pi
        new_current_theta = new_current_theta % (2*np.pi)
        
        fwd_motion_error = fwd_motion * self.sigma_fwd_pct * np.random.normal()
        fwd_motion += fwd_motion_error
        new_current_x = self.pose[0] + fwd_motion * np.cos(new_current_theta)
        new_current_y = self.pose[1] + fwd_motion * np.sin(new_current_theta)
        self.pose = np.array([new_current_x, new_current_y, new_current_theta])
        return self.pose

    def position_valid(self):
        nearest_xindex = self.pose[0]//10
        nearest_yindex = self.pose[1] //10
        try:
            # High map values = clear space ( > ~0.8), low values = obstacle
            if self.global_map.values[nearest_xindex, nearest_yindex] > 0.8:
                return True
            else:
                return False
        except IndexError:
            return False


def raycast_bresenham(x_cm, y_cm, theta, global_map,
                      freespace_min_val=0.5, max_dist_cm=8183):
     """Brensenham line algorithm
     Input: x,y in cm, theta in radians, 
            global_map with 800x800 10-cm occupancy grid

     Ref: https://mail.scipy.org/pipermail/scipy-user/2009-September/022601.html"""
     
     # Cast rays within 800x800 map (10cm * 800 X 10cm * 800)

     x = x_cm//10
     y = y_cm//10
     max_dist = max_dist_cm//10

     #TODO: Implement with x,y in range 0~800 - will be much faster.

     x0 = x
     y0 = y
     x2 = x + int(max_dist * np.cos(theta))
     y2 = y + int(max_dist * np.sin(theta))
     # Short-circuit if inside wall
     if global_map.values[x,y] < freespace_min_val :
        return x*10, y*10, 0
     steep = 0
     #coords = []
     dx = abs(x2 - x)
     if (x2 - x) > 0: sx = 1
     else: sx = -1
     dy = abs(y2 - y)
     if (y2 - y) > 0: sy = 1
     else: sy = -1
     if dy > dx: # Angle is steep - swap X and Y
         steep = 1
         x,y = y,x
         dx,dy = dy,dx
         sx,sy = sy,sx
     d = (2 * dy) - dx
     try:
         for i in range(0,dx):
             if steep: # X and Y have been swapped  #coords.append((y,x))
                if global_map.values[y, x] < freespace_min_val:
                    dist = np.sqrt((y - x0)**2 + (x - y0)**2)
                    return y*10, x*10, min(dist, max_dist)*10
             else: #coords.append((x,y))
                if global_map.values[x, y] < freespace_min_val:
                    dist = np.sqrt((x - x0)**2 + (y - y0)**2)
                    return x*10, y*10, min(dist, max_dist)*10
             while d >= 0:
                 y = y + sy
                 d = d - (2 * dx)
             x = x + sx
             d = d + (2 * dy)
         if steep:
             dist = np.sqrt((y - x0)**2 + (x - y0)**2)
             return y*10, x*10, min(dist, max_dist)*10
         else:
             dist = np.sqrt((x - x0)**2 + (y - y0)**2)
             return x*10, y*10, min(dist, max_dist)*10
     except IndexError: # Out of range
        dist = np.sqrt((y - x0)**2 + (x - y0)**2)
        return y*10, x*10, min(dist, max_dist)*10




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



def plot_particle(particle, ax=None, pass_pose=False, color='b'):
    if ax is None:
        ax = plt.gca()
    if pass_pose:
        x, y, theta = particle
    else:
        x, y, theta = particle.pose

    # 25cm is distance to actual sensor
    xt = x + 25*np.cos(theta)
    yt = y + 25*np.sin(theta)
    circle = patches.CirclePolygon((x,y),facecolor='none', edgecolor=color,
                                   radius=25, resolution=20)
    ax.add_artist(circle)  
    ax.plot([x, xt], [y, yt], color=color)
    return ax