from f110_agents.agents_numpy import BaseAgent
import numpy as np
from scipy.stats import truncnorm
"""
a pure pursuit agent to drive back to the raceline 
"""
class Raceline():
    def __init__(self):
        self.xs = []
        self.ys = []
        self.vxs = []

class Track():
    def __init__(self, file):
        self.track = self.load_track(file)
        self.centerline = Raceline()
        self.centerline.xs = self.track[:,0]
        self.centerline.ys = self.track[:,1]

        self.centerline.vxs = self.track[:,2]

    def load_track(self,file):
        # open and read in the track, a csv with x_m, x_y, vx_mps, delta_rad 
        track = np.loadtxt(file, delimiter=',')
        return track


class StochasticContinousPPAgent(BaseAgent):
    def __init__(self, deterministic=True, raceline_file='raceline.csv', fixed_speed = None, resetting=True, lookahead_points=200, num_starting_points=2,std=0.3, max_delta=0.05, **kwargs):
        # initalize parent class
        super().__init__(**kwargs)
        #if not deterministic:
        #    raise NotImplementedError
        self.deterministic = deterministic
        self.raceline_file = raceline_file
        self.fixed_speed = fixed_speed
        self.track = Track(raceline_file)
        self.current_track_point = None
        self.subsample = 20
        self.resetting = resetting
        self.starting_points_progress = np.linspace(0, 1, num_starting_points)
        self.std = std
        self.lookahead_points = lookahead_points
        self.max_delta_steering = max_delta
        self.max_change = max_delta
        self.exp_decay = 0.1

    def distance_point_to_line(self,px, py, x1, y1, x2, y2):
        """
        Calculate the distance of point (px, py) to the line formed by points (x1, y1) and (x2, y2).
        Only for reseting agent, does not support batch
        """
        # assert not batch
        """
        print(px, py, x1, y1, x2, y2)
        assert type(px) == float 
        assert type(py) == float
        assert type(x1) == float
        assert type(y1) == float
        assert type(x2) == float
        assert type(y2) == float
        """

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        distance = np.abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)
        return distance
    def find_closest_track_point(self, x, y):
        """
        Find the closest point on the raceline to the given x, y coordinates.

        @param x: (batch_size, 1) array of x-coordinates.
        @param y: (batch_size, 1) array of y-coordinates.
        @return: (batch_size, 1) array of indices representing the closest point on the raceline for each x, y pair.
        """
        xs = np.array(self.track.centerline.xs)
        ys = np.array(self.track.centerline.ys)
        
        # Reshape x and y for broadcasting
        x_reshaped = x.reshape(1, -1)
        y_reshaped = y.reshape(1, -1)

        # Calculate distances using broadcasting
        distances = np.sqrt((xs[:, None] - x_reshaped) ** 2 + (ys[:, None] - y_reshaped) ** 2)
        # Find the index of the minimum distance for each point
        return np.argmin(distances, axis=0)
    def __str__(self):
        return "pure_pursuit"
    def reset(self):
        self.current_track_point = None

    def find_next_track_point(self, current_track_point, x, y, lookahead_distance, max_lookahead_indices=200):
        """
        Find the next target point on the raceline based on the agent's current position and a lookahead distance.

        @param current_track_point: (batch_size, 1) array of indices representing the current closest point on the raceline.
        @param x: (batch_size, 1) array of x-coordinates of the agent.
        @param y: (batch_size, 1) array of y-coordinates of the agent.
        @param lookahead_distance: Floating-point number indicating the minimum distance ahead to look for the next point.
        @param max_lookahead_indices: Integer indicating the maximum number of indices to look ahead on the raceline. Defaults to 20.
        @return: (batch_size, 1) array of indices representing the next target point on the raceline for each current point.

        The function searches up to `max_lookahead_indices` ahead of the current track point for the first point that is farther than `lookahead_distance` from the agent's current position, considering wraparound at the end of the raceline.
        """
        xs = np.array(self.track.centerline.xs)
        ys = np.array(self.track.centerline.ys)
        track_length = len(xs)

        # Generate all indices to look ahead within the max_lookahead_indices range
        #print("current track point", current_track_point)
        indices_ahead = (current_track_point[:, None] + np.arange(max_lookahead_indices)) % track_length
        #print("indiecs ahead shape", indices_ahead.shape)
        # Calculate distances for all these indices
        distances = np.sqrt((xs[indices_ahead] - x[:, None]) ** 2 + (ys[indices_ahead] - y[:, None]) ** 2)
        #print(distances)
        # Find the first index where distance is greater than lookahead_distance
        #while True:
        # dirty fix, set the last distance to high number so we always find one
        distances[:,-1] = 2.0
        #print(distances)
        #print(lookahead_distance)
        first_valid_indices = np.argmax(distances > lookahead_distance, axis=1)
        #print("valids", first_valid_indices)
        #    if np.all(first_valid_indices):
        #        break
        #    lookahead_distance *= 2
        #print(first_valid_indices)
        # If no valid index is found within the range, keep the current track point
        # TODO! maybe make this a loop until we find a valid index
        # no_valid_index = np.all(distances <= lookahead_distance, axis=1)
        # first_valid_indices[no_valid_index] = 0

        # Adjust the shape of first_valid_indices to be compatible for indexing
        first_valid_indices = first_valid_indices.reshape(-1, 1)

        #print(first_valid_indices.shape)
        #print(first_valid_indices)
        return ((current_track_point + first_valid_indices) % track_length)[0]
    
    def calculate_steering_angle(self, x, y, theta, next_track_point, lookahead_distance):
        """
        Calculate the steering angle using the pure pursuit algorithm based on the C++ implementation.

        @param x: (batch_size, 1) array of x-coordinates of the agent.
        @param y: (batch_size, 1) array of y-coordinates of the agent.
        @param theta: (batch_size, 1) array of orientations of the agent. Values are between -π and π.
        @param next_track_point: (batch_size, 1) array of indices representing the next target point on the raceline for each current point.
        @param lookahead_distance: Floating-point number indicating the lookahead distance.
        
        The function computes the required steering angle for the agent to steer towards the lookahead point on the raceline.
        """
        assert (theta >= -np.pi).all() and (theta <= np.pi).all(), f"Theta: {theta}"

        # Extract the x and y coordinates of the next track points
        lookahead_x = np.array(self.track.centerline.xs)[next_track_point].flatten()
        lookahead_y = np.array(self.track.centerline.ys)[next_track_point].flatten()

        # Transform waypoints into car coordinates
        # This involves rotating the coordinate system by -theta (vehicle's orientation)
        dx = lookahead_x - x.flatten()
        dy = lookahead_y - y.flatten()
        transformed_x = dx * np.cos(-theta.flatten()) - dy * np.sin(-theta.flatten())
        transformed_y = dx * np.sin(-theta.flatten()) + dy * np.cos(-theta.flatten())

        # Calculate alpha, the angle in the car's coordinate system
        alpha = np.arctan2(transformed_y, transformed_x)

        # Wheelbase of the vehicle, which is 0.3 meters in the C++ implementation
        wheelbase = 0.3

        # Calculate the steering angle using the pure pursuit formula
        steering_angle = np.arctan((2 * wheelbase * np.sin(alpha)) / lookahead_distance)

        return steering_angle

    def __call__(self, model_input_dict: dict, action=None, std=None, deaccelerate=False, **kwargs):
        # model input dict values have a batch dimension
        # extract the current position and orientation
        #print(model_input_dict)
        model_input_dict_ = model_input_dict.copy()
        x = model_input_dict_['poses_x']
        y = model_input_dict_['poses_y']
        theta_sin = model_input_dict_['theta_sin']
        theta_cos = model_input_dict_['theta_cos']
        theta = np.arctan2(theta_sin, theta_cos)
        
        # find the current closest track point
        if True:#self.current_track_point is None:
            self.current_track_point = self.find_closest_track_point(x,y)
            # print(self.current_track_point)
            # exit()
        else: 
            # this needs an insurance that the point is in front of the car
            self.current_track_point = self.find_next_track_point(self.current_track_point,
                                                                    x,
                                                                    y,
                                                                    0.0,
                                                                    max_lookahead_indices=self.lookahead_points)
            self.current_track_point = self.current_track_point[0]
            #print("current_track point", self.current_track_point)
            #exit()
            # exit()
        # find the next track point
        lookahead = 1.0
        next_track_point = self.find_next_track_point(self.current_track_point, x, y, lookahead, max_lookahead_indices=self.lookahead_points)
        # next_track_point += 200
        # print(self.current_track_point, next_track_point)
        assert next_track_point.shape == self.current_track_point.shape
        # calculate the steering angle
        calculated_steering_angle = self.calculate_steering_angle(x,y,theta,next_track_point, lookahead)
        #print("target, steering:", calculated_steering_angle)
        speed = np.zeros_like(calculated_steering_angle)
        reseting = False
        if self.fixed_speed is not None:
            speed = speed + self.fixed_speed
            if self.resetting:
                # speed depends on the distance to the raceline and the angle to the next selected point
                # if both are small we set the speed to 0.0
                # check calculated steering angle
                # print("calculated steering angle", calculated_steering_angle[0])
                # check current progress
                if deaccelerate:
                    print("deaccelerating")
                    speed =np.zeros_like(calculated_steering_angle)
                
                """
                line_first_point = self.track.centerline.xs[self.current_track_point], self.track.centerline.ys[self.current_track_point]
                next_point = (self.current_track_point + 1) % len(self.track.centerline.xs)
                line_second_point = self.track.centerline.xs[next_point], self.track.centerline.ys[next_point]
                current_point = x, y
                dist = self.distance_point_to_line(current_point[0][0], current_point[1][0], line_first_point[0][0], line_first_point[1][0], 
                                                    line_second_point[0][0], 
                                                    line_second_point[1][0])
                if dist < 0.2 and abs(calculated_steering_angle) < 0.2:
                    speed = np.array([0.0])
                    reseting = True
                """
        else:
            raise NotImplementedError
        
        #print (calculated_steering_angle)
        # now compute delta action
        #print(model_input_dict_['previous_action'].shape)
        #print(model_input_dict_['previous_action'])
        # print("model_input_dict_['previous_action'] inside", model_input_dict_['previous_action'])
        # print(model_input_dict_['previous_action'])
        current_angle = model_input_dict["previous_action_steer"]  #model_input_dict_['previous_action'][:,0]
        current_speed = model_input_dict["previous_action_speed"] #model_input_dict_['previous_action'][:,1]
        
        delta_angles, abs_angle = self.get_delta_angle(calculated_steering_angle, current_angle)
        delta_speeds, abs_speed = self.get_delta_speed(speed, current_speed)
        #print("target angle", abs_angle)
        #print("target speed", abs_speed)
        #print("curr speed", current_speed)
        #print("current_speed inside", delta_speeds + current_speed)
        #print(delta_angles)
        #print(delta_speeds)
        means = np.vstack((delta_angles, delta_speeds)).T / self.max_delta_steering
        means = np.clip(means, -1.0, 1.0)
        # actions = means
        #actions = np.zeros_like(means)
        ####### JUST copied do better #######
        a = (- 1.0 - means) / self.std
        b = (1.0 - means ) / self.std
        #print(a)
        #print(b)
        if not self.deterministic:
            targets = np.random.normal(means, self.std)
        else:
            targets = means

        targets = np.clip(targets, -1.0, 1.0)

        #r = truncnorm.rvs(a[0,0], b[0,0], size=1000)
        #print(a[0,0])
        #print(b[0,0])
        # plot r
        #import matplotlib.pyplot as plt
        dist = truncnorm(a, b, loc=means, scale=self.std)
        # sample and plot
        #dis2t = truncnorm(a[0,0], b[0,0], loc=means[0,0], scale=std_angle)
        #samples = dis2t.rvs(size=1000)
        #plt.hist(samples, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
        #plt.show()
        if action is not None:
            assert (action <= 1.0).all() and (action >= -1.0).all(), "Action should be between -1 and 1"
            log_probs = dist.logpdf(action).sum(axis=1)
        else:
            #print("targets",targets)
            log_probs = dist.logpdf(targets) #.sum(axis=1)
            #print("not added log_probs", log_probs)
            #print(log_probs)
            log_probs = np.sum(log_probs, axis=1)

        # log_probs = 1.0
        #print("mean", means)
        #print("target", targets)
        assert (log_probs != -np.inf).all()
        # also assert no log_prob is nan
        assert (log_probs != np.nan).all()
        ## TODO! here we can add stochasticity!
        return [next_track_point], targets, log_probs

### this is deprecated ###
eval_config = {
    "collision_penalty": -10.0,
    "progress_weight": 1.0,
    "raceline_delta_weight": 0.0,
    "velocity_weight": 0.0,
    "steering_change_weight": 0.0,
    "velocity_change_weight": 0.0,
    "pure_progress_weight": 0.0,
    "inital_velocity": 1.5,
    "normalize": False,
}


if __name__ == "__main__":
    from f110_sim_env.base_env import make_base_env
    agent = StochasticContinousPPAgent(deterministic=True, fixed_speed=0.5)
    rays = np.ones((1,54,)) * 0.1
    rays[:,:10] += 0.1
    #rays += 0.5
    #print(rays)
    model_input_dict = {'lidar_occupancy': rays}
    #print(model_input_dict)
    #for _ in range(40):
    #    target = agent(model_input_dict)
    #    print(target)

    eval_env = make_base_env(map= "Infsaal",
            fixed_speed=None,
            random_start =True,
            train_random_start = False,
            reward_config = eval_config,
            eval=False,
            use_org_reward=True,
            min_vel=0.0,
            max_vel=0.0,
            max_delta_steering=0.05,
            max_acceleration=0.05,
        )

    action = np.array([[0.0,0.0],[0.0,0.0],[0.0,0.0]])
    i = 0
    while i < 6:
        eval_env.reset()
        agent.reset()
        i += 1
        done = False
        truncated = False
        j = 0
        while not done and not truncated:
            obs, reward, done, truncated, info = eval_env.step(action[2])
            obs_dict = info["observations"]
            obs_dict_batch = {}
            for key in obs_dict.keys():
                if key == 'previous_action':
                   
                    obs_dict_batch[key] = np.concatenate((np.array([obs_dict[key]]), np.array([obs_dict[key]]),np.array([obs_dict[key]])), axis=0)
                    continue
                obs_dict_batch[key] = np.concatenate((np.array([obs_dict[key]]),np.array([obs_dict[key]]),np.array([obs_dict[key]])), axis=0)
            
            #print(obs_dict_batch["previous_action"])
            _, action , log_prob = agent(model_input_dict=obs_dict_batch)
            #print("prev_action", info["observations"]["previous_action"])
            #print("action:", action)
            #print(obs_dict_batch["previous_action"])
            _, _ , test_prob = agent(model_input_dict=obs_dict_batch, actions=action)
            assert((log_prob == test_prob).all())
            #print(log_prob)
            # rescale action to be between -1 and 1, 0.05 is one
            #action = action/0.05
            eval_env.render()
