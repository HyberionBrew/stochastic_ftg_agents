import math as m

import numpy as np



class BaseAgent(object):
    def __init__(self):
        # self.env = env
        #self.action_space = env.action_space
        #self.observation_space = env.observation_space
        # there are 1080 rays over 270 degrees, calculate the angle increment
        self.angle_increment = 270. / 1080. * np.pi / 180.
    
    def __call__(self, obs, std=None, actions=None): 
        # Std is actually just ignored       
        if actions is None:
            speed, action = self.get_action(obs)
            return None, action, None
        
        else:
            log_prob = self.get_log_probs(obs, actions)
            return None, None, log_prob
        
    def get_action(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

    def lidar_ray_to_steering(self, ray, sub_sample):
        ray_angle = (ray *sub_sample) * self.angle_increment
        # rays start from -135
        ray_angle -= 3*np.pi/4
        return ray_angle

"""
@input lidar rays need to be already normalized
"""
class StochasticContinousFTGAgent(BaseAgent):
    def __init__(self, deterministic=False, horizon=0.2, subsample=20,gap_size_max_speed=10):
        # initalize parent
        super(StochasticContinousFTGAgent, self).__init__()
        self.deterministic = deterministic
        self.horizon = horizon 
        self.subsample = 20
        self.current_angle = 0.0
        self.current_velocity = 0.0
        self.max_change = 0.05
        self.exp_decay = 0.1
        self.gap_max = gap_size_max_speed
        self.gap_blocker = 3

    def get_delta_angle(self, target, current):
        angle_diff = target - current
        delta_angle = np.sign(angle_diff) * min(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/abs(angle_diff)))
        new_angle = current + delta_angle
        new_angle = np.clip(new_angle, -3*np.pi/4, 3*np.pi/4)
        return delta_angle, new_angle
    def compute_speed(self, gap):
        gap_size = len(gap)
        # if gap size is 10 full speed else, linear
        gap_size = min(gap_size, self.gap_max)
        speed = 0.5 + (gap_size / 10.0) * 1.5
        return speed
    
    def get_delta_speed(self, target, current):
        speed_diff = target - current
        delta_speed = np.sign(speed_diff) * min(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/abs(speed_diff)))
        new_speed = current + delta_speed
        new_speed = np.clip(new_speed, 0.5, 2.0)
        return delta_speed, new_speed

    def __call__(self, model_input_dict, std=None):
        # from the model_input_dict extract the lidar_occupancy
        # input can be a torch tensor or a numpy array
        scans = model_input_dict['lidar_occupancy'][0].copy()
        assert scans.ndim == 1, "Scans should be a 1D array"
        
        scans[scans < self.horizon] = -1.0
        negative_indices = np.where(scans < 0.0)[0]
        for index in negative_indices:
            start_index = max(0, index - self.gap_blocker)
            end_index = min(len(scans), index + self.gap_blocker + 1)
            scans[start_index:end_index] = -1.0
        # now lets find the index of the max
        max_index = np.argmax(scans)
        max_value = scans[max_index]
        # find the window of non-negative values around the target index
        # find the first non-negative value to the left
        # padd scans with -1.0 left and right
        left_scans = scans[max_index::-1]
        # find first negative

        first_left = np.argwhere(left_scans < 0.0)
        #print("first left")
        #print(first_left)
        if first_left.size == 0:
            first_left = 0
        else:
            first_left = max(max_index-first_left[0][0],0)
        
        right_scans = scans[max_index:]
        # find first negative
        first_right = np.argwhere(right_scans < 0.0)
        if first_right.size == 0:
            first_right = len(scans) -1
        else:
            first_right = min(max_index + first_right[0][0], len(scans)-1)

        target_ray = int((first_left + first_right) // 2) 

        if max_value < 0:
            target_angle = 0.0
        else:
            target_angle = self.lidar_ray_to_steering(target_ray, self.subsample)
        #print(target_angle)
        # now we need to implement delta change
        target_angle = target_angle #- self.current_angle
        # clip target angle between -3/4 pi and 3/4 pi
        target_angle = np.clip(target_angle, -3*np.pi/4, 3*np.pi/4)
        target_speed = self.compute_speed(scans[first_left:first_right])
        delta_angle, current_angle = self.get_delta_angle(target_angle, self.current_angle)
        delta_speed, current_velocity = self.get_delta_speed(target_speed, self.current_velocity)
        self.current_angle = current_angle
        self.current_velocity = current_velocity
        return np.array([[delta_speed, delta_angle]])
    
if __name__ == "__main__":
    agent = StochasticContinousFTGAgent()
    rays = np.ones((1,54,)) * 0.19
    rays[:,:10] += 0.1
    #rays += 0.5
    #print(rays)
    model_input_dict = {'lidar_occupancy': rays}
    #print(model_input_dict)
    for _ in range(40):
        target = agent(model_input_dict)
        print(target)