import math as m

import numpy as np
from scipy.stats import truncnorm

from scipy.stats import norm

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
    def __init__(self, current_speed=0.0, deterministic=False, horizon=0.2, subsample=20,gap_size_max_speed=10):
        # initalize parent
        super(StochasticContinousFTGAgent, self).__init__()
        self.deterministic = deterministic
        self.horizon = horizon 
        self.subsample = 20
        self.current_angle = 0.0
        self.current_velocity = current_speed
        self.max_change = 0.05
        self.exp_decay = 0.1
        self.gap_max = gap_size_max_speed
        self.gap_blocker = 2
        self.start_velocity = current_speed
        self.speed_multiplier = 2.0 # higher values slower agent

    def get_delta_angle(self, target, current):
        #print("delta")
        angle_diff = target - current
        if np.abs(angle_diff) < 0.001:
            #print(angle_diff)
            delta_angle = 0.0
            return delta_angle, current
        #print(angle_diff)
        delta_angle = np.sign(angle_diff) * min(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/abs(angle_diff)))
        new_angle = current + delta_angle
        #print(new_angle)
        new_angle = np.clip(new_angle, -3*np.pi/4, 3*np.pi/4)
        return delta_angle, new_angle
    
    def compute_speed(self, gap):
        gap_size = len(gap)
        if len(gap)==0:
            return 0.0
        # if gap size is 10 full speed else, linear
        gap_size = min(gap_size, self.gap_max)
        #print("gap size", gap_size)
        # max ray in gap 
        max_ray = max(gap)
        # clip max ray between 0 and 10
        max_ray = np.clip(max_ray, 0.0, self.speed_multiplier)
        speed = 0.5 + (max_ray / self.speed_multiplier) * 1.5
        return speed
    
    def reset(self):
        self.current_angle = 0.0
        self.current_velocity = self.start_velocity

    def get_delta_speed(self, target, current):
        speed_diff = target - current
        if np.abs(speed_diff) > 0.001:
            #print(speed_diff)
            delta_speed = np.sign(speed_diff) * min(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/abs(speed_diff)))
        else:
            delta_speed = 0.0
        new_speed = current + delta_speed
        new_speed = np.clip(new_speed, self.start_velocity, 2.0)
        return delta_speed, new_speed
    
    def __str__(self):
        return f"StochasticContinousFTGAgent_{self.speed_multiplier}_{self.gap_blocker}_{self.horizon}"
    
    def compute_target(self, scans):
        assert scans.ndim == 1, "Scans should be a 1D array"
        horizon = self.horizon
        while horizon>0.1:
            scans_temp = scans.copy()
            scans_temp[scans < horizon] = -1.0
            if np.all(scans_temp < 0.0):
                horizon -= 0.05
            else:
                break
        # block 10% of the left and rightmost scans
        scans_temp[:int(len(scans_temp)*0.15)] = -1.0
        scans_temp[-int(len(scans_temp)*0.15):] = -1.0

        scans = scans_temp
        negative_indices = np.where(scans < 0.0)[0]
        for index in negative_indices:
            start_index = max(0, index - self.gap_blocker)
            end_index = min(len(scans), index + self.gap_blocker + 1)
            scans[start_index:end_index] = -1.0
        # now lets find the index of the max
        max_index = np.argmax(scans)
        #print(max_index)
        max_value = scans[max_index]
        
        left_scans = scans[max_index::-1]
        # find first negative

        first_left = np.argwhere(left_scans < 0.0)
        if first_left.size == 0:
            first_left = 0
        else:
            first_left = max(max_index-first_left[0][0],0)
        
        right_scans = scans[max_index:]

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
        
        if len(scans[first_left:first_right]) == 0:
            gap = [0.5]
        else:
            gap = scans[first_left:first_right]
        target_speed = self.compute_speed(gap)
        return target_angle, target_speed
    
    # use the prev action from the model_input_dict to make this stateless
    def __call__(self, model_input_dict, action=None, std_angle=None, std_speed=0.1):
        # from the model_input_dict extract the lidar_occupancy
        # input can be a torch tensor or a numpy array
        if action is not None:
            assert action.ndim == 1, "Action should be a 1D array"  
        #if not(self.deterministic):
        std_angle = 0.2 if std_angle is None else std_angle
        std_speed = 0.1 if std_speed is None else std_speed
        scans = model_input_dict['lidar_occupancy'][0].copy()
        prev_action = model_input_dict['prev_action'][0].copy()
        current_angle = prev_action[0]
        current_velocity = prev_action[1]
        
        target_angle, target_speed = self.compute_target(scans)
        #print(target_angle)
        # now we need to implement delta change
        """
        if not(self.deterministic) or action is not None:
            stochastic_target_angle = np.random.normal(target_angle, std_angle)
            stochastic_target_speed = np.random.normal(target_speed, std_speed)
            stochastic_target_angle = np.clip(stochastic_target_angle, -3*np.pi/4, 3*np.pi/4)
            stochastic_target_speed = np.clip(stochastic_target_speed, self.start_velocity, 2.0)
            print("stochastic target angle", stochastic_target_angle)
            # Compute log probabilities
            if action is not None:
                # need to recover the originial target angle and speed
                stochastic_target_angle = prev_action[0] + action[0]
                print("input target angle", stochastic_target_angle)
                stochastic_target_speed = prev_action[1] + action[1]

            log_prob_angle = norm.logpdf(stochastic_target_angle, target_angle, std_angle)
            log_prob_speed = norm.logpdf(stochastic_target_speed, target_speed, std_speed)

            # Combine log probabilities
            log_prob = log_prob_angle + log_prob_speed
        else:
            log_prob = 0.0
        """
        # clip target angle between -3/4 pi and 3/4 pi
        
        # if the gap is not existent, then make a fake gap that is in the middle
        
        delta_angle, current_angle = self.get_delta_angle(target_angle, current_angle)
        delta_speed, current_velocity = self.get_delta_speed(target_speed, current_velocity)
        
        means = np.array([delta_angle,delta_speed]) / 0.05 
        # clip means to be between -1 and 1
        lower, upper = -1.0, 1.0

        # Adjust means and std if necessary
        means = np.clip(means, lower, upper)

        # Scale the bounds for the truncated normal distribution
        a, b = (lower - means) / std_angle, (upper - means) / std_angle

        if not self.deterministic:
            
            targets = np.random.normal(means, std_angle)
        else:
            targets = means
            
        targets = np.clip(targets, lower, upper)
        # clip targets to be between
        if action is not None:
            assert (action <= upper).all() and (action >= lower).all(), f"Action should be between -1 and 1 {action}"
            # Create a truncated normal distribution object
            dist = truncnorm(a, b, loc=means, scale=std_angle)
            log_prob = dist.logpdf(action)

        else:
            # Similarly for targets
            dist = truncnorm(a, b, loc=means, scale=std_angle)
            log_prob = dist.logpdf(targets)

        # TODO! make stochastic and calculate log prob
        return  targets , None , np.sum(log_prob, axis=-1)

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
    agent = StochasticContinousFTGAgent(current_speed = 0.0, deterministic=True)
    rays = np.ones((1,54,)) * 0.19
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
        )

    action = np.array([0.0,0.0])
    i = 0
    while i < 6:
        eval_env.reset()
        agent.reset()
        i += 1
        done = False
        truncated = False
        j = 0
        while not done and not truncated:
            obs, reward, done, truncated, info = eval_env.step(action)

            obs_dict = {'lidar_occupancy':  np.array([info["observations"]["lidar_occupancy"]])}
            obs_dict['prev_action'] = np.array(info["observations"]["previous_action"])
            action, _ , log_prob = agent(model_input_dict=obs_dict)
            _, _ , test_prob = agent(model_input_dict=obs_dict, action=action)
            assert(log_prob == test_prob)
            print(log_prob)
            # rescale action to be between -1 and 1, 0.05 is one
            #action = action/0.05
            eval_env.render()