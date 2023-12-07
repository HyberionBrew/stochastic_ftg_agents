import math as m

import numpy as np
from scipy.stats import truncnorm

from scipy.stats import norm

class BaseAgent(object):
    def __init__(self, max_change=0.05, exp_decay=0.1, start_velocity=0.0, max_speed=2.0 ):
        # self.env = env
        #self.action_space = env.action_space
        #self.observation_space = env.observation_space
        # there are 1080 rays over 270 degrees, calculate the angle increment
        self.angle_increment = 270. / 1080. * np.pi / 180.
        self.max_change = max_change
        self.exp_decay = exp_decay
        self.start_velocity = start_velocity
        self.max_speed = max_speed

    def __call__(self, obs: dict, std=None, actions=None): 
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
    
    def get_delta_angle(self, target, current):
        #print("delta")
        angle_diff = target - current[:None]
        angle_diff[np.isclose(angle_diff,0.0)] = 0.0001 
        #print(angle_diff)
        #print(self.max_change)
        #print(np.array(self.max_change))
        delta_angle = np.sign(angle_diff) * np.minimum(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/np.abs(angle_diff)))
        new_angle = current + delta_angle
        #print(new_angle)
        new_angle = np.clip(new_angle, -3*np.pi/4, 3*np.pi/4)
        return delta_angle, new_angle

    def get_delta_speed(self, target, current):
        speed_diff = target - current
        speed_diff[np.isclose(speed_diff,0.0)] = 0.0001 
            #print(speed_diff)
        delta_speed = np.sign(speed_diff) * np.minimum(self.max_change, self.max_change * np.exp(-self.exp_decay * 1/abs(speed_diff)))

        new_speed = current + delta_speed
        new_speed = np.clip(new_speed, self.start_velocity, 2.0)
        return delta_speed, new_speed

"""
@input lidar rays need to be already normalized
"""
class StochasticContinousFTGAgent(BaseAgent):
    def __init__(self, current_speed=0.0, deterministic=False, gap_blocker = 2, max_speed=2.0, horizon=0.2, subsample=20,gap_size_max_speed=10, speed_multiplier=2.0, std=0.3):
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
        self.gap_blocker = gap_blocker
        self.start_velocity = current_speed
        self.speed_multiplier = speed_multiplier# higher values slower agent
        self.std = std
        self.max_speed = max_speed 

 
    
    def compute_speed(self, gap):
        assert gap.ndim == 1, "Gap should be a 1D array"
        """
        gap_size = len(gap)
        if len(gap)==0:
            return 0.0
        # if gap size is 10 full speed else, linear
        gap_size = min(gap_size, self.gap_max)
        #print("gap size", gap_size)
        # max ray in gap 
        max_ray = max(gap)
        # clip max ray between 0 and 10
        # print(max_ray)
        # gap larger than 0 
        larger_than_zero = gap[np.where(gap>0.0)[0]]
        print(gap)
        max_ray = np.sum(larger_than_zero)**2 #/len(larger_than_zero)
        print(np.sum(larger_than_zero))
        print(len(larger_than_zero))
        max_ray = np.clip(max_ray, 0.0, self.speed_multiplier)
        print(max_ray)
        print("------")
        """
        max_ray = gap  #[0] # TODO gap is a value of shape (batch, 1)
        max_ray = np.clip(max_ray, 0.0, self.speed_multiplier)
        speed = 0.0 + (max_ray / self.speed_multiplier) * self.max_speed
        # print(max_ray)
        return speed
    
    def reset(self):
        self.current_angle = 0.0
        self.current_velocity = self.start_velocity


    
    def __str__(self):
        return f"StochasticContinousFTGAgent_{self.speed_multiplier}_{self.gap_blocker}_{self.horizon}_{self.std}_{self.max_speed}"
    
    def set_zeros_around_max(self,scans_processed, max_indices):
        batch_size, num_scans = scans_processed.shape

        # Identify where the -1.0 separators are
        separators = scans_processed == -1.0

        # Create a cumulative sum mask to identify contiguous regions
        cumsum_mask = np.cumsum(separators, axis=1)

        # For each batch, find the cumsum value at the max_index
        regions = cumsum_mask[np.arange(batch_size), max_indices]

        # Create masks for the contiguous regions containing the max_index
        contiguous_region_mask = (cumsum_mask == regions[:, None])

        # Zero out elements not in the contiguous region of max_index
        result = np.where(contiguous_region_mask, scans_processed, 0.0)
        # set the first non zero element in each row to 0.0
        # above code has small bug and leaves first -1.0 in conitnous result region
        # easy fix ~.+
        result = np.where(result == -1.0, 0, result)

        return result


    def compute_target(self, scans_batch):
        assert scans_batch.ndim == 2, "Scans should be a 2D array with shape (batches, dim)"
        horizon = self.horizon
        og_scans = scans_batch.copy()
        # block 15% of the left and rightmost scans
        block_length = int(len(scans_batch[0])*0.15)
        scans_batch[:,:block_length] = -1.0
        scans_batch[:,-block_length:] = -1.0
        # now lets iterate over the horizon
        #scans_finished_indices = np.zeros(len(scans_batch))
        scans_done = np.zeros(len(scans_batch))
        scans_processed = np.zeros_like(scans_batch) - 1.0

        while horizon > 0.05:
            #print(horizon)
            #print("done:", scans_done)
            scans_temp = scans_batch.copy()
            scans_missing_indices = np.where(scans_done == 0.0)[0]

            current_scan = scans_temp[scans_missing_indices] #= -1.0
            current_scan[current_scan < horizon] = -1.0

            # check each row if all are negative
            # block left and rights
            # get the index of all negatives
            negatives_batch, negatives_dim = np.where(current_scan < 0.0)
            # block out the gaps
            for i in range(1, self.gap_blocker + 1):
                negatives_dim_left = negatives_dim - i
                negatives_dim_right = negatives_dim + i
                # ensure we dont go out of bounds
                negatives_dim_left = np.clip(negatives_dim_left, 0, len(current_scan[0])-1)
                negatives_dim_right = np.clip(negatives_dim_right, 0, len(current_scan[0])-1)

                current_scan[negatives_batch, negatives_dim_left] = -1.0
                current_scan[negatives_batch, negatives_dim_right] = -1.0
            # check if all are negative

            # print(negatives)
            positives = np.any(current_scan > 0.0, axis=1)
            # where we have positives we write into scans_processed
            scans_processed[scans_missing_indices[positives]] = current_scan[positives]
            scans_done[scans_missing_indices[positives]] = 1.0
            #print(scans_done)
            #print(scans_processed)
            
            if np.any(scans_done == 0.0):
                horizon -= 0.05
            else:
                break

        # now all scans are processed
        # now lets find the index of the max
        max_indices = np.argmax(scans_processed, axis=1)
        # where the max is negative we set the index to scans.shape[1]//2
        #kinda hacky but implicitly true ~.~
        # where max_indices are 0 we set scans_processed to 0.5
        scans_processed[max_indices == 0, len(scans_processed[0])//2] = 0.5
        max_indices[max_indices == 0] = len(scans_processed[0])//2

        #print(max_indices)
        # for each find the left and rightmost non zero
        # left

        #right_scans = scans_processed[max_indices::]
        #left_scans = scans_processed[max_indices::-1]
        # find first negative
        
        #first_left = np.argwhere(left_scans < 0.0, axis=1)
        #first_right = np.argwhere(right_scans < 0.0,axis=1)
        new_scans = self.set_zeros_around_max(scans_processed, max_indices)
        #print(new_scans)
        # in each row find first and last non zero element
        left_indices = np.where(new_scans > 0.0)#[0]
        scans = new_scans.copy()
        scans[scans > 0] = 1.0
        left_indices = np.argmax(scans, axis=1)

        right_indices = np.argmax(scans[:,::-1], axis=1)
        right_indices = len(scans[0]) - right_indices - 1
        target_rays = (left_indices + right_indices) // 2

        target_angle = self.lidar_ray_to_steering(target_rays, self.subsample)
        target_speed = self.compute_speed(og_scans[:,og_scans.shape[1]//2])
        return target_angle, target_speed

    def compute_target_old(self, scans):
        # print(scans)
        assert scans.ndim == 1, "Scans should be a 1D array"
        horizon = self.horizon
        og_scans = scans.copy()
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
        target_speed = self.compute_speed([og_scans[int(len(og_scans)//2)]])
        return target_angle, target_speed
    
    # use the prev action from the model_input_dict to make this stateless
    def __call__(self, model_input_dict, actions=None, std=None):
        std_angle = self.std
        std_speed = self.std
        action =actions
        # Asserting the shape of action if it's not None
        if action is not None:
            assert action.ndim == 2 and action.shape[1] == 2, "Action should be a 2D array with shape (batches, 2)"
        assert "lidar_occupancy" in model_input_dict
        assert len(model_input_dict["lidar_occupancy"].shape) == 2, "Lidar occupancy should be a 2D array"
        assert "previous_action" in model_input_dict
        assert len(model_input_dict["previous_action"].shape) == 3, "Previous action should be a 3D (batch,1,2) array, yes weird TODO!"
        scans = model_input_dict['lidar_occupancy']
        prev_actions = model_input_dict['previous_action']
        current_angles = prev_actions[:,0, 0]
        #print(prev_actions.shape)
        #print(prev_actions[:10])
        current_velocities = prev_actions[:,0, 1]

        target_angles, target_speeds = self.compute_target(scans)  # Adapted for batch processing
        #print(target_angles)
        #print(target_speeds)
        delta_angles = target_angles - current_angles
        delta_speeds = target_speeds - current_velocities 
        # TODO! for now
        delta_angles, new_current_angles = self.get_delta_angle(target_angles, current_angles)  # Adapted for batch processing
        delta_speeds, new_current_velocities = self.get_delta_speed(target_speeds, current_velocities)  # Adapted for batch processing

        means = np.vstack((delta_angles, delta_speeds)).T / 0.05
        means = np.clip(means, -1.0, 1.0)

        a = (means - 1.0) / std_angle
        b = (means + 1.0) / std_angle

        if not self.deterministic:
            targets = np.random.normal(means, std_angle)
        else:
            targets = means

        targets = np.clip(targets, -1.0, 1.0)

        dist = truncnorm(a, b, loc=means, scale=std_angle)

        if action is not None:
            assert (action <= 1.0).all() and (action >= -1.0).all(), "Action should be between -1 and 1"
            log_probs = dist.logpdf(action).sum(axis=1)
        else:
            log_probs = dist.logpdf(targets).sum(axis=1)

        # log_probs = 1.0
        #print("mean", means)
        #print("target", targets)
        return None, targets, log_probs

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
    agent = StochasticContinousFTGAgent(current_speed = 0.0, deterministic=False, std=0.1, speed_multiplier=0.5)
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
            fake_lidar2 = np.zeros((1,54,))
            fake_lidar = np.ones((1,54,)) * 0.1
            fake_lidar[:,26] = 0.
            fake_lidar[:,30] = 1.0
            lidar = np.array([info["observations"]["lidar_occupancy"]])
            # concatenate along dim 0 fake  and real lidar
            fake_lidar = np.concatenate((fake_lidar2, fake_lidar,lidar), axis=0)
            #print(fake_lidar.shape)
            obs_dict = {'lidar_occupancy':  fake_lidar}#,info["observations"]["lidar_occupancy"]])}
            # print(fake_lidar)
            obs_dict['previous_action'] = np.array([info["observations"]["previous_action"]])
            _, action , log_prob = agent(model_input_dict=obs_dict)
            print("prev_action", info["observations"]["previous_action"])
            print("action:", action)

            _, _ , test_prob = agent(model_input_dict=obs_dict, actions=action)
            assert((log_prob == test_prob).all())
            #print(log_prob)
            # rescale action to be between -1 and 1, 0.05 is one
            #action = action/0.05
            eval_env.render()