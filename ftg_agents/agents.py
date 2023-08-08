import tensorflow as tf
import math as m
pi = tf.constant(m.pi)

class BaseAgent(object):

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        # there are 1080 rays over 270 degrees, calculate the angle increment
        self.angle_increment = 270. / 1080. * pi / 180.
    
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
        ray_angle -= 3*pi/4
        return ray_angle

class StochasticFTGAgent(BaseAgent):

    def __init__(self, env, sub_sample=10, speed=1.0, cutoff=10):
        super(StochasticFTGAgent, self).__init__(env)
        self.sub_sample = sub_sample
        self.speed = speed
        self.cutoff = cutoff

    def get_speed(self, obs):
        # sample from a normal distribution with mean speed and std cutoff
        speed = tf.random.normal([1], mean=self.speed, stddev=0.2)
        return speed[0]
    
    def get_probs(self,obs):
        obs = obs[::self.sub_sample]
        # add a zero at start and end
        obs = tf.concat([[0.], obs, [0.]], 0)
        # smooth the observations
        # import matplotlib.pyplot as plt
        window_size = 25
        obs = tf.expand_dims(tf.expand_dims(obs, 0), -1)  # add two dimensions for max_pool1d
        smoothed_obs_neg = tf.nn.max_pool1d(-obs, window_size, 1, 'SAME')
        smoothed_obs = -tf.squeeze(smoothed_obs_neg, [0, -1])  # remove the extra dimensions and negate
        obs = smoothed_obs
        # remove zero at start and end
        obs = obs[1:-1]
        # set to zero where not close enough to the max
        # obs = tf.where(obs < 0.9 * tf.reduce_max(obs), tf.zeros_like(obs), obs)
        # set to zero where more than x indices away from the max
        # obs = tf.where(tf.abs(tf.argmax(obs) - tf.range(0, 1080, self.sub_sample)) > 10, tf.zeros_like(obs), obs)
        # Create a range tensor from [0, ..., length-1]
        argmax_obs = tf.cast(tf.argmax(obs), tf.int32)
        indices = tf.range(tf.shape(obs)[0], dtype=tf.int32)
        mask = tf.math.logical_and(argmax_obs - self.cutoff <= indices, indices <= argmax_obs + self.cutoff)
        obs = tf.where(mask, obs, 0.001)

        probabilities = obs 
        total_length = tf.reduce_sum(obs)
        # print(probabilities)
        probabilities = probabilities / total_length
        return probabilities
    
    def get_action(self, obs):
        # obs is an array [1080] of lidar readings
        # first subsample the lidar array with sub_sample
        probabilities = self.get_probs(obs)
        # create a categorical distribution based on the probabilities
        # TODO! make this continous
        dist = tf.random.categorical(tf.math.log([probabilities]), 1)
        action = dist[0][0].numpy()
        action = self.lidar_ray_to_steering(action, self.sub_sample)

        return self.get_speed(obs), action
    
    def get_log_probs(self, obs, action):
        probabilities = self.get_probs(obs)
        # Compute log probabilities
        log_probs = tf.nn.log_softmax(probabilities)

        # Get log probability of chosen action
        log_prob_action = log_probs[action].numpy()
        return log_prob_action
    
class DeterministicFTGAgent(BaseAgent):
    def __init__(self, env, sub_sample=10, speed=1.0, cutoff=10):
        super(DeterministicFTGAgent, self).__init__(env)
        self.sub_sample = sub_sample
        self.speed = speed
        self.cutoff = cutoff

    def get_speed(self, obs):
        # sample from a normal distribution with mean speed and std cutoff
        return self.speed

    def get_action(self, obs):
        obs = obs[::self.sub_sample]
        # add a zero at start and end
        obs = tf.concat([[0.], obs, [0.]], 0)
        # smooth the observations
        # import matplotlib.pyplot as plt
        window_size = 25
        obs = tf.expand_dims(tf.expand_dims(obs, 0), -1)  # add two dimensions for max_pool1d
        smoothed_obs_neg = tf.nn.max_pool1d(-obs, window_size, 1, 'SAME')
        smoothed_obs = -tf.squeeze(smoothed_obs_neg, [0, -1])  # remove the extra dimensions and negate
        obs = smoothed_obs
        # remove zero at start and end
        obs = obs[1:-1]
        # set to zero where not close enough to the max
        # obs = tf.where(obs < 0.9 * tf.reduce_max(obs), tf.zeros_like(obs), obs)
        # set to zero where more than x indices away from the max
        # obs = tf.where(tf.abs(tf.argmax(obs) - tf.range(0, 1080, self.sub_sample)) > 10, tf.zeros_like(obs), obs)
        # Create a range tensor from [0, ..., length-1]
        argmax_obs = tf.cast(tf.argmax(obs), tf.int32)
        action = argmax_obs.numpy()
        print(action)
        action = self.lidar_ray_to_steering(action, self.sub_sample)
        return self.get_speed(obs), action , 1.0
    

class StochasticFTGAgentRandomSpeed(StochasticFTGAgent):
    # inherit from the StochasticFTGAgent
    def __init__(self, env, sub_sample=10, speed=3.0, cutoff=10):
        super(StochasticFTGAgentRandomSpeed, self).__init__(env, sub_sample, speed, cutoff= cutoff)
        # self.t = 0
    def get_speed(self, obs):
        
        return tf.random.uniform([1], minval=0.5, maxval=self.speed)[0]

class StochasticFTGAgentDynamicSpeed(StochasticFTGAgent):
    # inherit from the StochasticFTGAgent
    def __init__(self, env, sub_sample=10, speed=0.5, cutoff=10):
        super(StochasticFTGAgentDynamicSpeed, self).__init__(env, sub_sample, speed, cutoff= cutoff)
    def get_speed(self, obs):
        # divide the largest obs by self.speed
        speed = tf.reduce_max(obs) / self.speed
        print(speed)
        # normal distribution with mean speed and std cutoff
        speed = tf.random.normal([1], mean=speed, stddev=0.2)
        # sample speed from the normal distr

        return speed[0]