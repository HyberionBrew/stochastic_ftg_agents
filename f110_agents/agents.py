import math as m
pi = m.pi

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