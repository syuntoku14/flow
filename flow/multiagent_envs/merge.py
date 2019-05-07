"""
Environments for training multi-agent vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from copy import deepcopy
import numpy as np
from flow.multiagent_envs.multiagent_env import MultiEnv
from flow.core import rewards
from flow.core.params import InFlows

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    
    # not default parameters
    "t_min": 1.0,
    # weigts of cost function
    "eta1": 1.0,
    "eta2": 0.2,
    "eta3": 0.1,
    "reward_scale": 1.0,
    "FLOW_RATE": 2000,
    "FLOW_RATE_MERGE": 100,
    "RL_PENETRATION": 0.2
}

OUTFLOW_RATE_SPAN = 50

class MultiWaveAttenuationMergePOEnv(MultiEnv):

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()
        # names of the rl vehicles controlled at any step
        self.rl_veh = []
        # used for visualization
        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(5, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition"""
        if rl_actions:
            rl_ids = list(rl_actions.keys())
            accel = list(rl_actions.values())
            
            self.k.vehicle.apply_acceleration(rl_ids, accel)

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.scenario.max_speed()
        max_length = self.k.scenario.length()

        obs = collections.OrderedDict()
        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)
            
            observation = np.array([
            this_speed / max_speed,
            (lead_speed - this_speed) / max_speed,
            lead_head / max_length,
            (this_speed - follow_speed) / max_speed,
            follow_head / max_length,
            ], dtype='float32')
            obs.update({rl_id: observation})
        
        return obs

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            ## return a reward of 0 if a collision occurred
            #if kwargs["fail"]:
            #    return 0
            
            rew = collections.OrderedDict()
            info = collections.OrderedDict()
            
            scale = self.env_params.additional_params["reward_scale"]
            # weights for cost1, cost2, and cost3, respectively
            eta1 = self.env_params.additional_params["eta1"]
            eta2 = self.env_params.additional_params["eta2"]
            
            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])
            # print("cost1: {}".format(cost1))
            mean_vel = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
            outflow = self.k.vehicle.get_outflow_rate(OUTFLOW_RATE_SPAN)
            
            # penalize small time headways
            t_min = self.env_params.additional_params["t_min"]  # smallest acceptable time headway
            for rl_id in self.k.vehicle.get_rl_ids():
                cost2 = 0.0
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0.0)
                    # print("cost2: {}".format(cost2))
                rew.update({rl_id: max(eta1 * cost1 + eta2 * cost2, 0.0) * scale})
                info.update({rl_id: {'cost1': cost1, 'cost2': cost2, 'mean_vel': mean_vel, "outflow": outflow}})
                if kwargs["fail"]:
                    rew.update({rl_id: 0.0})
                    info.update({rl_id: {'cost1': cost1, 'cost2': cost2, 'mean_vel': mean_vel, "outflow": outflow}})
                    
            return rew, info

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
    

class MultiWaveAttenuationMergePOEnvOutFlowRew(MultiWaveAttenuationMergePOEnv):
    """
    Observation: (v_lead, v_lag, v, h_lead, h_lag, distance_to_merge, left_density, bottom_density)
    """
    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(8, ), dtype=np.float32)
    
    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            ## return a reward of 0 if a collision occurred
            #if kwargs["fail"]:
            #    return 0
            
            rew = collections.OrderedDict()
            info = collections.OrderedDict()
            
            scale = self.env_params.additional_params["reward_scale"]
            # weights for cost1, cost2, and cost3, respectively
            eta1 = self.env_params.additional_params["eta1"]
            eta2 = self.env_params.additional_params["eta2"]
            eta3 = self.env_params.additional_params["eta3"]
            FLOW_RATE = self.env_params.additional_params["FLOW_RATE"]
            
            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])
            # print("cost1: {}".format(cost1))
            mean_vel = np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
            outflow = self.k.vehicle.get_outflow_rate(OUTFLOW_RATE_SPAN) 
            if outflow == None:
                outflow = 0.0
            
            # penalize small time headways
            t_min = self.env_params.additional_params["t_min"]  # smallest acceptable time headway
            if rl_actions is not None:    
                for rl_id in self.k.vehicle.get_rl_ids():
                    current_edge = self.k.vehicle.get_edge(rl_id)
                    # don't control cars in inflow edge, 
                    if 'inflow' in current_edge:
                        continue
                        
                    cost2 = 0.0
                    lead_id = self.k.vehicle.get_leader(rl_id)
                    if lead_id not in ["", None] \
                            and self.k.vehicle.get_speed(rl_id) > 0:
                        t_headway = max(
                            self.k.vehicle.get_headway(rl_id) /
                            self.k.vehicle.get_speed(rl_id), 0)
                        cost2 += min((t_headway - t_min) / t_min, 0.0)
                    rew.update({rl_id: max(eta1 * cost1 + eta2 * cost2 + eta3 * outflow / FLOW_RATE, 0.0) * scale})
                    info.update({rl_id: {'cost1': cost1, 'cost2': cost2, 'mean_vel': mean_vel, "outflow": outflow}})
                    if kwargs["fail"]:
                        rew.update({rl_id: 0.0})
                        info.update({rl_id: {'cost1': cost1, 'cost2': cost2, 'mean_vel': mean_vel, "outflow": outflow}})
                    
            return rew, info
        
    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.scenario.max_speed()
        max_length = self.k.scenario.length()
        
        left_length = self.k.scenario.edge_length('left')
        left_car_ids = self.k.vehicle.get_ids_by_edge('left')
        bottom_car_ids = self.k.vehicle.get_ids_by_edge('bottom') + self.k.vehicle.get_ids_by_edge('inflow_merge')
        bottom_length = self.k.scenario.edge_length('bottom') + self.k.scenario.edge_length('inflow_merge')
        left_density = np.sum(self.k.vehicle.get_length(left_car_ids)) / left_length
        bottom_density = np.sum(self.k.vehicle.get_length(bottom_car_ids)) / bottom_length
        
        obs = collections.OrderedDict()
        for rl_id in self.k.vehicle.get_rl_ids():
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)
            distance_to_merge = left_length
            current_edge = self.k.vehicle.get_edge(rl_id)
            
            # don't control cars in inflow edge, 
            if 'inflow' in current_edge:
                continue
            
            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)
            
            # distance to the intersection
            if current_edge == 'left':
                distance_to_merge -= self.k.vehicle.get_position(rl_id)
                
            observation = np.array([
            this_speed / max_speed,
            (lead_speed - this_speed) / max_speed,
            lead_head / max_length,
            (this_speed - follow_speed) / max_speed,
            follow_head / max_length,
            distance_to_merge / left_length,
            left_density,
            bottom_density
            ], dtype='float32')
            obs.update({rl_id: observation})
        
        return obs

    
class MultiWaveAttenuationMergePOEnvBufferedObs(MultiWaveAttenuationMergePOEnvOutFlowRew):
    # obs: 3xobs + traffic info
    def __init__(self, env_params, sim_params, scenario, simulator='traci'):       
        super().__init__(env_params, sim_params, scenario, simulator)
        # historical observation
        self.buffered_obs = {}
        self.buffer_length = 3
        self.FLOW_RATE = self.env_params.additional_params["FLOW_RATE"]
        self.FLOW_RATE_MERGE = self.env_params.additional_params["FLOW_RATE_MERGE"] 
        self.RL_PENETRATION = self.env_params.additional_params["RL_PENETRATION"]
        self.flow_rate = self.FLOW_RATE
        self.flow_rate_merge = self.FLOW_RATE_MERGE
        self.rl_penetration = self.RL_PENETRATION
        
    @property
    def observation_space(self):
        """See class definition."""
        return Box(low=0, high=1, shape=(8*self.buffer_length + 3, ), dtype=np.float32)

    def get_state(self, rl_id=None, **kwargs):
        obs = super().get_state(rl_id, **kwargs)
        # add new car to buffered obs
        for key in obs.keys():
            if not key in self.buffered_obs:
                self.buffered_obs[key] = np.zeros(self.observation_space.shape, 'float32')
        # remvove exited car from buffered obs
        keys = deepcopy(list(self.buffered_obs.keys()))
        for key in keys:
            if not key in obs:
                self.buffered_obs.pop(key)
                
        # update buffered_obs
        for key, value in obs.items():
            obs_len = len(value)
            self.buffered_obs[key] = self.buffered_obs[key][obs_len:-3]
            self.buffered_obs[key] = np.hstack((self.buffered_obs[key], value))
            traffic_info = np.array([self.flow_rate / 3600, self.flow_rate_merge / 3600, self.rl_penetration])
            self.buffered_obs[key] = np.hstack((self.buffered_obs[key], traffic_info))
        
        return self.buffered_obs

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.buffered_obs = {}
        
        # perturbe the traffic condition
        self.flow_rate = self.FLOW_RATE * (0.85 + np.random.rand()*0.3)
        self.flow_rate_merge = self.FLOW_RATE_MERGE * (0.85 + np.random.rand()*0.3)
        self.rl_penetration = self.RL_PENETRATION * (0.9 + np.random.rand()*0.2)
        
        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="inflow_highway",
            vehs_per_hour=int((1 - self.rl_penetration) * self.flow_rate),
            #probability=FLOW_PROB,
            departLane="free",
            departSpeed=10)
        inflow.add(
            veh_type="rl",
            edge="inflow_highway",
            vehs_per_hour=int(self.rl_penetration * self.flow_rate),
            #probability=FLOW_PROB_MERGE,
            departLane="free",
            departSpeed=10)
        inflow.add(
            veh_type="human",
            edge="inflow_merge",
            vehs_per_hour=self.flow_rate_merge,
            #probability=FLOW_PROB_RL,
            departLane="free",
            departSpeed=7.5)
        self.scenario.net_params.inflows = inflow
        return super().reset()