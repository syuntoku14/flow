"""Benchmark for multi-merge

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%. This is multi-agent scenario

- **Action Dimension**: (1, )
- **Observation Dimension**: (5, )
- **Horizon**: 750 steps
- **Warmup**: 100 steps
"""

from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.multiagent_envs.merge import ADDITIONAL_ENV_PARAMS
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController, SimCarFollowingController

# time horizon of a single rollout
HORIZON = 750
SIM_STEP = 0.2  # same as 300 seconds
WARMUP = 100
# inflow rate at the highway
FLOW_RATE = 2000
FLOW_RATE_MERGE = 100
# percent of autonomous vehicles
RL_PENETRATION = 0.1

# inflow probability
#FLOW_PROB = 0.2
#FLOW_PROB_MERGE = 0.05
#FLOW_PROB_RL = 0.05

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = deepcopy(ADDITIONAL_NET_PARAMS)
additional_net_params["merge_lanes"] = 1
additional_net_params["highway_lanes"] = 1
additional_net_params["pre_merge_length"] = 600

additional_env_params = deepcopy(ADDITIONAL_ENV_PARAMS)

# RL vehicles constitute 5% of the total number of vehicles
vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=5)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=0)

# Vehicles are introduced from both sides of merge, with RL vehicles entering
# from the highway portion as well
inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="inflow_highway",
    vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
    #probability=FLOW_PROB,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="rl",
    edge="inflow_highway",
    vehs_per_hour=RL_PENETRATION * FLOW_RATE,
    #probability=FLOW_PROB_MERGE,
    departLane="free",
    departSpeed=10)
inflow.add(
    veh_type="human",
    edge="inflow_merge",
    vehs_per_hour=FLOW_RATE_MERGE,
    #probability=FLOW_PROB_RL,
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="multi_merge",

    # name of the flow environment the experiment is running on
    env_name="MultiWaveAttenuationMergePOEnv",

    # name of the scenario class the experiment is running on
    scenario="MergeScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=SIM_STEP,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        sims_per_step=2,
        warmup_steps=WARMUP,
        additional_params=additional_env_params
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

buffered_obs_flow_params = deepcopy(flow_params)
buffered_obs_flow_params["env_name"] = "MultiWaveAttenuationMergePOEnvBufferedObs"

custom_rew_flow_params = deepcopy(flow_params)
custom_rew_flow_params["env_name"] = "MultiWaveAttenuationMergePOEnvCustomRew"

gail_flow_params = deepcopy(flow_params)
gail_flow_params["env_name"] = "MultiWaveAttenuationMergePOEnvGAIL"
