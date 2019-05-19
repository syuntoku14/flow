"""Benchmark for multi-merge

Trains a small percentage of autonomous vehicles to dissipate shockwaves caused
by merges in an open network. The autonomous penetration rate in this example
is 10%. This is multi-agent scenario

- **Action Dimension**: (1, )
- **Observation Dimension**: (5, )
- **Horizon**: 750 steps
- **Warmup**: 100 steps
"""

import math
from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams, SumoLaneChangeParams, InFlows
from flow.scenarios.loop_inflow import LoopInflowScenario, \
    ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams
from flow.controllers import IDMController, RLController, \
    SimCarFollowingController, SimLaneChangeController, \
    ContinuousRouter
from flow.multiagent_envs.merge import ADDITIONAL_ENV_PARAMS

# time horizon of a single rollout
HORIZON = 750
SIM_STEP = 0.2  # same as 300 seconds
WARMUP = 100
# percent of autonomous vehicles
VEHICLE_NUM = 25
INFLOW_PROB = 0.1
RL_PENETRATION = 0.15

RL_NUM = math.ceil(VEHICLE_NUM * RL_PENETRATION)
HUMAN_NUM = VEHICLE_NUM - RL_NUM

# We consider a highway network with an upstream merging lane producing
# shockwaves
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["ring_radius"] = 600 / (2*math.pi)
additional_net_params["lane_length"] = 100
additional_net_params["inner_lanes"] = 1
additional_net_params["outer_lanes"] = 1

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()

# RL vehicles constitute 5% of the total number of vehicles
# note that the vehicles are added sequentially by the scenario,
# so place the merging vehicles after the vehicles in the ring
vehicles = VehicleParams()

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode=0,
    ),
    num_vehicles=RL_NUM)
vehicles.add(
    veh_id="idm",
    acceleration_controller=(IDMController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=HUMAN_NUM,
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    lane_change_params=SumoLaneChangeParams())

inflow = InFlows()
inflow.add(
    veh_type="idm",
    edge="top",
    probability=INFLOW_PROB,
    departLane="free",
    departSpeed=7.5)

flow_params = dict(
    # name of the experiment
    exp_tag="multi_loop_merge",

    # name of the flow environment the experiment is running on
    env_name="MultiWaveAttenuationLoopMergePOEnvBufferedObs",

    # name of the scenario class the experiment is running on
    scenario="LoopInflowScenario",

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
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=additional_net_params,
        inflows=inflow
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(shuffle=True, x0=50, spacing="uniform", additional_params={"merge_bunching": 0}),
)
