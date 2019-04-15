"""Benchmark for ringroad.

Ringroad in which the actions are specifying a desired velocity in a segment
of space. 

- **Action Dimension**: (?, )
- **Observation Dimension**: (?, )
- **Horizon**: 1500 steps
"""

from copy import deepcopy
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController, ContinuousRouter, RLController


HORIZON = 1500

vehicles = VehicleParams()

vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {
        "noise": 0.2
    }),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=49)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
    ),
    num_vehicles=1)

ADDITIONAL_NET_PARAMS = {
    # length of the ring road
    "length": 1400,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 60,
    # resolution of the curves on the ring
    "resolution": 40
}

flow_params = dict(
    # name of the experiment
    exp_tag="loop_0",

    # name of the flow environment the experiment is running on
    env_name="AccelEnv",

    # name of the scenario class the experiment is running on
    scenario="LoopScenario",

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),


    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 20,
            "max_accel": 3,
            "max_decel": 3,
            "sort_vehicles": False
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(bunching=20),
)