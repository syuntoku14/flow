"""Runs the environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json, pickle
import argparse
from itertools import product
import numpy as np
from flow.core.params import InFlows

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import Experiment, run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

EXAMPLE_USAGE = """
example usage:
    python ppo_runner.py grid0
Here the arguments are:
benchmark_name - name of the benchmark to run
num_rollouts - number of rollouts to train across
num_cpus - number of cpus to use for training
"""

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

parser.add_argument(
    "--exp_tag", type=str, help="experiment tag")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=60,
    help="The number of rollouts to train over.")

# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=63,
    help="The number of cpus to use.")


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["cost1"] = []
    episode.user_data["cost2"] = []
    episode.user_data["mean_vel"] = []
    episode.user_data["outflow"] = []


def on_episode_step(info):
    episode = info["episode"]
    agent_ids = episode.agent_rewards.keys()
    infos = [episode.last_info_for(id_[0]) for id_ in agent_ids]
    cost1, cost2, mean_vel, outflow = 0, 0, 0, 0
    if len(infos) != 0:
        cost1 = np.mean([info['cost1'] for info in infos])
        cost2 = np.mean([info['cost2'] for info in infos])
        mean_vel = np.mean([info['mean_vel'] for info in infos])
        outflow = np.mean([info['outflow'] for info in infos])
    episode.user_data["cost1"].append(cost1)
    episode.user_data["cost2"].append(cost2)
    episode.user_data["mean_vel"].append(mean_vel)
    episode.user_data["outflow"].append(outflow)
    
    
def on_episode_end(info):
    episode = info["episode"]
    cost1 = np.sum(episode.user_data["cost1"])
    cost2 = np.sum(episode.user_data["cost2"])
    mean_vel = np.mean(episode.user_data["mean_vel"])
    outflow = np.mean(episode.user_data["outflow"][-500:])  # 1/3 of the whole steps
    episode.custom_metrics["cost1"] = cost1
    episode.custom_metrics["cost2"] = cost2
    episode.custom_metrics["system_level_velocity"] = mean_vel
    episode.custom_metrics["outflow_rate"] = outflow

if __name__ == "__main__":
    args = parser.parse_args()
    # base environment to resume training
    base = '/headless/ray_results/resume/' + \
        'PPO_MultiWaveAttenuationMergePOEnvBufferedObs-v0_[FLOW_RATE, FLOW_RATE_MERGE, RL_PENETRATION]:[2097,146,0.127]_0_2019-04-28_23-23-18c2m88oa_'
    checkpoint = 250
    config_path = base + '/params.pkl'
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus
    # initialize a ray instance
    ray.init()
    
    # training_iter
    training_iter = 1
    exp_list = []
    
    for i in range(training_iter):
        benchmark = __import__(
            "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
        flow_params = benchmark.buffered_obs_flow_params
        
        # inflow rate at the highway
        FLOW_RATE = np.random.randint(1900, 2300)
        FLOW_RATE_MERGE = np.random.randint(90, 150)
        # percent of autonomous vehicles
        RL_PENETRATION = 0.08 + np.random.rand() * 0.1

        inflow = InFlows()
        inflow.add(
            veh_type="human",
            edge="inflow_highway",
            vehs_per_hour=int((1 - RL_PENETRATION) * FLOW_RATE),
            #probability=FLOW_PROB,
            departLane="free",
            departSpeed=10)
        inflow.add(
            veh_type="rl",
            edge="inflow_highway",
            vehs_per_hour=int(RL_PENETRATION * FLOW_RATE),
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

        # generate new flow_params
        net = flow_params['net']
        net.inflows = inflow
        create_env, env_name = make_create_env(params=flow_params, version=0)
        env_name = env_name + '_[FLOW_RATE, FLOW_RATE_MERGE, RL_PENETRATION]:[{},{},{:.3f}]'.format(FLOW_RATE, FLOW_RATE_MERGE, RL_PENETRATION)
        # Register as rllib env
        register_env(env_name, create_env)
        
        checkpoint_path = base + '/checkpoint_{}/checkpoint-{}'.format(checkpoint, checkpoint)
        config_path = base + '/params.pkl'
        
        with open(config_path, mode='rb') as f:
            config = pickle.load(f)
        exp_tag = {
            "run": 'PPO',
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 25,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": checkpoint + 50
            },
            "restore": checkpoint_path,
            "num_samples": 1,
        }
        exp_list.append(Experiment.from_json(args.exp_tag, exp_tag))
        
    trials = run_experiments(
        experiments=exp_list
    )
