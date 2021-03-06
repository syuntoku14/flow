"""Runs the environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the DDPG algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json
import argparse
from itertools import product
import numpy as np

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
    python ddpg_runner.py grid0
Here the arguments are:
benchmark_name - name of the benchmark to run
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
    benchmark_name = args.benchmark_name
    # number of parallel workers
    num_cpus = args.num_cpus

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # initialize a ray instance
    ray.init()

    alg_run = "DDPG"

    horizon = flow_params["env"].horizon
    sim_step = flow_params["sim"].sim_step

    agent_cls = get_agent_class(alg_run)
    # use almost defalt config
    config = agent_cls._default_config.copy()
    config["num_workers"] = num_cpus
    config["train_batch_size"] = 3000 
    config["sample_batch_size"] = 750 
    config["horizon"] = horizon
    config['timesteps_per_iteration'] = int(horizon / sim_step)
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["observation_filter"] = "NoFilter"
    config["batch_mode"] = "complete_episodes"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    config['callbacks']['on_episode_start'] = ray.tune.function(on_episode_start)
    config['callbacks']['on_episode_step'] = ray.tune.function(on_episode_step)
    config['callbacks']['on_episode_end'] = ray.tune.function(on_episode_end)
    
    # tunning parameters
    e2_list = [0.1, 0.3]
    e3_list = [0.0]
    t_min = [10.0]
        
    env_name_list = []
    i = 0
    for e2, e3, t in product(e2_list, e3_list, t_min):
        flow_params["env"].additional_params["eta1"] = 1.0# e[0]
        flow_params["env"].additional_params["eta2"] = e2 # e[1]
        flow_params["env"].additional_params["eta3"] = e3 # e[2]
        # flow_params["env"].additional_params["reward_scale"] = rew
        flow_params["env"].additional_params["t_min"] = t
        flow_params["env"].additional_params["buf_length"] = 1

        # get the env name and a creator for the environment
        create_env, env_name = make_create_env(params=flow_params, version=0)
        env_name = env_name + '_[eta1, eta2, eta3]:[{}, {}, {}]'.format(1.0, e2, e3) + '_t_min:{}'.format(t)
        env_name_list.append(env_name)
        # Register as rllib env
        register_env(env_name, create_env)
        
    exp_list = []
    for env_name in env_name_list:
        exp_tag = {
            "run": alg_run,
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 25,
            "max_failures": 999,
            "stop": {
                "training_iteration": 50
            },
            "num_samples": 1,
        }
        exp_list.append(Experiment.from_json(args.exp_tag, exp_tag))

    trials = run_experiments(
        experiments=exp_list
    )
