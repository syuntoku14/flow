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
from ray.tune import run_experiments
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


if __name__ == "__main__":
    benchmark_name = 'grid0'
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
    config["train_batch_size"] = horizon 
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

    # tunning parameters
    eta = [[1.0, 0.1], [0.9, 0.2], [0.8, 0.3]]
    reward_scale = [1.0, 1.2, 0.8]
    
    env_name_list = []
    for e, rew in product(eta, reward_scale):
        flow_params["env"].additional_params["eta1"] = e[0]
        flow_params["env"].additional_params["eta2"] = e[1]
        flow_params["env"].additional_params["reward_scale"] = rew

        # get the env name and a creator for the environment
        create_env, env_name = make_create_env(params=flow_params, version=0)
        env_name = env_name + '_[eta1, eta2]:[{}, {}]'.format(e[0], e[1]) + '_scale:{}'.format(rew)
        env_name_list.append(env_name)
        # Register as rllib env
        register_env(env_name, create_env)

    exp_tag_list = []
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
                "training_iteration": 200
            },
            "num_samples": 1,
        }
        exp_tag_list.append(exp_tag)

    trials = run_experiments(
        experiments={args.exp_tag: exp_tag for exp_tag in exp_tag_list},
    )
