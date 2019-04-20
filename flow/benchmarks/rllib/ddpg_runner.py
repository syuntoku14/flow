"""Runs the environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the DDPG algorithm in rllib
and utilizes the hyper-parameters specified in:
Proximal Policy Optimization Algorithms by Schulman et. al.
"""
import json
import argparse
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
    episode.user_data["episode_rew_mean"] = []
    
def on_episode_step(info):
    episode = info["episode"]
    rewds = list(episode.agent_rewards.values())
    rew = 0
    if not len(rewds) == 0:
        rew = np.mean(rewds)
    episode.user_data["episode_rew_mean"].append(rew)

def on_episode_end(info):
    episode = info["episode"]
    total_reward = np.sum(episode.user_data["episode_rew_mean"])
    episode.custom_metrics["episode_total_rew"] = total_reward

def on_train_result(info):
    info["result"]["callback_ok"] = True

if __name__ == "__main__":
    benchmark_name = 'grid0'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    alg_run = "DDPG"

    horizon = flow_params["env"].horizon
    agent_cls = get_agent_class(alg_run)
    # use almost defalt config
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(num_cpus, num_rollouts)
    config["train_batch_size"] = horizon * num_rollouts
    config["horizon"] = horizon
    config['timesteps_per_iteration'] = horizon
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["observation_filter"] = "NoFilter"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run
    config["callbacks"]["on_episode_start"] = ray.tune.function(on_episode_start)
    config["callbacks"]["on_episode_step"] = ray.tune.function(on_episode_step)
    config["callbacks"]["on_episode_end"] = ray.tune.function(on_episode_end)
    config["callbacks"]["on_train_result"] = ray.tune.function(on_train_result)

    # Register as rllib env
    register_env(env_name, create_env)

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

    trials = run_experiments({
        flow_params["exp_tag"]: exp_tag
    })
