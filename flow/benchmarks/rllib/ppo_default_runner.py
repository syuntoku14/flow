"""Runs the environments located in flow/benchmarks.
The environment file can be modified in the imports to change the environment
this runner script is executed on. This file runs the PPO algorithm in rllib
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
    "--upload_dir", type=str, help="S3 Bucket to upload to.")

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

parser.add_argument(
    "--exp_name", type=str, help="Name of the experiment.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=50,
    help="The number of rollouts to train over.")

# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=2,
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
    benchmark_name = 'grid0'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus

    upload_dir = args.upload_dir

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    alg_run = "PPO"

    horizon = flow_params["env"].horizon
    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(num_cpus, num_rollouts)
    config["train_batch_size"] = horizon * num_rollouts
    config["use_gae"] = True
    config["horizon"] = horizon
    gae_lambda = 0.97
    step_size = 5e-4
    if benchmark_name == "grid0":
        gae_lambda = 0.5
        step_size = 5e-5
    elif benchmark_name == "grid1":
        gae_lambda = 0.3
    config["lambda"] = gae_lambda
    config["lr"] = step_size
    config["vf_clip_param"] = 1e6
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run
    config['callbacks']['on_episode_start'] = ray.tune.function(on_episode_start)
    config['callbacks']['on_episode_step'] = ray.tune.function(on_episode_step)
    config['callbacks']['on_episode_end'] = ray.tune.function(on_episode_end)
    
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

    if upload_dir:
        exp_tag["upload_dir"] = "s3://" + upload_dir

    trials = run_experiments({
        args.exp_name: exp_tag
    })
