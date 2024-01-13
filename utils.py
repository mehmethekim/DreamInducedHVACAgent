import os
import sys
from datetime import datetime
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat, Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
import sinergym
from sinergym.utils.callbacks import LoggerEvalCallback, LoggerCallback
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.wrappers import LoggerWrapper
import gymnasium as gym

def initialize_logging(model_name):
    models_dir = "models/"+ model_name
    logs_dir = "logs/"+ model_name
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir,models_dir


def setup_experiment(environment_name:str,episodes:int,algoritm_name:str):
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = f'{algoritm_name}-{environment_name}-episodes-{episodes}_{experiment_date}'
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version,
        'environment': environment_name,
        'episodes': episodes,
        'algorithm': algoritm_name
    }

    wandb_params = {"project": 'DreamInducedHVACAgent', "entity": 'mehmetbh'}

    run = wandb.init(name=experiment_name + '_' + wandb.util.generate_id(), config=experiment_params, **wandb_params)

    return experiment_name, run

def create_environments(environment_name:str,experiment_name:str):
    env = gym.make(environment_name, env_name=experiment_name)
    eval_env = gym.make(environment_name, env_name=experiment_name + '_EVALUATION')

    env = LoggerWrapper(env)
    eval_env = LoggerWrapper(eval_env)

    return env, eval_env



def setup_callbacks(eval_env,model,n_eval_episodes=1):
    callbacks = []
    eval_callback = LoggerEvalCallback(
        eval_env,
        best_model_save_path=eval_env.get_wrapper_attr('workspace_path') + '/best_model/',
        log_path=eval_env.get_wrapper_attr('workspace_path') + '/best_model/',
        eval_freq=(eval_env.get_wrapper_attr('timestep_per_episode') - 1) * 2 - 1,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes)
    callbacks.append(eval_callback)

    logger = SB3Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(sys.stdout, max_length=120),
            WandBOutputFormat()])
    model.set_logger(logger)
    log_callback = LoggerCallback()
    callbacks.append(log_callback)

    return CallbackList(callbacks)

def train_model(model, timesteps, callback):
    model.learn(total_timesteps=timesteps, callback=callback, log_interval=1)

def save_and_log_results(model, env, eval_env, experiment_name, run,artifact_name):
    model.save(str(env.get_wrapper_attr('timestep_per_episode')) + '/' + experiment_name)
    env.close()
    artifact = wandb.Artifact(name=artifact_name, type="training")
    artifact.add_dir(env.get_wrapper_attr('workspace_path'), name='training_output/')
    artifact.add_dir(eval_env.get_wrapper_attr('workspace_path'), name='evaluation_output/')
    run.log_artifact(artifact)
