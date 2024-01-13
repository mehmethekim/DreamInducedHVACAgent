from utils import *
from RNN_PPO import *
from RBC_SB3 import *
ENVIRONMENT_NAME = 'Eplus-5zone-hot-continuous-stochastic-v1'
EPISODES = 10
ALGORITHM_NAME = 'RNN_PPO'


def main():
    experiment_name, run = setup_experiment(ENVIRONMENT_NAME, EPISODES, ALGORITHM_NAME)
    env, eval_env = create_environments(ENVIRONMENT_NAME,experiment_name)
    #Define model below
    #model = PPO('MlpPolicy', env, verbose=1)
    model = PPO(CustomActorCriticPolicy, env, verbose=1)
    callbacks = setup_callbacks(eval_env,model)
    timesteps = EPISODES * (env.get_wrapper_attr('timestep_per_episode') - 1)
    train_model(model, timesteps, callbacks)
    save_and_log_results(model, env, eval_env, experiment_name, run)
    run.finish()

if __name__ == "__main__":
    main()