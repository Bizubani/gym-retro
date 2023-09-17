import retro
import gym
from grimm_p import Grimm_Sebastian
import numpy as np
import cv2


def main():
    env = gym.make("CartPole-v1")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    sebastian = Grimm_Sebastian(
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        batch_size=batch_size,
        N=N,
        n_epochs=n_epochs,
        alpha=alpha,
    )
    n_games = 500
    filename = "cartpole.png"
    figure_file = "plots/" + filename
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, logprob, value = sebastian.choose_action(observation)
            n_steps += 1
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            sebastian.remember(observation, action, logprob, value, reward, done)
            if n_steps % sebastian.N == 0:
                sebastian.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            sebastian.save_models()
        print(
            f"episode {i} score {score:.1f} average score {avg_score:.1f} best score {best_score:.1f} "
            f"learning_steps {learn_iters}, n_steps {n_steps}"
        )

    # x = [i + 1 for i in range(n_games)]
    # plot_learning_curve(x, score_history, figure_file)


if __name__ == "__main__":
    main()

# def grimm_runner(
#     game,
#     max_episode_steps=9000,
#     loaded_actions=None,
#     counter=0,
#     timestep_limit=2e7,
#     state=retro.State.DEFAULT,
#     scenario=None,
#     discretizer=None,
#     record_path=None,
#     penalty_scale_arg=1000,
# ):
#     global penalty_scale
#     penalty_scale = penalty_scale_arg
#     print(
#         f"\x1B[34m\x1B[3mRunning grimm_runner with game {game} and max_episode_steps {max_episode_steps}\x1B[0m"
#     )
#     env = retro.make(game, state, scenario=scenario)
#     env = discretizer(env)
#     env = Frameskip(env)
#     env = TimeLimit(env, max_episode_steps=max_episode_steps)

#     johan = Grimm_Sebastian(
#         env, max_episode_steps=max_episode_steps, actions=loaded_actions
#     )
#     timesteps = 0
#     best_rew = float("-inf")
#     while True:
#         actions, rew = johan.run()
#         counter += 1

#         timesteps += len(actions)
#         print(
#             "info: counter={}, timesteps={}, nodes={}, reward={}".format(
#                 counter, timesteps, johan.node_count, rew
#             )
#         )

#         if rew > best_rew:
#             print("\x1B[35mnew best reward {} => {}!\x1B[0m".format(best_rew, rew))

#             best_rew = rew

#             env.unwrapped.record_movie(f"{record_path}/best_{rew}_at_{timesteps}.bk2")
#             env.reset()
#             for act in actions:
#                 env.step(act)
#             env.unwrapped.stop_record()

#         if timesteps > timestep_limit:
#             print("t
