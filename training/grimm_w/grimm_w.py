"""
An implementation of the PPO algorithm
"""
import torch as T
import grimm_w.actor_critic_network as ACN
import numpy as np
import grimm_w.replay_buffer as PPO
import gym
import retro
import time
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as TV
import torch.nn.utils as nn_utils

from utilities.logger import Logger


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            if gym.__version__ < "0.26":
                obs, reward, done, info = self.env.step(action)
            else:
                obs, reward, done, _, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        if gym.__version__ < "0.26":
            return obs, total_reward, done, info
        else:
            return obs, total_reward, done, _, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = T.tensor(observation.copy(), dtype=T.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = TV.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = TV.Compose(
            [TV.Resize(self.shape, antialias=True), TV.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class Grimm_Wilhelm:
    # class constructor. Receives the environment and the max number of steps
    # optionally receives a list of actions to loaded from a file
    def __init__(
        self,
        game,
        n_actions,
        input_dims,
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        batch_size=64,
        N=2048,
        n_epochs=10,
        alpha=0.00001,
        load_model=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.N = N
        self.alpha = alpha
        self.n_actions = n_actions
        self.actor_loss = 0
        self.critic_loss = 0
        self.total_loss = 0
        self.avg_actor_loss = 0
        self.avg_critic_loss = 0
        self.avg_total_loss = 0
        self.top_scores = [
            -np.inf,
            -np.inf,
            -np.inf,
        ]

        self.actor = ACN.ActorNetwork(
            game=game,
            n_actions=n_actions,
            input_dims=input_dims,
            alpha=alpha,
        )
        self.critic = ACN.CriticNetwork(
            game=game,
            input_dims=input_dims,
            alpha=alpha,
        )
        if load_model is not None:
            # load the model from file

            self.load_models(load_model)

        self.memory = PPO.PPOMemory(batch_size)
        self.learn_step_counter = 0

    """
    Store the memory of the agent
    """

    def remember(self, state, action, logprob, value, reward, done):
        self.memory.store(state, action, logprob, value, reward, done)

    """
    Store the model to file
    """

    def save_models(self, final=False):
        self.actor.save_checkpoint(final=final)
        self.critic.save_checkpoint(final=final)

    """
    Load the model from file
    """

    def load_models(self, model_to_load):
        self.actor.load_checkpoint(model_to_load)
        self.critic.load_checkpoint(model_to_load)

    """
    Choose an action to take
    """

    def choose_action(self, observation):
        observation = (
            observation[0].__array__()
            if isinstance(observation, tuple)
            else observation.__array__()
        )
        state = T.tensor(observation, device=self.actor.device).unsqueeze(0)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    """
1.  Loop through the memory for a specified number of epochs (self.n_epochs).
2.  For each epoch, fetch the batches of experiences from the memory.
3.  Calculate the advantages using GAE (Generalized Advantage Estimation) for each time step in the batch.
4.  Normalize the advantages and convert them, along with values, to tensors.
5.  Iterates through the batches, and calculate the actor and critic losses, and then perform gradient clipping.
6.  Accumulated the losses for logging purposes.
7.  Update the actor and critic networks.
8.  Finally, after completing all epochs, clear the memory, and average the losses.
    """

    def learn(self):
        # loop through the memory N times
        # fetching the batches and learning from them
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_logprobs_arr,
                values_arr,
                reward_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()
            # assign the last value to the last state
            values = values_arr
            # and zero out the advantage
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # calculate the Generalized Advantage Estimation for each
            # time step in this batch
            for t in range(len(reward_arr) - 1):
                discount = 1
                advantage_at_time = 0
                for k in range(t, len(reward_arr) - 1):
                    advantage_at_time += discount * (
                        reward_arr[k]
                        + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                        - values[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = advantage_at_time

            # Now, normalize the calculated advantages
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8)
            # convert all the data to tensors
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_logprobs = T.tensor(old_logprobs_arr[batch], dtype=T.float).to(
                    self.actor.device
                )
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_logprobs = dist.log_prob(actions)
                prob_ratio = new_logprobs.exp() / old_logprobs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = (
                    T.clamp(prob_ratio, 1 - self.clip, 1 + self.clip) * advantage[batch]
                )

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]

                # Normalize returns
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                # Perform gradient clipping for actor and critic networks
                nn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                nn_utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

                # keep the loss values for logging
                self.actor_loss += actor_loss.item()
                self.critic_loss += critic_loss.item()
                self.total_loss += total_loss.item()

                # Udpate the actor and critic networks
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # clear the memory
        self.memory.clear_memory()
        # calculate the average loss for this episode
        self.avg_actor_loss = self.actor_loss / self.n_epochs
        self.avg_critic_loss = self.critic_loss / self.n_epochs
        self.avg_total_loss = self.total_loss / self.n_epochs
        # reset the loss values
        self.actor_loss = 0
        self.critic_loss = 0
        self.total_loss = 0

    def update_top_rewards(self, reward):
        if reward > min(self.top_scores) and reward not in self.top_scores:
            self.top_scores[self.top_scores.index(min(self.top_scores))] = reward
            print(f"\x1B[35m\x1B[1m\x1B[3mNew top rewards: {self.top_scores}\x1B[0m")


def grimm_runner(
    game,
    n_games=1000,
    state=retro.State.DEFAULT,
    scenario=None,
    discretizer=None,
    record_path=None,
    max_episode_steps=4500,
    model_to_load=None,
    use_custom_integrations=False,
    play_only=False,
    tag=None,
):
    print(
        f"\x1B[34m\x1B[3mRunning Wilhelm with game {game} and playing {n_games} times with max steps {max_episode_steps} per game"
        f"\nIt {'is' if use_custom_integrations else 'is not'} using custom integrations\x1B[0m"
    )
    if model_to_load is not None:
        print(f"\x1B[3\x1B[33mLoading model from {model_to_load}\x1B[0m")
    if use_custom_integrations:
        integration_type = retro.data.Integrations.ALL
    else:
        integration_type = retro.data.Integrations.DEFAULT
    env = retro.make(
        game,
        state,
        scenario=scenario,
        inttype=integration_type,
    )
    env = discretizer(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    # hyperparameters seleccted using stable baselines implementation settings
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    N = 2048
    n_games = n_games
    game_name = game + str(int(time.time()))
    if tag is not None:
        game_name += "_" + tag
    wilhelm = Grimm_Wilhelm(
        game=game_name,
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        batch_size=batch_size,
        N=N,
        n_epochs=n_epochs,
        alpha=alpha,
        load_model=model_to_load,
    )
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    actions = []

    log_headers = [
        "episode",
        "score",
        "avg_score",
        "avg_actor_loss",
        "avg_critic_loss",
        "avg_total_loss",
        "steps  taken",
    ]
    logger = Logger(
        log_dir="./grimm_w/temp/logs/",
        log_name=f"{game_name}.csv",
        log_headers=log_headers,
    )

    timesteps = 0
    best_rew = float("-inf")
    for i in range(n_games):
        observation = env.reset()
        if gym.__version__ > "0.26":
            # discard the map from the observation tuple
            observation = observation[0]
        done = False
        score = 0
        while not done and timesteps < max_episode_steps:
            # set the channel as last dimension
            action, logprob, value = wilhelm.choose_action(observation=observation)
            actions.append(action)
            n_steps += 1
            timesteps += 1
            if gym.__version__ < "0.26":
                observation_, reward, done, info = env.step(action)
            else:
                observation_, reward, done, _, info = env.step(action)
            # only try to render if the gym version is 0.26 or above (uses stable-retro)
            if gym.__version__ > "0.26":
                env.render()
            score += reward
            wilhelm.remember(observation, action, logprob, value, reward, done)
            # only run this code if we are teaching the agent
            if n_steps % wilhelm.N == 0 and not play_only:
                wilhelm.learn()
                learn_iters += 1
                print(
                    f"\x1B[3m\x1B[32mWe are on the {n_steps}th step and have completed {i} episodes.\nSebastain has been trained {learn_iters} times\x1B[0m"
                    f"\n\x1B[3m\x1B[32mThe average actor loss is {wilhelm.avg_actor_loss} and the average critic loss is {wilhelm.avg_critic_loss}\x1B[0m"
                    f"\n\x1B[3m\x1B[32mThe average total loss is {wilhelm.avg_total_loss}\x1B[0m"
                )
            observation = observation_
        score_history.append(score)
        wilhelm.update_top_rewards(score)
        to_log = {
            "episode": i,
            "score": score,
            "avg_score": avg_score,
            "avg_actor_loss": wilhelm.avg_actor_loss,
            "avg_critic_loss": wilhelm.avg_critic_loss,
            "avg_total_loss": wilhelm.avg_total_loss,
            "steps  taken": n_steps,
        }

        if score > best_rew:
            print(
                f"\x1B[3m\x1B[34mAccomplished new best score! {best_rew} => {score}\x1B[0m"
            )
            best_rew = score
            env.unwrapped.record_movie(f"{record_path}/best_{best_rew}_{tag}.bk2")
            env.reset()
            for act in actions:
                env.step(act)
            env.unwrapped.stop_record()

        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            if not play_only:
                print("Saving better models")
                wilhelm.save_models()
            best_score = avg_score
        actions.clear()
        logger.log(to_log)
        print(
            f"\x1B[3mepisode {i} score {score:.1f} average score {avg_score:.1f} best score {best_score:.1f} "
            f"learning_steps {learn_iters}, n_steps {n_steps} and timesteps {timesteps}\x1B[0m"
        )
        # reset the timestep counter
        timesteps = 0

    # save the final models
    if not play_only:
        wilhelm.save_models(final=True)
    print("\x1B[1mtimestep limit exceeded\x1B[0m")
    print("\x1B[3m\x1B[1mHere are the stats of the execution:\x1B[0m")
    print(f"\x1B[34m\x1B[3mBest reward: {best_rew}")
    print(f"Average reward: {avg_score}")
    print(f"Total episodes: {n_games}")
    print(f"Total timesteps: {n_steps}")
    print(f"Final epsiode actor loss: {wilhelm.avg_actor_loss}")
    print(f"Final epsiode critic loss: {wilhelm.avg_critic_loss}")
    print(f"Final epsiode total loss: {wilhelm.avg_total_loss}")
    print(f"\x1B[34m\x1B[3mTop scores: {wilhelm.top_scores}\x1B[0m")

    print("\x1B[3m\x1B[1mWriting logs to file!\x1B[0m")
    logger.close()
    print("\x1B[3m\x1B[1mDone!\x1B[0m")
