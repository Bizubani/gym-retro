"""
An implementation of the PPO algorithm
"""
import torch as T
import grimm_p.actor_critic_network as ACN
import numpy as np
import grimm_p.replay_buffer as PPO
import gym
import retro
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as TV

"""
Implementation of the Jerk algorithm with look ahead

Creates and manages the tree storing game actions and rewards
"""


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
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


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
        transforms = TV.Compose([TV.Resize(self.shape), TV.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


class Grimm_Sebastian:
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
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip = clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.N = N
        self.alpha = alpha
        self.n_actions = n_actions
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

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    """
    Load the model from file
    """

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

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

        # _, _, action_probs = self.actor.forward(state)
        # dist = Categorical(action_probs)
        # action = dist.sample()
        # logprob = dist.log_prob(action)
        # value = self.critic.forward(state)
        # return action.item(), logprob, value
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    """
    Learn from the memory
    """

    def learn(self):
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

            values = values_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
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

                actor_loss = (
                    -T.min(weighted_probs, weighted_clipped_probs).mean()
                    # not in the implementation I followed. Do we need this?
                    # - 0.001 * dist.entropy().mean()
                )

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                # print learning stats
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()


def grimm_runner(
    game,
    n_games=1000,
    state=retro.State.DEFAULT,
    scenario=None,
    discretizer=None,
    record_path=None,
):
    print(
        f"\x1B[34m\x1B[3mRunning Sebastian with game {game} and playing {n_games} times\x1B[0m"
    )
    env = retro.make(game, state, scenario=scenario)
    env = discretizer(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    # hyperparameters seleccted using stable baselines implementation settings
    batch_size = 64
    n_epochs = 10
    alpha = 0.0001
    N = 2048
    n_games = n_games
    sebastian = Grimm_Sebastian(
        game=game,
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        batch_size=batch_size,
        N=N,
        n_epochs=n_epochs,
        alpha=alpha,
    )
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    actions = []

    timesteps = 0
    best_rew = float("-inf")
    for i in range(n_games):
        observation = env.reset()
        done = False
        timesteps += 1
        score = 0
        while not done:
            # set the channel as last dimension
            action, logprob, value = sebastian.choose_action(observation=observation)
            actions.append(action)
            n_steps += 1
            observation_, reward, done, info = env.step(action)
            score += reward
            sebastian.remember(observation, action, logprob, value, reward, done)
            if n_steps % sebastian.N == 0:
                sebastian.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        if score > best_rew:
            print(
                f"\x1B[3m\x1B[34mAccomplished new best score! {best_rew} => {score}\x1B[0m"
            )
            best_rew = score
            env.unwrapped.record_movie(f"{record_path}/best_{best_rew}.bk2")
            env.reset()
            for act in actions:
                env.step(act)
            env.unwrapped.stop_record()

        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            print("Saving better models")
            sebastian.save_models()
            best_score = avg_score
        actions.clear()
        print(
            f"\x1B[3mepisode {i} score {score:.1f} average score {avg_score:.1f} best score {best_score:.1f} "
            f"learning_steps {learn_iters}, n_steps {n_steps} and timesteps {timesteps}\x1B[0m"
        )
    # while True:
    #     actions, rew = sebastian.run()
    #     counter += 1

    #     timesteps += len(actions)
    #     print(
    #         "info: counter={}, timesteps={}, nodes={}, reward={}".format(
    #             counter, timesteps, johan.node_count, rew
    #         )
    #     )

    #     if rew > best_rew:
    #         print("\x1B[35mnew best reward {} => {}!\x1B[0m".format(best_rew, rew))

    #         best_rew = rew

    #         env.unwrapped.record_movie(f"{record_path}/best_{rew}_at_{timesteps}.bk2")
    #         env.reset()
    #         for act in actions:
    #             env.step(act)
    #         env.unwrapped.stop_record()

    #     if timesteps > timestep_limit:
    #         print("timestep limit exceeded")
    #         break
