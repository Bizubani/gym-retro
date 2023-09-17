"""
Inspired by the JERK algorithm described in the paper "Gotta Learn Fast: 
A New Benchmark for Generalization in RL" by Nichol et al.
https://arxiv.org/abs/1804.03720
"""


import random
import numpy as np
import math
import retro
import gym

castlevania4_actions = []
penalty_scale = None

# The exploration parameter used to select random actions
EXPLORATION_PARAM = 0.0025

"""
The agents that we will use to play the game will select actions
and then each action will be executed for a number of frames.
"""


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
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


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        if gym.__version__ < "0.26":
            observation, reward, done, info = self.env.step(action)
        else:
            observation, reward, done, _, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        if gym.__version__ < "0.26":
            return observation, reward, done, info
        else:
            return observation, reward, done, _, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


"""
A node in the tree of action states. It holds the best reward that has been
found with this node in the path, as well as the immediate reward of this action.
It also holds the number of times that this node has been visited. 
"""


class Node:
    def __init__(
        self, max_reward=-np.inf, immediate_reward=np.inf, children=None, visits=0
    ):
        self.max_reward = max_reward
        self.immediate_reward = immediate_reward
        self.visits = visits
        self.children = {} if children is None else children

    def __repr__(self):
        return (
            "<Node Max Reward of this path=%f\nImmediate Reward of this action=%f\nVisits=%d len(children)=%d>"
            % (
                self.max_reward,
                self.immediate_reward,
                self.visits,
                len(self.children),
            )
        )


# move to gradius only code section
def return_action(action_space):
    restriction = 0.05
    nextAction = action_space.sample()
    # restrict the usage of the button A
    if nextAction == 9:
        if random.random() > restriction:
            nextAction = action_space.sample()
    return nextAction


"""
Walk the graph of actions and choose the next action to take.

Generally we will lean heavily towards the greedy action and 
tendancy increases as the number of visits increases.  However,
there is a small chance that we will select a random action.

If we are at a leaf node, we will always select a random action.

Takes the root node of the tree, the valid list of actions 
and the maximum number of steps to take.
"""


def select_actions(
    root,
    action_space,
    max_episode_steps,
    longest_path,
    top_rewards,
):
    # start at the root node
    node = root
    actions = []
    steps = 0
    # while we still have steps to take
    while steps < max_episode_steps:
        if node is None:
            # we've fallen off the explored area of the tree, just select random actions
            next_action = return_action(action_space)
        else:
            # let's look ahead to see if there is a negative reward coming up
            # if there is, then we will give it a random chance to explore another path
            epsilon_boost = 0
            # however, only look ahead if we are on a path with a good chance of
            # getting a high reward
            if node.max_reward in top_rewards:
                upcoming_reward, distance_ahead = look_ahead(node, node.max_reward)
                if upcoming_reward < 0:
                    # bump up the exploration parameter
                    epsilon_boost = abs(upcoming_reward) / penalty_scale
                    # scale it by how deep in the tree we are, but only up to the
                    # longest path we've seen so far
                    if steps < longest_path:
                        epsilon_boost = epsilon_boost * steps / longest_path

            # calculate the epsilon value for this node
            epsilon = EXPLORATION_PARAM / np.log(node.visits + 2) + epsilon_boost
            # roll the dice to see if we take a random action
            if random.random() < epsilon:
                if epsilon_boost > 0:
                    print(
                        f"\x1B[36mTaking random action with epsilon boost {epsilon_boost}. There have been {node.visits} visits and max reward on this path is: {node.max_reward}. \nUpcoming penalty is {upcoming_reward} and is {distance_ahead} steps away. \x1B[0m"
                    )
                else:
                    print(
                        f"\x1B[32mTaking random action. There have been {node.visits} visits to this node\x1B[0m"
                    )
                # select an action at random
                next_action = return_action(action_space)
            else:
                # We are going to tend towards the greedy action
                action_reward = {}
                for action in range(action_space.n):
                    if node is not None and action in node.children:
                        action_reward[action] = node.children[action].max_reward
                    else:
                        action_reward[action] = -np.inf
                best_reward = max(action_reward.values())
                best_actions = [
                    action
                    for action, reward in action_reward.items()
                    if reward == best_reward
                ]
                next_action = random.choice(best_actions)

            if next_action in node.children:
                node = node.children[next_action]
            else:
                node = None

        actions.append(next_action)
        steps += 1

    return actions


"""
perform the requested set of actions and find the total reward
"""


def perform_actions(env, actions):
    total_reward = 0
    env.reset()
    steps = 0
    rewards = []
    for action in actions:
        if gym.__version__ < "0.26":
            observation_, reward, done, info = env.step(action)
        else:
            observation_, reward, done, _, info = env.step(action)
        steps += 1
        total_reward += reward
        rewards.append(reward)
        if done:
            break

    return steps, rewards, total_reward


"""
The agent has performed a set of actions and received it's reward.
Use this information to update the recorded paths in the tree.

Takes the root node of the tree, the list of actions that were executed,
the rewards at each step, the total reward and a flag indicating if this
is a loaded set of actions.
"""


def update_path(root, executed_actions, rewards, total_reward, loaded=False):
    # if this is a loaded set of actions, then we need to update the visit count
    # of the root node
    if loaded:
        visited = len(executed_actions)
        root.visits = visited
    root.max_reward = max(total_reward, root.max_reward)
    root.visits += 1
    new_nodes = 0

    # walk the tree from the root updating the nodes as we go
    node = root
    # for each step in the list of executed actions
    for step, act in enumerate(executed_actions):
        # grab the immediate reward for this action
        immediate_reward = rewards[step]
        # if we haven't seen this action before, then create a new node
        if act not in node.children:
            if loaded:
                # if we loaded this file in, then set up the child node with a
                # scaled visit count, getting smaller as we approach the leaf nodes
                scaled_visits = math.floor(visited // np.exp(np.log(step + 2)))
                node.immediate_reward = immediate_reward
                node.children[act] = Node(visits=scaled_visits)
            else:
                node.children[act] = Node()
            new_nodes += 1
        node = node.children[act]
        # update the immediate reward and the max reward for this node
        node.immediate_reward = immediate_reward
        node.max_reward = max(total_reward, node.max_reward)
        node.visits += 1

    return new_nodes


"""
Allow the agent to look ahead along a path it has already explored.
If there is a negative reward coming up in the next few steps, then
we will give it a random chance to explore another path.
"""


def look_ahead(start_node, target_reward):
    # look ahead up to 8 steps (this will be 30 frames or ~1 second)
    # If we find a negative reward in the next 8 steps, then we will
    # give it a random chance to explore another path
    # and then check it's immediate reward
    next_node = start_node
    look_ahead_steps = 8
    found_at = None
    return_value = np.inf
    for level in range(look_ahead_steps):
        # we haven't found the target reward yet
        target_node = None
        # look through the children of this node
        for node in next_node.children.values():
            # if we find a node with the target reward, update the target node
            if node.max_reward == target_reward:
                # this could be the path we are going to take
                if target_node is None:
                    # if target_node is None, then this is the first path we've found
                    # with the target reward, so update the target node
                    target_node = node
                else:
                    # we've found another path with the same reward
                    # there will be too much uncertainty, so just bail
                    target_node = None
                    break
        # if we found no target node in the children, we are at the leaf nodes
        # just return the default value
        if target_node is None:
            # no need to continue looking
            break
        elif target_node.immediate_reward < 0:
            found_at = level
            # we have found a negative reward in the next few steps
            # scale it by how far away it is
            return_value = target_node.immediate_reward * (
                (look_ahead_steps - level) / look_ahead_steps
            )
            break
        else:
            # we found a target node, but it's immediate reward is positive
            # keep looking
            next_node = target_node
    # we have made it to the end of the look ahead steps
    # now check the immediate reward of the target node
    return return_value, found_at


"""
Implementation of the Jerk algorithm with look ahead

Creates and manages the tree storing game actions and rewards
"""


class Grimm_Jocob:
    # class constructor. Receives the environment and the max number of steps
    # optionally receives a list of actions to loaded from a file
    def __init__(self, env, max_episode_steps, actions=None):
        self.node_count = 1
        self._root = Node()
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._actions = actions
        self.longest_path = 0
        self.top_scores = [
            -np.inf,
            -np.inf,
            -np.inf,
        ]
        if self._actions is not None:
            self.load_actions()

    def load_actions(self):
        print("\x1B[34mLoading from file...\x1B[0m")
        actions = self._actions
        steps, rewards, total_reward = perform_actions(self._env, actions)
        executed_actions = actions[:steps]
        self.longest_path = steps
        print("\x1B[1mLoaded actions, now updating path...\x1B[0m")
        self.update_top_rewards(total_reward)
        self.node_count += update_path(
            self._root,
            executed_actions,
            rewards,
            total_reward,
            loaded=True,
        )
        print(
            f"\x1B[34mInitial info: nodes: {self.node_count}, reward from saved input: {total_reward}!\x1B[0m"
        )

    def run(self):
        actions = select_actions(
            self._root,
            self._env.action_space,
            self._max_episode_steps,
            self.longest_path,
            top_rewards=self.top_scores,
        )
        steps, rewards, total_reward = perform_actions(self._env, actions)
        if steps > self.longest_path:
            self.longest_path = steps
            print(
                f"\x1B[32m\x1B[1m\x1B[3mNew longest path: {self.longest_path}!\x1B[0m"
            )
        executed_actions = actions[:steps]
        self.update_top_rewards(total_reward)
        self.node_count += update_path(
            self._root,
            executed_actions,
            rewards,
            total_reward,
        )
        return executed_actions, total_reward

    def update_top_rewards(self, reward):
        if reward > min(self.top_scores) and reward not in self.top_scores:
            self.top_scores[self.top_scores.index(min(self.top_scores))] = reward
            print(f"\x1B[35m\x1B[1m\x1B[3mNew top rewards: {self.top_scores}\x1B[0m")


def grimm_runner(
    game,
    max_episode_steps=9000,
    loaded_actions=None,
    counter=0,
    timestep_limit=2e7,
    state=retro.State.DEFAULT,
    scenario=None,
    discretizer=None,
    record_path=None,
    penalty_scale_arg=500,
):
    global penalty_scale
    penalty_scale = penalty_scale_arg
    print(
        f"\x1B[34m\x1B[3mRunning grimm_runner with game {game} and max_episode_steps {max_episode_steps}\x1B[0m"
    )
    env = retro.make(
        game, state, scenario=scenario, inttype=retro.data.Integrations.ALL
    )
    env = discretizer(env)
    env = SkipFrame(env)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    johan = Grimm_Jocob(
        env, max_episode_steps=max_episode_steps, actions=loaded_actions
    )
    timesteps = 0
    best_rew = float("-inf")
    while True:
        actions, rew = johan.run()
        counter += 1

        timesteps += len(actions)
        print(
            "info: counter={}, timesteps={}, nodes={}, reward={}".format(
                counter, timesteps, johan.node_count, rew
            )
        )

        if rew > best_rew:
            print("\x1B[35mnew best reward {} => {}!\x1B[0m".format(best_rew, rew))

            best_rew = rew

            env.unwrapped.record_movie(f"{record_path}/best_{rew}_at_{timesteps}.bk2")
            env.reset()
            for act in actions:
                env.step(act)
            env.unwrapped.stop_record()

        if timesteps > timestep_limit:
            print("timestep limit exceeded")
            print("\x1B[3m\x1B[1mHere are the stats of the execution:\x1B[0m")
            print(f"\x1B[34m\x1B[3mBest reward: {best_rew}")
            print(f"Total timesteps: {timesteps}")
            print(f"Longest path: {johan.longest_path} steps")
            print(f"Total nodes: {johan.node_count}")
            print(f"\x1B[34m\x1B[3mTop scores: {johan.top_scores}\x1B[0m")
            break
