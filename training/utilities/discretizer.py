import gym
import numpy as np

gradius_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['UP', 'B'], ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['A']]
battletoad_and_double_dragon = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['Y'], ['B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B', 'Y'], ['LEFT', 'Y'], ['RIGHT', 'Y']]
rtype_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A', 'B'], ['Y'], ['X'], ['LEFT', 'X'], ['RIGHT', 'X'], ['UP', 'X'], ['DOWN', 'X']]
draculax_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['Y'], ['B'], ['X'], ['LEFT', 'B'], ['RIGHT', 'B'], ['UP', 'Y'], ['LEFT', 'Y'], ['RIGHT', 'Y'], ['DOWN', 'Y']]

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        print(f'\x1B[32mAction space size is {env.action_space.n}\x1B[0m')
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()
    

class GradiusDiscretizer(Discretizer):
    """
    Use Gradius-specific discrete actions
    based on
     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=gradius_actions)

class BattletoadsDiscretizer(Discretizer):
    """
    Use Battletoads and Double Dragon-specific discrete actions
    based on
     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=battletoad_and_double_dragon)

class RTypeDiscretizer(Discretizer):
    """
    Use Battletoads and Double Dragon-specific discrete actions
    based on
     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=rtype_actions)

class DraculaXDiscretizer(Discretizer):
    """
    Use Battletoads and Double Dragon-specific discrete actions
    based on
     based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=draculax_actions)
   