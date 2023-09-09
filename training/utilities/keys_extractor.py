"""
Class to extract the button presses from a .bk2 file

"""
import retro

class ButtonExtractor:
    def __init__(self, filepath, game_actions, frameskip=4):
        self._filepath = filepath
        self._game_actions = game_actions
        self._frameskip = frameskip

    def map_to_discrete(self, actions, game_actions, system_buttons):
        """
        Map a list of actions to a discrete action space
        """
        # select the buttons for the system
        discrete_actions = []
        for x in range(0, len(actions), self._frameskip):
            button_combo = []
            store_button = None
            action = actions[x]
            for i in range(len(action)):
                if action[i] is True:
                    action_taken = system_buttons[i]
                    if action_taken == 'B' or action_taken == 'Y':
                        store_button = action_taken
                        continue
                    button_combo.append(action_taken)
            if store_button is not None:
                button_combo.append(store_button)
                store_button = None
            discrete_actions.append(game_actions.index(button_combo))
        return discrete_actions

    def get_actions_from_movie(self):
        movie = retro.Movie(self._filepath)
        movie.step()

        env = retro.make(
            game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        )
        env.initial_state = movie.get_state()
        env.reset()
            
        acts = []
        while movie.step():
            act = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    act.append(movie.get_key(i, p))
            acts.append(act)
        buttons = env.unwrapped.buttons
        env.close()
        print(buttons)
        return self.map_to_discrete(acts, self._game_actions, buttons)
    




