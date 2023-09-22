import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class ActorNetwork(nn.Module):
    def __init__(
        self,
        game,
        n_actions,
        input_dims,
        alpha,
        fc1_dims=512,
        fc2_dims=512,
        chkpt_dir="./grimm_w/temp/ppo",
    ):
        c, h, w = input_dims
        self.chkpt_dir = chkpt_dir
        self.game = game
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, f"actor_{game}_ppo")
        self.actor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = self.actor(state)
        dist = Categorical(state)
        return dist

    # Save the model to file
    def save_checkpoint(self, final=False):
        if final:
            self.checkpoint_file = os.path.join(
                self.chkpt_dir, f"actor_{self.game}_ppo_final"
            )
        print(f"Saving checkpoint to {self.checkpoint_file}")
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model from file
    def load_checkpoint(self, model_to_load=None):
        if model_to_load is not None:
            model_to_load = model_to_load.replace("critic", "actor")
            self.game = model_to_load
            self.checkpoint_file = os.path.join(self.chkpt_dir, model_to_load)
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        game,
        input_dims,
        alpha,
        fc1_dims=512,
        fc2_dims=512,
        chkpt_dir="./grimm_w/temp/ppo",
    ):
        self.chkpt_dir = chkpt_dir
        self.game = game
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, f"critic_{game}_ppo")

        c, h, w = input_dims

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")
        self.critic = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state = self.critic(state)
        return state

    # Save the model to file
    def save_checkpoint(self, final=False):
        if final:
            self.checkpoint_file = os.path.join(
                self.chkpt_dir, f"critic_{self.game}_ppo_final"
            )
        print(f"Saving checkpoint to {self.checkpoint_file}")
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model from file
    def load_checkpoint(self, model_to_load=None):
        if model_to_load is not None:
            model_to_load = model_to_load.replace("actor", "critic")
            self.game = model_to_load
            self.checkpoint_file = os.path.join(self.chkpt_dir, model_to_load)
        self.load_state_dict(T.load(self.checkpoint_file))
