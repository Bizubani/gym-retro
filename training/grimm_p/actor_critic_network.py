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
        chkpt_dir="./grimm_p/temp/ppo",
    ):
        c, h, w = input_dims

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
            nn.Linear(3136, fc1_dims),  # Adjust fc1_dims based on your needs
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),  # Adjust fc2_dims based on your needs
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
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model from file
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        game,
        input_dims,
        alpha,
        fc1_dims=512,
        fc2_dims=512,
        chkpt_dir="./grimm_p/temp/ppo",
    ):
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
            nn.Linear(3136, fc1_dims),  # Adjust fc1_dims based on your needs
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),  # Adjust fc2_dims based on your needs
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
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    # Load the model from file
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# class ActorNetwork(nn.Module):
#     def __init__(
#         self,
#         n_actions,
#         input_dims,
#         alpha,
#         input_channels=4,
#         fc1_dims=256,
#         fc2_dims=256,
#         chkpt_dir="temp/ppo",
#     ):
#         super(ActorNetwork, self).__init__()
#         self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo")
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#         # Calculate the input dimensions for the fully connected layers
#         fc_input_dims = self.calculate_fc_input_dims(input_channels, 256, 224)
#         print(f"fc_input_dims, {fc_input_dims} ")
#         self.actor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(3136, 512),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, n_actions),
#             nn.Softmax(dim=-1),
#         )

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
#         self.to(self.device)

#     def forward(self, state):
#         features = self.features(state)
#         dist = self.actor(features)
#         dist = Categorical(dist)
#         return dist

#     # Save the model to file
#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     # Load the model from file
#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))

#     def calculate_fc_input_dims(self, input_channels, height, width):
#         # Calculate the output size after the convolutional layers
#         x = T.zeros(1, input_channels, height, width)
#         x = self.features(x)
#         return x.view(1, -1).size(1)


# class CriticNetwork(nn.Module):
#     def __init__(
#         self,
#         input_dims,
#         alpha,
#         input_channels=4,
#         fc1_dims=256,
#         fc2_dims=512,
#         chkpt_dir="temp/ppo",
#     ):
#         super(CriticNetwork, self).__init__()
#         self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#         # Calculate the input dimensions for the fully connected layers
#         fc_input_dims = self.calculate_fc_input_dims(input_channels, 256, 224)
#         self.critic = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(3136, fc2_dims),
#             nn.ReLU(),
#             nn.Linear(fc2_dims, 1),
#         )
#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         print(f"\x1B[1mUsing cuda. Is it available? {T.cuda.is_available}\x1B[0m")
#         self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
#         self.to(self.device)

#     def forward(self, state):
#         features = self.features(state)
#         value = self.critic(features)
#         return value

#     def calculate_fc_input_dims(self, input_channels, height, width):
#         # Calculate the output size after the convolutional layers
#         x = T.zeros(1, input_channels, height, width)
#         x = self.features(x)
#         return x.view(1, -1).size(1)

#     # Save the model to file
#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     # Load the model from file
#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))
