"""Knight's Tour benchmark."""
import gymnasium

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qdglue.tasks.qd_task import QDTask

# todo (rboldi) figure out if this is the right place to put these classes


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, device):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, latent_dims)
        self.linear5 = nn.Linear(16, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)  # sample on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.reshape((-1, 64))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        mu = self.linear4(x)
        logvar = self.linear5(x)
        sigma = torch.exp(logvar * 0.5)

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 32)
        self.linear4 = nn.Linear(32, 64)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        z = torch.sigmoid(self.linear4(z))
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device="cuda:0"):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, device)
        self.fitness_layer = nn.Linear(latent_dims, 1)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = F.relu(self.encoder.forward(x))
        return self.decoder(z)

    def freeze_encoder(self):
        # this function is just for safety
        for name, para in self.encoder.named_parameters():
            para.requires_grad = False

    def forward_fitness(self, x):
        f = self.forward_fitness_deagg(x)
        return torch.clip(torch.sum(f, axis=1), 0, 64)

    def forward_fitness_deagg(self, x):
        z = self.encoder.forward(x)
        w = self.fitness_layer.weight
        return F.relu(z * w)

    def forward_features(self, x):
        z = self.encoder.forward(x)
        return z


class KnightsTour(QDTask):
    """Implementation of the Knight's Tour benchmark suite

    Args:
        parameter_space_dims (int): Dimensionality of each solution.
        method (str): The name of the method to determine the features. Supported options
            are: ["hand", "vae"]
    """

    def __init__(self, method):
        super().__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using device: {device}")
        self._parameter_space_dims = 64
        max_bound = 8
        self._measure_space_dims = [(0, 8), (0, 8)]

        self.directions = np.array(
            [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]]
        )
        self.grid_w = 8
        self.grid_h = 8

        if method not in ["vae", "hand"]:
            raise ValueError(f"Unsupported method `{method}`")

        self._method = method

        if method == "vae":
            # todo (rboldi) figure out how to do gpu stuff between jax and pytorch
            self.fitness_model = VariationalAutoencoder(8, device)
            self.fitness_model.load_state_dict(torch.load("qdglue/tasks/vae_model.pt", map_location=torch.device(device)))
            self.fitness_model = self.fitness_model.to(device)
            self.fitness_model.eval()
            self.fitness_model.freeze_encoder()
        else:
            self.fitness_model = None

    # computes end position and tiles covered
    def calculate(self, g):
        moves = np.concatenate(
            (np.array([[g[0], g[1]]]), self.directions[g[2:]]), axis=0
        )
        visited = np.cumsum(moves, axis=0)

        valid_visited = []
        # slow way for now, speed up later
        for i in visited:
            coord = tuple(i)
            if coord in valid_visited:
                break  # already seen
            elif (
                coord[0] >= self.grid_w
                or coord[0] < 0
                or coord[1] >= self.grid_h
                or coord[1] < 0
            ):
                break  # out of bounds
            else:
                valid_visited.append(coord)

        return visited, valid_visited, len(valid_visited)

    def evaluate(self, parameters, random_key=None):
        # parameters is a list of 66 integers, with a leading batch dimension
        # the first 64 are the moves to make (1-8).
        # the last two are the start position (x, y)
        # TODO(looka): should _parameter_space_dims be equal to 66 instead
        #  instead of 64 then?

        # todo (rboldi) figure out how to batch the calculate function
        # todo (rboldi) documentation like bryon

        if self._method == "hand":
            objective_batch = []
            measures_batch = []
            # run through calculate function, unbatched for now
            for i in range(parameters.shape[0]):
                # g is the ith genome
                g = parameters[i]
                visited, valid_visited, tiles_covered = self.calculate(g)
                objective = tiles_covered
                measures = valid_visited[-1]  # end pos

                objective_batch.append(objective)
                measures_batch.append(measures)

            # todo (rboldi) replace once decided on np vs jnp
            objective_batch = np.array(objective_batch)
            measures_batch = np.array(measures)
        elif self._method == "vae":
            # run batched through the nn
            measures_batch = self.fitness_model.forward_features(
                torch.Tensor(parameters)
            )
            objective_batch = np.array(
                [self.calculate(parameters[g])[2] for g in range(parameters.shape[0])]
            )

        return (
            objective_batch,
            None,  # no gradients here
            measures_batch,
            None,  # no gradients here
        )

    @property
    def parameter_space_dims(self):
        """Dimensions of the parameter space."""
        return self._parameter_space_dims

    @property
    def objective_space_dims(self):
        """Dimensions of the objective space."""
        return 1

    @property
    def descriptor_space_dims(self):
        """Dimensions of the descriptor space.

        Always equal to 2.
        """
        return len(self._measure_space_dims)

    @property
    def descriptor_space_bounds(self):
        """Bounds of the descriptor space."""
        return self._measure_space_dims

    @property
    def parameter_type(self):
        return "discrete"

    @property
    def parameter_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.MultiDiscrete(nvec=[8 for _ in range(self._parameter_space_dims)],)