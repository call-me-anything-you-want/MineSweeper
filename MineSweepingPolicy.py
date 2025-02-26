from torch import nn
import torch
class MineSweepingPolicy(nn.Module):
    def __init__(self, mine_field_shape: torch.Tensor):
        super().__init__()
        '''
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 100)
        )
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            # nn.LeakyReLU(),
            nn.PReLU(),
            nn.Conv2d(8, 64, 3),
            # nn.LeakyReLU()
            nn.PReLU(),
            nn.Conv2d(64, 256, 3),
            # nn.LeakyReLU()
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3),
            # nn.LeakyReLU(),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 8, 3),
            # nn.LeakyReLU(),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 1, 3),
            # nn.LeakyReLU()
        )
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 4),
            nn.PReLU(),
            nn.Conv2d(4, 16, 4),
            nn.PReLU(),
            nn.Conv2d(16, 64, 4),
            nn.PReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 4, 4),
            nn.PReLU(),
            nn.ConvTranspose2d(4, 1, 4),
        )
        '''
    def forward(self, input):
        # return self.net(input.view(input.shape[0], -1)).view(input.shape[0], 10, 10)
        return self.decoder(self.encoder(input))
        '''
        encoded_input = self.encoder(input)
        mlped_input = self.mlp(encoded_input.view(encoded_input.shape[0], encoded_input.shape[1]))
        return self.decoder(mlped_input.view(mlped_input.shape[0], mlped_input.shape[1], 1, 1))
        '''

if __name__ == "__main__":
    from MineSweepingEnv import MineSweepingEnv, random_mine_layout
    import math
    env = MineSweepingEnv()
    obs = env.reset(random_mine_layout((10, 10), 10))
    policy = MineSweepingPolicy(torch.tensor([10, 10]))
    while True:
        obs = obs.view(1, *obs.shape)
        Qs = policy(obs)
        print("========================")
        print("obs")
        print(obs)
        print("========================")
        print("Q")
        print(Qs)
        print("========================")
        Qs[obs != -1] = float("-inf")
        print("========================")
        print("processed Q")
        print(Qs)
        print("========================")
        Qs = Qs.view(Qs.shape[1], Qs.shape[2])
        best_Q_pos = torch.argmax(Qs)
        best_action = torch.zeros_like(Qs)
        best_action[math.floor(best_Q_pos / Qs.shape[1]), best_Q_pos % Qs.shape[1]] = 1
        print(best_action)
        print("========================")
        input()
        obs, reward, terminated, truncated, info = env.step(best_action)
        print("reward: ")
        print(reward)
        if terminated:
            break
