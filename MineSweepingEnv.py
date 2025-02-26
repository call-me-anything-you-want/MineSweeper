import torch
import torch.nn.functional as F
import pprint
class MineSweepingEnv:
    def reset(self, mine_tensor: torch.Tensor):
        # mine_tensor should be a 01 tensor of 10*10
        # in which 1 represents mine
        while mine_tensor.shape[0] == 1:
            mine_tensor = mine_tensor[0, :]
        self.mine_tensor = mine_tensor.float()
        self.total_spot_num = self.mine_tensor.numel()
        self.total_mine_num = mine_tensor.sum()
        assert self.total_mine_num != 0, "There needs to be at least one mine in the field."

        conv_filter = torch.tensor(
            [[[
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ]]],
            dtype = torch.float
        )
        # each number indicates the number of mines around it
        self.full_state_tensor = F.conv2d(
            input = self.mine_tensor.view(1, 1, *self.mine_tensor.shape),
            weight = conv_filter,
            padding = 1
        )[0, 0, :, :]

        # -1 represents covered spots
        self.current_state_tensor = -1 * torch.ones_like(self.full_state_tensor, requires_grad=False)

        # randonly open a spot
        possible_positions = torch.where(self.mine_tensor != 1)
        open_idx = torch.randint(0, possible_positions[0].shape[0], (1,))
        action_tensor = torch.zeros_like(self.full_state_tensor)
        action_tensor[possible_positions[0][open_idx], possible_positions[1][open_idx]] = 1
        self.step(action_tensor)

        # return observation and whether the env reset successfully or not
        return self.current_state_tensor.clone(), (self.current_state_tensor == -1).sum() != self.total_mine_num

    def step(self, action_tensor: torch.Tensor):
        assert action_tensor.sum() == 1, f"The number of spots uncovered each time should be 1, however, we got {action_tensor.sum().item()}"
        truncated = False
        info = {}
        reward = 0
        distance = self.get_closest_uncovered_distance(action_tensor)
        if distance > 0:
            reward += (- distance) * 0.5
        if (self.mine_tensor * action_tensor).sum() == 1:
            # uncover a mine
            terminated = True
            # reward = -1 - (self.current_state_tensor != -1).sum() / (self.total_spot_num - self.total_mine_num)
            reward += -1
        else:
            uncover_num = self.uncover(action_tensor)
            if (self.current_state_tensor == -1).sum() == self.total_mine_num:
                # all spots are uncovered
                terminated = True
            else:
                terminated = False
            # reward = uncover_num / (self.total_spot_num - self.total_mine_num)
            reward += 1
        observation = self.current_state_tensor.clone()
        return observation, reward, terminated, truncated, info

    def render(self):
        pprint.pprint(self.current_state_tensor.long().tolist())
    
    def get_closest_uncovered_distance(self, action_tensor):
        action_spot = torch.where(action_tensor == 1)
        process_positions = [(action_spot[0][0], action_spot[1][0], 0)]
        processed_tensor = torch.zeros_like(self.mine_tensor)
        while len(process_positions) > 0:
            current_position= process_positions.pop()
            if processed_tensor[current_position[0], current_position[1]] == 1:
                continue
            else:
               processed_tensor[current_position[0], current_position[1]] = 1
            if self.mine_tensor[current_position[0], current_position[1]] != -1:
                return current_position[2]
            neighbors = self.neighbors((current_position[0], current_position[1]))
            process_positions = [(p[0], p[1], current_position[2] + 1) for p in neighbors] + process_positions
        return -1


    def uncover(self, action_tensor):
        # update self.current_state_tensor
        # return the total number of uncovered spots
        uncover_spot = torch.where(action_tensor == 1)
        process_positions = [(uncover_spot[0][0], uncover_spot[1][0])]
        total_uncover_num = 0
        while len(process_positions) > 0:
            current_position = process_positions.pop()
            if self.mine_tensor[current_position[0], current_position[1]] == 1:
                # a mine position
                continue
            if self.current_state_tensor[current_position[0], current_position[1]] != -1:
                # has been uncovered before
                continue
            self.current_state_tensor[current_position[0], current_position[1]] = self.full_state_tensor[current_position[0], current_position[1]]
            total_uncover_num += 1
            if self.full_state_tensor[current_position[0], current_position[1]] == 0:
                process_positions.extend(self.neighbors(current_position))
            else:
                continue
                # process_positions.extend([p for p in self.neighbors(current_position) if self.full_state_tensor[p[0], p[1]] == 0])
        return total_uncover_num

    def neighbors(self, current_position: tuple):
        neighbors = [
            (current_position[0] - 1, current_position[1] - 1),
            (current_position[0], current_position[1] - 1),
            (current_position[0] + 1, current_position[1] - 1),
            (current_position[0] - 1, current_position[1]),
            (current_position[0] + 1, current_position[1]),
            (current_position[0] - 1, current_position[1] + 1),
            (current_position[0], current_position[1] + 1),
            (current_position[0] + 1, current_position[1] + 1),
        ]
        return [n for n in neighbors if n[0] >= 0 and n[0] < self.full_state_tensor.shape[0] and n[1] >= 0 and n[1] < self.full_state_tensor.shape[1]]

def random_mine_layout(layout_size: tuple, mine_num: int, initial_mines: torch.Tensor = None):
    assert mine_num <= layout_size[0] * layout_size[1], f"{mine_num} mines are too many for layout of size {layout_size}"
    if initial_mines is not None:
        assert initial_mines.shape == layout_size, f"The initial_mines should have the same shape as layout_size"
        mines = initial_mines
    else:
        mines = torch.zeros(layout_size)
    while mines.sum() < mine_num:
        while True:
            x = torch.randint(0, mines.shape[0], (1,))
            y = torch.randint(0, mines.shape[1], (1,))
            if mines[x, y] != 1:
                mines[x, y] = 1
                break
    return mines

if __name__ == "__main__":
    env = MineSweepingEnv()
    mines = torch.zeros((10, 10))
    mine_positions = torch.randint(0, 10, (2, 2))
    for i in range(2):
        mines[mine_positions[i, 0], mine_positions[i, 1]] = 1
    print(mines)
    env.reset(mines)
    env.render()
    while True:
        new_position = torch.tensor(eval(input()))
        action = torch.zeros((10, 10))
        action[new_position[0], new_position[1]] = 1
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"r: {reward}")
        if terminated:
            break
        env.render()
