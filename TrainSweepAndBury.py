from MineSweepingEnv import MineSweepingEnv, random_mine_layout
from MineSweepingPolicy import MineSweepingPolicy
import torch
from torch.optim import Adam
import math
import copy
# import multiprocessing
from torch import multiprocessing
import torch.nn.functional as F
import random

def generate_mines(mine_num, bury_policy, epsilon):
    # the second return is the history of generation, doesn't contain reward
    collected_data = []
    mines = torch.zeros(1, 1, 10, 10)
    with torch.no_grad():
        for i in range(mine_num):
            Qs = bury_policy(mines)
            Qs[mines == 1] = float("-inf")
            Qs = Qs.view(Qs.shape[2], Qs.shape[3])
            if torch.rand(1) > epsilon: 
                action_pos = torch.argmax(Qs)
                action = torch.zeros_like(Qs)
                action[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] = 1
            else:
                while True:
                    action_pos = torch.randint(0, Qs.numel(), (1,))
                    if Qs[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] != float("-inf"):
                        action = torch.zeros_like(Qs)
                        action[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] = 1
                        break
            new_mines = mines + action.view(1, 1, *action.shape)

            if i == mine_num - 1:
                terminated = True
            else:
                terminated = False
            collected_data.append([(mines, action, new_mines, terminated), None])

            mines = new_mines
    return mines, collected_data


def sweep_mine(mines, sweep_policy, epsilon):
    env = MineSweepingEnv()
    collected_data = []
    while True:
        obs, reset_successful = env.reset(mines)
        if reset_successful:
            break
    with torch.no_grad():
        while True:
            obs = obs.view(1, 1, *obs.shape)
            Qs = sweep_policy(obs)
            Qs[obs != -1] = float("-inf")
            Qs = Qs.view(Qs.shape[2], Qs.shape[3])
            if torch.rand(1) > epsilon: 
                action_pos = torch.argmax(Qs)
                action = torch.zeros_like(Qs)
                action[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] = 1
            else:
                while True:
                    action_pos = torch.randint(0, Qs.numel(), (1,))
                    if Qs[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] != float("-inf"):
                        action = torch.zeros_like(Qs)
                        action[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] = 1
                        break
            new_obs, reward, terminated, truncated, info = env.step(action)
            collected_data.append([(obs, action, reward, new_obs.view(1, *new_obs.shape), terminated), None])
            obs = new_obs
            if terminated:
                break
    return collected_data, sum([d[0][2] for d in collected_data])

def prioritized_sample(replay_buffer, sample_num):
    replay_buffer_length = len(replay_buffer)
    not_none_priorities = [replay_buffer[i][1] for i in range(replay_buffer_length) if replay_buffer[i][1] is not None]
    if len(not_none_priorities) > 0:
        max_priority = max(not_none_priorities)
    else:
        max_priority = 1.0
    return random.choices(
        list(range(replay_buffer_length)),
        weights = [replay_buffer[i][1] if replay_buffer[i][1] is not None else max_priority for i in range(len(replay_buffer))],
        k = sample_num
    )

def train_policy(policy, optimizer, replay_buffer, batch_size, bootstrap):
    loss = torch.tensor([0.0])
    true_batch_size = min(len(replay_buffer), batch_size)
    for index in prioritized_sample(replay_buffer, true_batch_size):
        transition, weight = replay_buffer[index]
        obs, action, reward, new_obs, terminated = transition
        Q = policy(obs)[0, 0, :]
        if terminated or not bootstrap:
            target_q = reward
        else:
            target_q = reward + torch.max(policy(new_obs))
        td_error = Q[action == 1] - target_q
        # update priority
        replay_buffer[index][1] = torch.abs(td_error).item() ** omega
        if weight is not None:
            loss += 1 / (weight + 1e-6) * td_error ** 2
        else:
            loss += td_error ** 2
    loss /= true_batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return policy, replay_buffer, loss

if __name__ == "__main__":
    env = MineSweepingEnv()
    sweep_policy = MineSweepingPolicy(torch.tensor([10, 10]))
    bury_policy = MineSweepingPolicy(torch.tensor([10, 10]))
    learning_rate = 5e-4
    sweep_optimizer = Adam(sweep_policy.parameters(), lr = learning_rate)
    bury_optimizer = Adam(bury_policy.parameters(), lr = learning_rate)
    iter = 0
    sweep_epsilon = 0.2 # used for epsilon greedy
    bury_epsilon = 0.2 # used for epsilon greedy
    mine_num = 1
    best_win_rate = 0.0
    bury_replay_buffer = []
    sweep_replay_buffer = []
    max_replay_buffer_size = int(1e5)
    batch_size = int(1e3)
    omega = 0.5 # used for prioritized replay
    bury_percentage = 0.0
    while True:
        iter += 1
        loss = torch.tensor([0.0])
        # with multiprocessing.Pool(processes=10) as pool:
        results = []
        '''
        pool = multiprocessing.Pool(processes=1)
        for _ in range(100):
            env = MineSweepingEnv()
            collected_data = []
            while True:
                obs, reset_successful = env.reset(random_mine_layout((10, 10), mine_num))
                if reset_successful:
                    break
            results.append(pool.apply_async(collect_data, (env, obs, policy, epsilon)))
        pool.close()
        pool.join()
        for data in results:
            replay_buffer.extend(data.get())
        '''
        bury_results = [generate_mines(mine_num, bury_policy, bury_epsilon) if i < 100 * bury_percentage else (random_mine_layout((10, 10), mine_num), None) for i in range(100)]
        sweep_results = [sweep_mine(bury_result[0], sweep_policy, sweep_epsilon) for bury_result in bury_results]

        for bury_data, sweep_data in zip(bury_results, sweep_results):
            sweep_replay_buffer.extend(sweep_data[0])
            bury_data = bury_data[1]
            if bury_data is not None:
                # not randomly generated mines
                for i in range(len(bury_data)):
                    # reward = -sweep_data[1] if i == len(bury_data) - 1 else 0.0
                    reward = -sweep_data[1]
                    bury_data[i] = [(bury_data[i][0][0], bury_data[i][0][1], reward, bury_data[i][0][2], bury_data[i][0][3]), bury_data[i][1]]
                bury_replay_buffer.extend(bury_data)
        sweep_replay_buffer = sweep_replay_buffer[-max_replay_buffer_size:]
        bury_replay_buffer = bury_replay_buffer[-max_replay_buffer_size:]

        print(f"iter: {iter}")
        print(f"mine num: {mine_num}")
        for _ in range(5):
            sweep_policy, sweep_replay_buffer, sweep_loss = train_policy(sweep_policy, sweep_optimizer, sweep_replay_buffer, batch_size, bootstrap=False)
            if len(bury_replay_buffer) != 0:
                bury_policy, bury_replay_buffer, bury_loss = train_policy(bury_policy, bury_optimizer, bury_replay_buffer, batch_size, bootstrap=False)
            else:
                bury_loss = torch.tensor(float("inf"))

            print(f"sweep loss: {sweep_loss.item()}; bury loss: {bury_loss.item()}")

        # test
        test_num = 100
        win = 0
        for i in range(test_num):
            f = None
            if i == 0:
                f = open(f"./two_policy_result/{mine_num}.txt", "a")
                f.write("====================================\n")
                f.write("begin")
            while True:
                obs, reset_successful = env.reset(random_mine_layout((10, 10), mine_num))
                if reset_successful:
                    break
            while True:
                obs = obs.view(1, 1, *obs.shape)
                Qs = sweep_policy(obs)
                Qs[obs != -1] = float("-inf")
                Qs = Qs.view(Qs.shape[2], Qs.shape[3])
                action_pos = torch.argmax(Qs)
                action = torch.zeros_like(Qs)
                action[math.floor(action_pos / Qs.shape[1]), action_pos % Qs.shape[1]] = 1
                if f is not None:
                    f.write("====================================\n")
                    f.write(str(obs) + "\n")
                    f.write("=====================================\n")
                    f.write(str(Qs) + "\n")
                    f.write("=====================================\n")
                    f.write(str(action) + "\n")
                    f.write("=====================================\n")
                new_obs, reward, terminated, truncated, info = env.step(action)
                obs = new_obs
                if terminated:
                    if reward >= 0:
                        win += 1
                    if f is not None:
                        f.write("finished")
                        f.write("=====================================\n")
                    break
        print(f"win: {win} / {test_num}")
        if win / test_num > best_win_rate:
            best_win_rate = win / test_num
            torch.save(sweep_policy.state_dict(), f"./two_policy_saved_models/sweep_{mine_num}_{best_win_rate:.2f}_{iter}.pt")
            torch.save(bury_policy.state_dict(), f"./two_policy_saved_models/bury_{mine_num}_{best_win_rate:.2f}_{iter}.pt")
            print(f"policy saved in ./two_policy_saved_models/{mine_num}_{best_win_rate:.2f}_{iter}.pt")
        sweep_epsilon = (test_num - win) / test_num * 0.2
        bury_epsilon = win / test_num * 0.2
        bury_percentage = win / test_num
        if win / test_num > 0.5:
            if mine_num < 10:
                mine_num += 1
                print(f"updated mine num: {mine_num}")
                best_win_rate = 0.0
                sweep_optimizer = Adam(sweep_policy.parameters(), lr = learning_rate)
                bury_optimizer = Adam(bury_policy.parameters(), lr = learning_rate)
                bury_percentage = 0.0
                bury_replay_buffer = []
