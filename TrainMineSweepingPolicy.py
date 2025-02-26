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

def collect_data(mine_num, policy, epsilon):
    env = MineSweepingEnv()
    collected_data = []
    while True:
        obs, reset_successful = env.reset(random_mine_layout((10, 10), mine_num))
        if reset_successful:
            break
    with torch.no_grad():
        while True:
            obs = obs.view(1, 1, *obs.shape)
            Qs = policy(obs)
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
    return collected_data

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

if __name__ == "__main__":
    env = MineSweepingEnv()
    policy = MineSweepingPolicy(torch.tensor([10, 10]))
    learning_rate = 2.5e-4
    optimizer = Adam(policy.parameters(), lr = learning_rate)
    iter = 0
    epsilon = 0.1 # used for epsilon greedy
    mine_num = 1
    best_win_rate = 0.0
    replay_buffer = []
    max_replay_buffer_size = int(1e5)
    batch_size = int(1e3)
    omega = 0.5 # used for prioritized replay
    while True:
        iter += 1
        loss = torch.tensor([0.0])
        pool = multiprocessing.Pool(processes=1)
        # with multiprocessing.Pool(processes=10) as pool:
        results = []
        '''
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
        results = [collect_data(mine_num, policy, epsilon) for _ in range(100)]
        for data in results:
            replay_buffer.extend(data)
        replay_buffer = replay_buffer[-max_replay_buffer_size:]
        print(f"iter: {iter}")
        print(f"mine num: {mine_num}")
        for _ in range(5):
            loss = torch.tensor([0.0])
            true_batch_size = min(len(replay_buffer), batch_size)
            for index in prioritized_sample(replay_buffer, true_batch_size):
                transition, weight = replay_buffer[index]
                obs, action, reward, new_obs, terminated = transition
                Q = policy(obs)[0, 0, :]
                if terminated:
                    target_q = reward
                else:
                    target_q = reward
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
            print(f"loss: {loss}")
            optimizer.step()

        # test
        test_num = 100
        win = 0
        for i in range(test_num):
            f = None
            if i == 0:
                f = open(f"./result/{mine_num}.txt", "a")
                f.write("====================================\n")
                f.write("begin")
            while True:
                obs, reset_successful = env.reset(random_mine_layout((10, 10), mine_num))
                if reset_successful:
                    break
            while True:
                obs = obs.view(1, 1, *obs.shape)
                Qs = policy(obs)
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
            torch.save(policy.state_dict(), f"./saved_models/{mine_num}_{best_win_rate:.2f}_{iter}.pt")
            print(f"policy saved in ./saved_models/{mine_num}_{best_win_rate:.2f}_{iter}.pt")
        epsilon = (test_num - win) / test_num * 0.1
        if win / test_num > 0.5:
            if mine_num < 10:
                mine_num += 1
                print(f"updated mine num: {mine_num}")
                best_win_rate = 0.0
            optimizer = Adam(policy.parameters(), lr = learning_rate)
