# Install
```bash
conda create -n mine_rl python=3.10
conda activate mine_rl
pip install -r requirements.txt
```

# Train Sweeping only
```bash
mkdir result
mkdir saved_models
python ./TrainMineSweepingPolicy.py
```

Note: The training code won't stop automatically. You need to stop it using Ctrl-C.

# Train Sweeping policy as well as Burying policy
```bash
mkdir two_policy_result
mkdir two_policy_saved_models
python ./TrainSweepAndBury.py
```

Note: The training code won't stop automatically. You need to stop it using Ctrl-C.

# Visualize the result
I provide several visualization functions in `visualize.py`. Please refer to that file.
