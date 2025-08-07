# CHES2025 Challenge

## Setup for Experiments

- Prepare a `Python 3.11` environment and install the dependency by
```
pip install -r requirements.txt
```

- Download the data from https://drive.google.com/drive/folders/1JGbphwZXQvN_tEhpBIbQ-q-pN9wkqKQ- and put the `.h5` file at `./Dataset/CHES_2025/CHES_Challenge.h5`.

- run `python main.py` to start one experiment. It will read the configurations in `expr_config.json` and execute the training and evaluation.

- run `tensorboard --logdir=Results` to monitor the training and evaluation results through a browser.

## Projecture Structure

- `config_arxiv`: The arxiv of training configurations for execellent models we discovered.

- `profiler`: The profiling tool. See `PROFILING_README.md`.

- `src`: The source for training and evaluation.

- `main.py`: The launcher for one experiment of training and evaluation according to the configurations in `expr_config.json`.

- `expr_config.json`: Configuration file for `main.py`.

- `run_profiling.py`: The launcher for profiling.

- `random_training_loop.py`: The launcher for hyperparameter searching random training, according to `random_config_ranges.json`. The information summary for different random trainings will be collected and presented in a `.csv` file in the output folder.

- `random_config_ranges.json`: Configuration file for `random_training_loop.py`.

- `Results`: The configurations, model parameters and evaluation results of all trainings are preserved here.


## Ideas

- Since we have data of `plaintext` and `key`, we can try to predict both of them.

- The current raw data only has one channel. We can augment the training data with other channels of pre-processed information (e.g. ewm, differentiated data).

- **!!** The original method provided in the code tries to extract the leakage after `S-BOX` operation, and then reconstruct the key. Maybe it's better to use two models to predict the key and the leakage at the same time, and make sure they are consistent.

- We can let the model predict both the `ID` leakage and the `HW` leakage, and make sure they are consistent.

- We have 500K training traces for 256 different keys. This means 2K traces per key in average. If our model performs really well, we can use these data to test the model more comprehensively.

## Observations

- The `leakage_model` determines what kind of leakge the DL model tries to extract from the power trace. The key is then reconstructed from the model.

- How to interprete the `GE` and `NTGE` score provided by evaluation: The `i`-th element of `GE` score represents the mean rank of corect key (across different attacks) when `i` traces are utilized. Therefore the smaller, the better. Ideally we will observe successive zeros at the end. In this case, the NTGE score evaluates from which step the subsequent `GE` scores are zero. Therefore it is also the smaller, the better.

- However, the competition rule says that only 1 attack is considered when scoring. This is consistent with the fact that all scores in the leaderboard are integers. 

- The original code provides the searching in the hyper parameter space.

- The division of training-validation-test splits are incorrect. Originally the validation split is from the traces of same key, and the model can be misled and optimize for this key only. I think the validation set should be from the training data.

- Overfitting is a huge problem. Even if the validation loss keeps dropping after the first several epochs, the actual test performance will still decrease.

- Even validation loss cannot reliably predict the result of real test NTGE score. This is strange.

- Data augumentation indeed helps reduce overfitting, but later training still cannot improve the performance. The non-arguemented version is still better.

## Log
- 8/7 Add hyperparameter searching random training.
- 8/6 Add data augmentation by Gaussian noise and random shift
- **8/6 We obtained an excellent model `M8`, which achieves 10K score by a 4-layer MLP.**
- 8/6 Add early stop by validation loss.
- 8/6 Added profiling script (AI Generated).
- 8/6 Refactorized the project. Now we can specify all experimenting parameters using a json file.
- 8/7 Now the validation set is split from the training data, not the attack data (to avoid bais on the same key).
- 8/5 Add plot output for GE scores in the evaluation script.
- 8/5 Use `numba` just-in-time compilation optimization and evaluation speeds up for 10 times. Correctness verified.

## TODO
- Add monitor for key_log_prob evolution.
- Add same key validation set