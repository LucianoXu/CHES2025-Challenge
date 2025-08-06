# SCA

## Ideas

- Since we have data of `plaintext` and `key`, we can try to predict both of them.

- The current raw data only has one channel. We can augment the training data with other channels of pre-processed information (e.g. ewm, differentiated data).

- **!!** The original method provided in the code tries to extract the leakage after `S-BOX` operation, and then reconstruct the key. Maybe it's better to use two models to predict the key and the leakage at the same time, and make sure they are consistent.

- We can let the model predict both the `ID` leakage and the `HW` leakage, and make sure they are consistent.

## Observations

- The `leakage_model` determines what kind of leakge the DL model tries to extract from the power trace. The key is then reconstructed from the model.

- How to interprete the `GE` and `NTGE` score provided by evaluation: The `i`-th element of `GE` score represents the mean rank of corect key (across different attacks) when `i` traces are utilized. Therefore the smaller, the better. Ideally we will observe successive zeros at the end. In this case, the NTGE score evaluates from which step the subsequent `GE` scores are zero. Therefore it is also the smaller, the better.

- However, the competition rule says that only 1 attack is considered when scoring. This is consistent with the fact that all scores in the leaderboard are integers. 

- The original code provides the searching in the hyper parameter space.

- The division of training-validation-test splits are incorrect. Originally the validation split is from the traces of same key, and the model can be misled and optimize for this key only. I think the validation set should be from the training data.

## Log

- 8/6 Refactorized the project. Now we can specify all experimenting parameters using a json file.
- 8/7 Now the validation set is split from the training data, not the attack data (to avoid bais on the same key).
- 8/5 Add plot output for GE scores in the evaluation script.
- 8/5 Use `numba` just-in-time compilation optimization and evaluation speeds up for 10 times. Correctness verified.
