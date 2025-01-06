# When Do Curricula Work in Federated Learning? [[Paper]](https://ieeexplore.ieee.org/document/10378572)

In this work, we investigate the effect of ordered learning in a federated system. Specifically, we aim to investigate how ordered learning principles can contribute to alleviating the heterogeneity effects in FL. We present theoretical analysis (in the paper) and conduct extensive empirical studies on the efficacy of orderings spanning three kinds of learning strategies:

1. Curriculum: Easily learned examples are presented first, followed by more difficult examples.
2. Anti-curriculum: Difficult examples are presented first, followed by easier examples.
3. Random curriculum (the usual): Examples are presented in a random order.

We find that curriculum learning largely alleviates non-IIDness. Interestingly, the more disparate the data distributions across clients the more they benefit from ordered learning. We show that curriculum training makes the objective landscape progressively less convex, suggesting fast converging iterations at the beginning of the training procedure.  

We derive quantitative results of convergence for both convex and nonconvex objectives by modeling the curriculum training on federated devices as local SGD with locally biased stochastic gradients.  

Further, inspired by ordered learning, we propose a novel client selection technique that benefits from the realworld disparity in the clients. Our proposed approach to client selection has a synergic effect when applied together with ordered learning in FL.

## Usage  

The code is organized as follows:

- `main.py` is the main script that is called by the scripts in `scripts/` to run the experiments.
- `src/` contains the source code for the experiments.  
  - `benchmark/` contains the algorithm main loop for federated learning.
  - `client` contains the client class for the various FL algorithms.
  - `data/` contains the dataset sharding scripts.
  - `datasets/` contains the dataset classes.
  - `models/` contains the NN model classes.
  - `pacing_fn` contains the pacing functions.
  - `utils/` contains the utility functions such as logging, scheduling, etc.
- `scripts/` contains the scripts to run the experiments.

### Running `main.py`  

| Parameter                     | Description                              |
| ----------------------------- | ---------------------------------------- |
| `ntrials`                     | Number of trials to run the experiment |
| `seed`                        | Random seed for reproducibility          |
| `rounds`                      | The number of rounds to train the model |
| `num_users`                   | Number of users/clients in the federated setting |
| `frac`                        | Fraction of users to be selected per round |
| `local_ep`                    | Number of local epochs each client performs per round |
| `local_bs`                    | Local batch size for training on each client |
| `lr0`                         | Initial learning rate                   |
| `lr_sched_a`                  | Parameter 'a' for learning rate scheduler |
| `lr_sched_b`                  | Parameter 'b' for learning rate scheduler |
| `w_decay`                     | Weight decay parameter for optimizer    |
| `momentum`                    | Momentum parameter for local client optimizers |
| `glob_momentum`               | Momentum parameter for global optimizer |
| `model`                       | Model architecture to use (e.g., 'resnet-9') |
| `dataset`                     | Dataset to use (e.g., 'isic2019') |
| `partition`                   | The partition way (e.g., 'noniid-labeldir') |
| `partition_difficulty_dist`   | Distribution of difficulty for partitions |
| `partition_ordering_f`        | Parameter 'f' for ordering partitions   |
| `num_partitions`              | Number of partitions to split data into |
| `ordering`                    | Curriculum Ordering ('rand') |
| `pacing_f`                    | Pacing Function ('linear') |
| `pacing_a`                    | Pacing parameter 'a' |
| `pacing_b`                    | Pacing parameter 'b' |
| `client_ordering`             | Ordering method for clients |
| `client_pacing_f`             | Pacing function for clients |
| `client_pacing_a`             | Client pacing parameter 'a' |
| `client_pacing_b`             | Client pacing parameter 'b' |
| `client_bs`                   | Client batch size |
| `exp_label`                   | Label for the experiment            |
| `datadir`                     | The path of the dataset. |
| `logdir`                      | The path to store the logs. |
| `ptdir`                       | The path to store the pre-trained models. |
| `train_expert`                | Boolean to train an expert model, default False |
| `log_clientnet`               | Boolean to log the client network, default False |
| `data_score_sample_p`         | Percentage of data to sample for scoring |
| `client_score_sample_n`       | Number of clients to sample for scoring |
| `log_filename`                | name of logfile|
| `alg`                         | Name of the algorithm ('fedavg_curr_lg_loss')|
| `beta`                        | The concentration parameter of the Dirichlet distribution for heterogeneous partition. (default: 0.2) |
| `local_view`                  | Boolean indicating whether to use a local view of the data distribution (True) |
| `lg_scoring`                  | Scoring method for local gradients ('LG') |
| `noise`                       | Noise level to add to data |
| `gpu`                         | GPU ID to use for training |
| `print_freq`                  | Frequency of printing logs during training |

## Citation

```bibtex
@inproceedings{vahidian2023curricula,
  title={When do curricula work in federated learning?},
  author={Vahidian, Saeed and Kadaveru, Sreevatsank and Baek, Woonjoon and Wang, Weijia and Kungurtsev, Vyacheslav and Chen, Chen and Shah, Mubarak and Lin, Bill},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5084--5094},
  year={2023}
}
```

## Acknowledgements  

Code inspired from https://github.com/google-research/understanding-curricula  
