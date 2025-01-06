# Curriculum-FL

code based on https://github.com/google-research/understanding-curricula

## Usage
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dir` | The path to store the logs. |
| `rounds` | The number of rounds to train the model |
| `dataset`      | Dataset to use. Options: `cifar10`, `fmnist`, `svhn`. |
| `partition`    | The partition way. Options: `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `homo`. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. (default: 0.5) |
| `datadir` | The path of the dataset. |
| `savedir` | The path to store the final results and models. |
| `ordering` | Curriculum Ordering ('curr', 'anti_curr', 'random'). |
| `pacing_f` | Pacing Fucntions ('linear', 'quad', 'root', 'step', 'exp', 'log'). |
| `pacing_a` | Pacing_a for Pacing Function |
| `pacing_b` | Pacing_b for Pacing Function |
| `alg` | name of algorithm ('fedavg_curr', 'fedavg', 'fedprox', 'fednova', 'scaffold', ...)|
| `log_filename` | name of logfile|


To Run the code:
```
cd scripts_rci
sh fedavg_curr.sh 
```
Checklist of things you need to change: 
```
log_filename: "$ordering_$pacingf_$pacinga_$pacingb' --> ex: 'curr_linear_0.2a_0.4b'
partition: 'homo', 'noniid-#label2', 'noniid-labeldir'
ordering
pacing_f
pacing_a
pacing_b
```
