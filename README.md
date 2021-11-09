# Learning Queueing Policies for Organ Transplantation Allocation using _Interpretable_ Counterfactual Survival Analysis  </br><sub><sub>J. Berrevoets, A. M. Alaa, Z. Qian, J. Jordon, A. E. S. Gimson, M. van der Schaar [[ICML 2021]](http://proceedings.mlr.press/v139/berrevoets21a/berrevoets21a.pdf)</sub></sub>

In this repository we provide code for our ICML21 paper introducing OrganSync, a novel organ-to-patient allocation system. Note that this code is used for research purposes and is __not intented for use in practice__.

In our paper we benchmark against OrganITE a previously introduced paper of ours. We have reimplemented OrganITE (as well as other benchmarks) using the same frameworks in this repository, such that all code is comparable throughout. For all previous implementations, we refer to OrganITE's dedicated [repository](https://github.com/vanderschaarlab/mlforhealthlabpub/tree/main/alg/organite).

_Code author: J. Berrevoets ([jb2384@cam.ac.uk](mailto:jb2384@cam.ac.uk))_

## Repository structure
This repository is organised as follows:
```bash
organsync/
    |- src/
        |- organsync/                       # Python library core
            |- data/                        # code to preprocess data
            |- eval_policies/               # code to run allocation simulations
            |- models/                      # code for inference models
    |- experiments/
        |- data                             # data modules
        |- models                           # training logic for models
        |- notebooks/wandb
            |- simulation_tests.ipynb       # experiments in Tab.1
            |- a_composition                # experiments in Fig.3
            |- sc_influence.ipynb           # experiments in Fig.4, top row
            |- rep_influence.ipynb          # experiments in Fig.4, bottom row
    |- test                                 # unit tests
    |- data                                 # datasets
```

## Used frameworks
We make extensive use of Weights and Biases ([W&B](https://wandb.com)) to log our model performance as well as trained model weights. To run our code, we recommend to create a W&B account ([here](https://wandb.ai/login?signup=true)) if you don't have one already. All code is written in [pytorch](https://pytorch.org) and [pytorch-lightning](http://pytorchlightning.ai/).


## Installing
We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
pip install .
```
Please use the above in a newly created virtual environment to avoid clashing dependencies. All code was written for `python 3.8.6`.

## Running experiments
As indicated above, each notebook represents one experiment. The comments provided in the project hierarchy indicate the figure or table, and the specific paper the experiment is presented in. As a sidenote, in order to run simulation experiments (`experiments/notebooks/wandb/simulation_tests.ipynb`), you will need to have trained relevant inference models if the allocation policy requires them.

Training a new model (e.g. `src/organsync/models/organsync_network.py`) is done simply as
```bash
python -m experiments.models.organsync
```
(Please run python -m experiments.models.organsync --help to see options). When training is done, the model is automatically uploaded to W&B for use later in the experiments.*

## Citing
Please cite our paper and/or code as follows:
```tex

@InProceedings{organsync,
  title = 	 {{Learning Queueing Policies for Organ Transplantation Allocation using Interpretable Counterfactual Survival Analysis}},
  author =       {Berrevoets, Jeroen and Alaa, Ahmed M. and Qian, Zhaozhi and Jordon, James and Gimson, Alexander E.S. and van der Schaar, Mihaela},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {792--802},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/berrevoets21a/berrevoets21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/berrevoets21a.html},
}
```

<sub>* Note that we retrain the models used in TransplantBenefit to give a fair comparison to the other benchmarks, as well as compare on the UNOS data.</sub>
