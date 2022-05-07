# G-Bert

Originally forked from [this repository](https://github.com/jshang123/G-Bert). Please show them appreciation and be sure to cite them in any published work:

```latex
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
```

Pre-training of Graph Augmented Transformers for Medication Recommendation

## Intro

G-Bert combined the power of **G**raph Neural Networks and **BERT** (Bidirectional Encoder Representations from Transformers) for medical code representation and medication recommendation. We use the graph neural networks (GNNs) to represent the structure information of medical codes from a medical ontology. Then we integrate the GNN representation into a transformer-based visit encoder and pre-train it on single-visit EHR data. The pre-trained visit encoder and representation can be fine-tuned for downstream medical prediction tasks. Our model is the first to bring the language model pre-training schema into the healthcare domain and it achieved state-of-the-art performance on the medication recommendation task.

## Requirements

- pytorch
- python
- torch_geometric

## Docker Image

A docker image [mpossing/dlh-final](https://hub.docker.com/r/mpossing/dlh-final) has been prebuilt for
this task. The image could also be built by using the docker file in this repository.

## Running

*Note: All the run commands below assume that users are utilizing docker, and are being run from the
root of this repository. If docker is not available to the user, a python environment with the
requirements outlined in the [Requirements](#requirements) section of this document, as well as those
in the [requirements.txt](./requirements.txt) should be installed*

### Makefile

If you are on a unix system a makefile has been provided which runs the 3 commands automatically and can
pull the docker image. The commands are

```bash
make pull # pulls the docker image
make runBertTune # Runs the bert tuning process
make runFromConfig # Runs the configuration file code/batch_config.txt
make runLrTune # Runs the lr tuning process
```

### Docker

By using the docker image, a user can run the 3 processes directly. If using a windows environment, the
following steps are recommended:

```bash
# Be sure to replace the repo path in this line
docker run -it --rm -p 6006:6006 --gpus all -v $[insert repo path here]:/opt/project mpossing/dlh-final:latest bash
cd /opt/project/code
# Be sure to select the desired run file here
python $[run_bert_tune.py | run_from_config.py | run_lr_tune.py]
```

If using a unix environment, the commands are somewhat more simple:

```bash
# bert tune
docker run --rm -p 6006:6006 --gpus all -v $(pwd):/opt/project -w /opt/project/code mpossing/dlh-final:latest \
  python run_bert_tune.py
# run_from_config
docker run --rm -p 6006:6006 --gpus all -v $(pwd):/opt/project -w /opt/project/code mpossing/dlh-final:latest \
  python run_from_config.py
# run_lr_tune.py
docker run --rm -p 6006:6006 --gpus all -v $(pwd):/opt/project -w /opt/project/code mpossing/dlh-final:latest \
  python run_lr_tune.py
```

At the end of each one of these processes, a [tensorboard](https://www.tensorflow.org/tensorboard/) service is
started at localhost. navigating to [http://localhost:6006](http://localhost:6006) should bring up tensor board
such that the most recent results can be reviewed.

## Data

We have provided the same pkl file which the authors of the original repository provided. This file is all
it takes to run the models.

## Results

| Variant | Source       | Jaccard | F1   | PR-AUC |
|---------|--------------|---------|------|--------|
| G-      | Original     | 0.430   | 0.590| 0.677  |
|         | Current      | 0.449   | 0.606| 0.683  |
|         | % difference | 4.398   | 2.586| 0.829  |
| G-, P-  | Original     | 0.419   | 0.580| 0.665  |
|         | Current      | 0.382   | 0.541| 0.628  |
|         | % difference |-9.143   |-6.889|-5.708  |
| G-Bert  | Original     | 0.457   | 0.615| 0.696  |

