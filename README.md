# CVRL
This repo contains the Tensorflow 2.0 implementation for the CoRL 2020 paper

Xiao Ma, Siwei Chen, David Hsu, Wee Sun Lee: Contrastive Variational Model-Based Reinforcement Learning for Complex Observations. In Proc. 4th Conference on Robot Learning. [[paper]](https://arxiv.org/abs/2008.02430)

For visualzations, please visit our [project page](https://sites.google.com/view/cvrl/home)

## Setup
```
pip3 install --user tensorflow-gpu==2.2.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

You will need the [Mujoco license](https://www.roboti.us/license.html) to run the Mujoco tasks. 

To play with the natural Mujoco tasks, download the natural Mujoco background dataset from [here](https://drive.google.com/drive/folders/1r7i1PYY_Yhfhu7T8hlhi2DJtaeD6lIvp?usp=sharing) and put it at the root of this folder. 


## Train the agent:

```
python3 cvrl.py --logdir ./logdir/dmc_walker_walk/natural_walker_walk/1 --task dmc_walker_walk --natural True --obs_model contrastive --use_dreamer True --use_sac True --trajectory_opt True
```

To view the training logs and execution videos, please use 
```
tensorboard --logdir ./logdir --bind_all
```

## Cite CVRL

If you find this repo useful, please consider citing our paper

```bibtex
@inproceedings{
    ma2020contrastive,
    title={Contrastive Variational Model-Based Reinforcement Learning for Complex Observations},
    author={Xiao Ma and Siwei Chen and David Hsu and Wee Sun Lee},
    booktitle={Proceedings of the 4th Conference on Robot Learning},
    year={2020}
}
```

## Reference
The code borrows heavily from Danijar Hafner's Dreamer [implementation](https://github.com/danijar/dreamer).
