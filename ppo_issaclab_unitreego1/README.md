
# Setup

```

$ conda create -n py310 python=3.10
$ conda activate py310

$ pip install --upgrade pip
$ pip install "jax[cuda12]" flax
$ pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

$ sudo apt update
$ sudo apt install -y libsm6 libglu1-mesa
$ sudo apt install -y libxt6
$ sudo apt install -y cmake build-essential
$ git clone https://github.com/isaac-sim/IsaacLab.git
```
