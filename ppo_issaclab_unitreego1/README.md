
# Setup

```
uv sync
uv add "jax[cuda12]" flax
uv pip install --upgrade pip
uv pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

sudo apt update
sudo apt install -y libsm6 libglu1-mesa
sudo apt install -y libxt6
git clone https://github.com/isaac-sim/IsaacLab.git
cd IssacLab
```
