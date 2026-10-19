# Slab-PINN

Physics-Informed Neural Networks for predicting the thermal structure, velocity field, and pressure field of subducting slabs. The model embeds mass conservation, energy conservation, Stokes momentum conservation, and boundary/initial conditions into the loss function, fusing observed data with physical laws.

## Demo

**[Online Demo (Streamlit)](https://pinn-subduction-futwhyg4pztd7yxmqcjq2c.streamlit.app/)**

## Language & Framework

- **Language**: Python 3.11
- **Deep Learning Framework**: PyTorch 2.1.1
- **Acceleration**: CUDA (GPU recommended)

## Installation

```bash
pip install torch==2.1.1 numpy tqdm pillow
```

## Dataset

The full training dataset is hosted on ModelScope:

**[pinn-slab-dataset](https://www.modelscope.cn/datasets/dandified/pinn-slab-dataset)**

### Data Format

Each sample is a CSV file (comma-separated). File naming convention:

```
TempHeat{T}Age{A}Vel{V}.{step}
```

- `TempHeat{T}`: temperature / heat-flow parameter
- `Age{A}`: subduction age parameter
- `Vel{V}`: subduction velocity parameter
- `{step}`: time-step index (0 – 15000, step 50)

Input columns: `x, z, time, heat_flow, dip, age, velocity` (7-dim).
Output columns (space-separated within each record): `T, Vx, Vz, P, Vi` — temperature, two velocity components, pressure, and pointwise relative viscosity. The network predicts the first 4 channels (`T, Vx, Vz, P`); `Vi` is only used by the Stokes momentum residual, not a prediction target.

Normalization (see `dataset/NNDataset.py`):
- `T`: clipped to [0, 1320] K, then / 1320
- `Vx, Vz`: symmetric scaling by the plate velocity → [0, 1]
- `P`: symmetric scaling `Pn = P / (2 * P_SCALE) + 0.5` with `P_SCALE = 1.2e12` Pa (dataset-specific; rescan `Dat_uniP` extremes when using new simulation data)

### Directory Layout

After downloading, arrange the data as:

```
data/fnn/
├── train/
│   ├── input/
│   └── output/
├── valid/
│   ├── input/
│   └── output/
└── test/
    ├── input/
    └── output/
```

`input/` and `output/` filenames must match one-to-one for pairing.

## Usage

### Training

```bash
# PINN with default loss weights (alpha=beta=gamma=delta=epsilon=1.0)
python train_pifnn.py

# Train several weight configurations in one run (alpha,beta,gamma,delta,epsilon)
python train_pifnn.py --weights 1.0,1.0,1.0,1.0,1.0 1.0,1.0,1.0,1.0,1e-8 1.0,1.0,1.0,1.0,1e-12

# Also run test-set evaluation after each configuration
python train_pifnn.py --weights 1.0,1.0,1.0,1.0,1e-8 --test
```

### Loss Weights

Passed via `--weights` (order: `alpha,beta,gamma,delta,epsilon`):

| Weight | Loss term |
|--------|-----------|
| `alpha` | Mass conservation (incompressibility) residual |
| `beta` | Energy conservation (thermal advection-diffusion) residual |
| `gamma` | Initial condition (t = 0 thermal structure) |
| `delta` | Boundary condition (top/bottom temperature) |
| `epsilon` | Stokes momentum residual (pressure gradient vs. viscous stress) |

Each configuration's checkpoints and logs are written to `./result/` with the weights encoded in the filename, e.g. `pifnn_1.0_1.0_1.0_1.0_1e-08.pth`, `pifnn_1.0_1.0_1.0_1.0_1e-08_trainingData.txt`, so multiple runs never overwrite each other.

> Note: the raw momentum residual is of order ~1e7 in `(Pa/m)^2`, much larger than the data MSE. Sweep `epsilon` over small values (e.g. 1e-8 to 1e-12) and check the logged momentum component.

### Key Parameters

Edit `train()` in `train_pifnn.py`:

```python
learning_rate = 5e-4
batch_size = 128
epoch = 5
```

### Data Conversion

`dataset/GenerateProcessData.py` converts raw simulation output (`origin_*` directories with `Dat_uniT/XV/ZV/P/Vi` files) into the training format:

```python
split_fnn5(tag="train_20260831_p", target_tag="train_20260831")
```

### Output

Trained checkpoints are saved to `./result/` (e.g. `pifnn_*.pth`).

## Project Structure

```
slab-pinn/
├── model/                 # Network definitions (FNN, ConvAE)
├── dataset/               # Dataset loaders (NNDataset.py) and data conversion (GenerateProcessData.py)
├── data/fnn/              # Data directory (train/valid/test), not tracked
├── train_pifnn.py         # Main PINN training script (7 inputs -> 4 outputs: T, Vx, Vz, P)
└── README.md
```

The training script expects to run from a directory containing `data/fnn/` and `generate_boundary.py` (boundary condition & autograd gradient utilities).

## License

Released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Slab-PINN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
