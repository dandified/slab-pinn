# Slab-PINN

Physics-Informed Neural Networks for predicting the thermal structure of subducting slabs. The model embeds mass conservation, energy conservation, and boundary/initial conditions into the loss function, fusing observed data with physical laws.

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
Output: temperature (normalized to [0, 1], i.e. 0–1320 K) and two velocity components.

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
# PINN with adaptive weights (recommended)
python train_pifnn.py

# Pure FNN (no physics constraints)
python train_fnn.py

# Other architectures
python train_dcnn.py     # Deep Convolutional Generator
python train_aenn.py     # Autoencoder + NN
python train_gan.py      # GAN
python train_nn.py       # Basic NN
```

### Key Parameters

Edit the top of `train_pifnn.py`:

```python
learning_rate = 5e-4
batch_size = 128
epoch = 5
adapt_mode = 'gradnorm'          # 'gradnorm' | 'fixed' | 'dynamic'
gradnorm_update_freq = 100       # weight update frequency (steps)
gradnorm_alpha = 0.1             # GradNorm exponent
gradnorm_min_weight = 1e-4
gradnorm_max_weight = 1e16
```

- `adapt_mode='gradnorm'`: dynamically balances data / mass / energy / IC / BC loss terms via gradient norm matching.
- `adapt_mode='fixed'`: uses constant weights `initial_alpha/beta/gamma/delta`.

### Output

Trained checkpoints are saved to `./result/` (e.g. `pifnn_*.pth`).

## Project Structure

```
PINN/
├── model/                 # Network definitions (FNN, DCNN, AE, LSTM, ...)
├── dataset/               # Dataset loaders and data generation
├── data/fnn/              # Data directory (train/valid/test)
├── train_pifnn.py         # Main PINN training script
├── train_fnn.py           # FNN training
├── train_dcnn.py          # DCNN training
├── train_aenn.py          # Autoencoder + NN
├── train_gan.py           # GAN
├── train_nn.py            # Basic NN
├── generate_boundary.py   # Boundary condition & gradient utilities
├── generate_tempfile.py   # Export temperature field (GMT format)
└── plot1NN.sh             # GMT plotting script
```

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
