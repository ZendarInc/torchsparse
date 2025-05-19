# TorchSparse

TorchSparse is a high-performance neural network library for point cloud processing.

---

# TorchSparse for Zendar

This repository is used to build custom TorchSparse wheels that depend on specific versions of PyTorch. Updates are made only when a new PyTorch version is required—typically driven by CUDA driver updates at Zendar.

When CUDA is updated, we often need to update PyTorch to a version that supports the new CUDA version. In turn, this means TorchSparse must be rebuilt against the updated PyTorch. This is often as simple as updating the dependency, though in some cases minor source code changes may be required to accommodate changes in the PyTorch interface.

This guide outlines how to build TorchSparse wheels, particularly when upgrading to a newer PyTorch version.

---

## 🔧 Building TorchSparse

### 1. Install PDM (Python Development Master)

TorchSparse builds require [PDM](https://pdm-project.org/). To install it:

```bash
curl -sSL -o install-pdm.py https://raw.githubusercontent.com/pdm-project/pdm/2.5.3/install-pdm.py
python3 install-pdm.py --version 2.5.3 --path /tmp
```

> This installs PDM in `/tmp/bin/pdm`.

---

### 2. Create a New Branch

Create a new branch off of `zendar-main` for your changes:

```bash
git checkout -b build/torchsparse-new-version zendar-main
```

---

### 3. Update Dependencies

Update `pyproject.toml` if dependencies have changed:

- Modify the `dependencies` section.
- Also update the `[build-system] requires` section to reflect the correct PyTorch version.

---

### 4. Build the Project (Editable Mode)

Set the correct CUDA environment and run the build:

```bash
export CUDA_PATH=/usr/local/cuda # Make sure this is the correct CUDA version for PyTorch
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64

/tmp/bin/pdm sync --clean
```

#### Troubleshooting

If the command fails, consider the following:

- Ensure only one CUDA installation exists on your machine.
- Complete the CUDA post-installation steps. For example:

```bash
# Replace 12.8 with your CUDA version
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

- Re-run the build with verbosity enabled to help diagnose issues:

```bash
/tmp/bin/pdm sync --clean -v
```

- If you encounter `nvcc` errors, check that the correct compiler is being used.
- Some source-level changes may be needed if the TorchSparse code is not compatible with the new PyTorch interface. These changes are usually minor.

---

### 5. Update Version

Update the TorchSparse version in `torchsparse/version.py`.

---

### 6. Build the Wheel

Build the new wheels:

```bash
/tmp/bin/pdm build
```

---

### 7. Submit Changes

- Open a Pull Request targeting `zendar-main`.
- After review and approval, merge the PR.

---

### 8. Create a Release

Tag a new release and upload the built wheels under the release assets.

---

## 📦 Integrating with RadarProcessor

We recently switched to a package registry hosted in GCP, so to upload your package to the registry, make sure you are signed in with `gcloud auth application-default login` and then use twine to upload your new release.

```
/tmp/bin/pdm twine upload --repository-url https://us-central1-python.pkg.dev/artifacts-443721/python-packages/ dist/*
```

That’s it! 🎉

### Zendar renames this module Torchsparseplusplus

To aid with integration of the new torchsparse version 2.1.0 into RadarProcessor, the module torchsparse has been renamed torchsparseplusplus. This will be maintained in the branch `zendar-main-tspp`. Torchsparse version 1.4.5 will be maintained on the branch `zendar-main`.


## Installation

TorchSparse depends on the [Google Sparse Hash](https://github.com/sparsehash/sparsehash) library.

* On Ubuntu, it can be installed by

  ```bash
  sudo apt-get install libsparsehash-dev
  ```

* On Mac OS, it can be installed by

  ```bash
  brew install google-sparsehash
  ```

* You can also compile the library locally (if you do not have the sudo permission) and add the library path to the environment variable `CPLUS_INCLUDE_PATH`.

The latest released TorchSparse (v1.4.0) can then be installed by

```bash
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

If you use TorchSparse in your code, please remember to specify the exact version as your dependencies.

## Benchmark

We compare TorchSparse with [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) (where the latency is measured on NVIDIA GTX 1080Ti):

|                          | MinkowskiEngine v0.4.3 | TorchSparse v1.0.0 |
| :----------------------- | :--------------------: | :----------------: |
| MinkUNet18C (MACs / 10)  |        224.7 ms        |      124.3 ms      |
| MinkUNet18C (MACs / 4)   |        244.3 ms        |      160.9 ms      |
| MinkUNet18C (MACs / 2.5) |        269.6 ms        |      214.3 ms      |
| MinkUNet18C              |        323.5 ms        |      294.0 ms      |

## Getting Started

### Sparse Tensor

Sparse tensor (`SparseTensor`) is the main data structure for point cloud, which has two data fields:
* Coordinates (`coords`): a 2D integer tensor with a shape of N x 4, where the first three dimensions correspond to quantized x, y, z coordinates, and the last dimension denotes the batch index.
* Features (`feats`): a 2D tensor with a shape of N x C, where C is the number of feature channels.

Most existing datasets provide raw point cloud data with float coordinates. We can use `sparse_quantize` (provided in `torchsparse.utils.quantize`) to voxelize x, y, z coordinates and remove duplicates:

```python
coords -= np.min(coords, axis=0, keepdims=True)
coords, indices = sparse_quantize(coords, voxel_size, return_index=True)
coords = torch.tensor(coords, dtype=torch.int)
feats = torch.tensor(feats[indices], dtype=torch.float)
tensor = SparseTensor(coords=coords, feats=feats)
```

We can then use `sparse_collate_fn` (provided in `torchsparse.utils.collate`) to assemble a batch of `SparseTensor`'s (and add the batch dimension to `coords`). Please refer to [this example](https://github.com/mit-han-lab/torchsparse/blob/dev/pre-commit/examples/example.py) for more details.

### Sparse Neural Network

The neural network interface in TorchSparse is very similar to PyTorch:

```python
from torch import nn
from torchsparse import nn as spnn

model = nn.Sequential(
    spnn.Conv3d(in_channels, out_channels, kernel_size),
    spnn.BatchNorm(out_channels),
    spnn.ReLU(True),
)
```

## Citation

If you use TorchSparse in your research, please use the following BibTeX entry:

```bibtex
@inproceedings{tang2020searching,
  title = {{Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution}},
  author = {Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

## Acknowledgements

TorchSparse is inspired by many existing open-source libraries, including (but not limited to) [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [SECOND](https://github.com/traveller59/second.pytorch) and [SparseConvNet](https://github.com/facebookresearch/SparseConvNet).
