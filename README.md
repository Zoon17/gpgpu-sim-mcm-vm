# gpgpu-sim-mcm-vm

This repository provides the GPGPU-Sim framework extended with integrated Multi-Chip Module 
(MCM) functionality and a robust Virtual Memory (VM) subsystem.

This distribution focuses on the core functional baseline used in the research below.

If you use or build on this, please cite:

> **Junhyeok Park, Sungbin Jang, Osang Kwon, Yongho Lee, and Seokin Hong**, **Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs**, *MICRO 2025*.

```bibtex
@inproceedings{park2025leveraging,
  title={{Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs}},
  author={Park, Junhyeok and Jang, Sungbin and Kwon, Osang and Lee, Yongho and Hong, Seokin},
  booktitle={Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages={1040--1057},
  year={2025},
  url={https://doi.org/10.1145/3725843.3756090}
}
```

If you have any question, please email me. Contact information is available on [my GitHub Profile](https://github.com/Zoon17).

## Feature

### Virtual Memory (VM) subsystem

Portions of the initial logic for virtual memory support were referenced from
[Mosaic (MICRO'17), MASK (ASPLOS'18)](https://github.com/CMU-SAFARI/Mosaic), and [DWS (HPCA'21)](https://github.com/csl-iisc/DWS-HPCA2021).
These components have been modified, optimized, and extended to improve functionality.

1. **Memory Mapping:** Supports 4KB, 64KB, and 2MB page sizes via a GPU Page Table.
2. **Address Translation:** Implements hierarchical L1/L2 TLBs and hardware Page Table Walkers with Page Walk Caches.

### Multi-Chip Module (MCM)
1. **MCM Architecture:** Supports multi-chip configurations with a inter-chip interconnection network.
2. **Locality Optimization:** Implements chiplet-aware CTA (Thread Block) scheduling and memory mapping.
3. **MCM-Extended VM:** Features a virtual memory subsystem scaled for multi-module GPU architectures.
4. **Remote Data Caching:** Supports caching mechanisms for data accessed across chiplet boundaries.

## Copyright

> **Copyright (c) 2023-2026 Junhyeok Park. All rights reserved.**

Please refer to the copyright notice in the `COPYRIGHT` file in the root of this repository for detailed licensing information.

## Building

This framework has been tested on the following systems:
* **Ubuntu 22.04.5 LTS**
	* **Compiler:** `GCC/G++ 9.5`
	* **Toolchain:** `CUDA 11.7`

To build the framework, add the following lines to your `~/.bashrc` file

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-<version>
export CC=<your-gcc-path>
export CXX=<your-g++-path>

# Example:
# export CUDA_INSTALL_PATH=/usr/local/cuda-11.7
# export CC=/usr/bin/gcc-9
# export CXX=/usr/bin/g++-9
```

After updating your environment variables, run the following commands:

```
source setup_environment
make -j$(nproc)
```
If the build fails, please verify that all dependencies are installed as specified in [the original GPGPU-Sim documentation](https://github.com/accel-sim/gpgpu-sim_distribution). 

Once built successfully, the execution process remains identical to the original GPGPU-Sim framework.
For detailed instructions on running simulations and configuring benchmarks, please refer to the original GPGPU-Sim documentation.

A manual for experiment configurations can be found **[here](configs/manual.md)**.