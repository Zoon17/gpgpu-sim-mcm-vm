# Manual for Experiments

For experiments, please use gpgpusim.config in [the "mcm-tested-cfgs" folder](mcm-tested-cfgs) (other configurations are deprecated)

## Virtual Memory Settings

These are the key options for the virtual memory subsystem. For detailed information on other configurations, please refer to the source code.

```bash
# Virtual Memory Configuration Modes:
# 0 : Baseline
# 1 : Ideal TLB (No translation overhead)
# 2 : Ideal L2 TLB
# 3 : Ideal Page Walk (All page walks hit in the L2 cache)
-vm_config 0  # Set to 1 to bypass address translation simulation

# Page Size Configuration:
# Defines the page size for memory mapping.
# 1 = 4KB, 16 = 64KB, 512 = 2MB (Default: 512)
-set_page_size 512
```

## Baseline MCM Setting (Non-Locality-Aware)

To simulate a baseline MCM GPU where the CTA scheduling policy and memory mapping remain similar to those of a non-MCM GPU, please update the configurations in the `gpgpusim.config` file as follows:

```bash
-mcm_cta_schedule 0
-mcm_data_schedule 2
-gpgpu_memory_partition_indexing 6
```

## Locality-Aware MCM Setting

To simulate a locality-aware MCM GPU, please update the parameters in the `gpgpusim.config` file as follows:

> ⚠️ The policies are **independently implemented** based on the designs proposed in the cited research. While every effort has been made to follow the original methodologies accurately, this implementation may contain discrepancies or unintentional errors.


### Static Approaches

These configurations implement locality-aware policies based on the concepts presented in **[NUMA-aware GPUs (MICRO'17)](https://dl.acm.org/doi/10.1145/3123939.3124534)** and **[LADM (MICRO'20)](https://ieeexplore.ieee.org/document/9251964)**.

The Kernel-wide scheduling option provides a simpler approach that typically performs well for regular GPU kernels. Other options, such as Align-aware, Column-based, and Row-based scheduling, require specific granularity inputs (e.g., the column size of the data structure for column-based mapping).

```bash
-mcm_cta_schedule 2
-mcm_data_schedule 2
-gpgpu_memory_partition_indexing 5
```

### Dynamic Approaches


These configurations implement a locality-aware policy based on the concepts presented in **[MCM-GPU (ISCA'17)](https://dl.acm.org/doi/10.1145/3079856.3080231)**.

The policy places data near the chiplet that initially requests it (first-touch-based). While this differs from conventional GPU memory mapping where all required data is copied before kernel execution, it performs consistently well across both regular and irregular GPU kernels.

```bash
-mcm_cta_schedule 1
-mcm_data_schedule 1
-gpgpu_memory_partition_indexing 5
```

## Scaling the Number of Chiplets

Use the following configurations to adjust the number of chiplets for the simulation.

> ⚠️ Increasing the chiplet count significantly may lead to deadlocks in the inter-chiplet interconnection network due to high volumes of remote memory access.

### 2-Chiplet Configuration

```bash
-chiplet_num 2
-gpgpu_n_clusters 128
-total_sm_num 128
-gpgpu_n_mem 32
-DRAM_size 8589934592  ## 8GB
```

### 8-Chiplet Configuration

```bash
-chiplet_num 8
-gpgpu_n_clusters 512
-total_sm_num 512
-gpgpu_n_mem 128
-DRAM_size 34359738368  ## 32GB
```

### 8-Chiplet Configuration

```bash
-chiplet_num 16
-gpgpu_n_clusters 1024
-total_sm_num 1024
-gpgpu_n_mem 256
-DRAM_size 68719476736  ## 64GB
```