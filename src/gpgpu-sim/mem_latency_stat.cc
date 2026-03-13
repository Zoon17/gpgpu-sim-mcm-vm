// ---------------------------------------------------------------------------
// Modified by: Junhyeok Park (2023-2026)
// Purpose: Add logic for address translation and handling multi-chip module
// (MCM) GPUs
// ---------------------------------------------------------------------------
// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "mem_latency_stat.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx-stats.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "shader.h"
#include "stat-tool.h"
#include "visualizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"

memory_stats_t::memory_stats_t(unsigned n_shader,
                               const shader_core_config *shader_config,
                               const memory_config *mem_config,
                               const class gpgpu_sim *gpu) {
  assert(mem_config->m_valid);
  assert(shader_config->m_valid);

  unsigned i, j;

  concurrent_row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  num_activates =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  max_conc_access2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  max_servicetime2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));

  for (unsigned i = 0; i < mem_config->m_n_mem; i++) {
    concurrent_row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    num_activates[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    max_conc_access2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    max_servicetime2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
  }

  m_n_shader = n_shader;
  m_memory_config = mem_config;
  m_gpu = gpu;
  total_n_access = 0;
  total_n_reads = 0;
  total_n_writes = 0;
  max_mrq_latency = 0;
  max_dq_latency = 0;
  max_mf_latency = 0;
  max_icnt2mem_latency = 0;
  max_icnt2sh_latency = 0;
  tot_icnt2mem_latency = 0;
  tot_icnt2sh_latency = 0;
  tot_mrq_num = 0;
  tot_mrq_latency = 0;
  memset(mrq_lat_table, 0, sizeof(unsigned) * 32);
  memset(dq_lat_table, 0, sizeof(unsigned) * 32);
  memset(mf_lat_table, 0, sizeof(unsigned) * 32);
  memset(icnt2mem_lat_table, 0, sizeof(unsigned) * 24);
  memset(icnt2sh_lat_table, 0, sizeof(unsigned) * 24);
  memset(mf_lat_pw_table, 0, sizeof(unsigned) * 32);
  mf_num_lat_pw = 0;
  max_warps =
      n_shader *
      (shader_config->n_thread_per_shader / shader_config->warp_size + 1);
  mf_tot_lat_pw = 0;  // total latency summed up per window. divide by
                      // mf_num_lat_pw to obtain average latency Per Window
  mf_total_lat = 0;
  num_mfs = 0;
  printf("*** Initializing Memory Statistics ***\n");
  totalbankreads =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  totalbankwrites =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  totalbankaccesses =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  mf_total_lat_table = (unsigned long long int **)calloc(
      mem_config->m_n_mem, sizeof(unsigned long long *));
  mf_max_lat_table =
      (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
  bankreads = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  bankwrites = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  num_MCBs_accessed = (unsigned int *)calloc(
      mem_config->m_n_mem * mem_config->nbk, sizeof(unsigned int));
  if (mem_config->gpgpu_frfcfs_dram_sched_queue_size) {
    position_of_mrq_chosen = (unsigned int *)calloc(
        mem_config->gpgpu_frfcfs_dram_sched_queue_size, sizeof(unsigned int));
  } else
    position_of_mrq_chosen = (unsigned int *)calloc(1024, sizeof(unsigned int));
  for (i = 0; i < n_shader; i++) {
    bankreads[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    bankwrites[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    for (j = 0; j < mem_config->m_n_mem; j++) {
      bankreads[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
      bankwrites[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    }
  }

  for (i = 0; i < mem_config->m_n_mem; i++) {
    totalbankreads[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    totalbankwrites[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    totalbankaccesses[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    mf_total_lat_table[i] = (unsigned long long int *)calloc(
        mem_config->nbk, sizeof(unsigned long long int));
    mf_max_lat_table[i] = (unsigned *)calloc(mem_config->nbk, sizeof(unsigned));
  }

  mem_access_type_stats =
      (unsigned ***)malloc(NUM_MEM_ACCESS_TYPE * sizeof(unsigned **));
  for (i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    int j;
    mem_access_type_stats[i] =
        (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
    for (j = 0; (unsigned)j < mem_config->m_n_mem; j++) {
      mem_access_type_stats[i][j] =
          (unsigned *)calloc((mem_config->nbk + 1), sizeof(unsigned *));
    }
  }

  // AerialVision L2 stats
  L2_read_miss = 0;
  L2_write_miss = 0;
  L2_read_hit = 0;
  L2_write_hit = 0;

  /****************************************************/
  pt_space_size = 0;

  for (int i = 0; i < 10; i++) {
      tlb_level_hits[i] = 0;
      tlb_level_misses[i] = 0;
      tlb_level_fails[i] = 0;
  }

  //Level 2 TLB stats
  l2_tlb_tot_hits = 0;
  l2_tlb_tot_misses = 0;
  l2_tlb_tot_accesses = 0;
  l2_tlb_tot_mshr_hits = 0;
  l2_tlb_tot_mshr_fails = 0;
  l2_tlb_tot_backpressure_fails = 0;
  l2_tlb_tot_backpressure_stalls = 0;

  pwq_tot_lat = 0;
  pw_tot_lat = 0;
  pw_tot_num = 0;

  pwc_tot_accesses = 0;
  pwc_tot_hits = 0;
  pwc_tot_misses = 0;

  for (int i = 0; i < 10; i++) {
      pwc_tot_addr_lvl_accesses[i] = 0;
      pwc_tot_addr_lvl_hits[i] = 0;
      pwc_tot_addr_lvl_misses[i] = 0;
  }

   for (unsigned i = 0; i < 100; i++) {
      tot_access_malloc.insert(std::pair<unsigned, uint64_t>(i, 0));
      tot_local_access_malloc.insert(std::pair<unsigned, uint64_t>(i, 0));
      tot_remote_access_malloc.insert(std::pair<unsigned, uint64_t>(i, 0));
   }

   for (unsigned i = 1; i <= 512; i = i * 2) {  // now, set for 4KB ~ 2MB
      l1_hit_per_size.insert(std::pair<unsigned, uint64_t>
          (i, 0));
      l2_hit_per_size.insert(std::pair<unsigned, uint64_t>
          (i, 0));
   }
}

// record the total latency
unsigned memory_stats_t::memlatstat_done(mem_fetch *mf) {
  unsigned mf_latency = (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) - mf->get_timestamp();

  //total latency of page walk is captured by the last page walk return time
  //minus the timestamp of the memory fetch
  mf_num_lat_pw++;
  mf_tot_lat_pw += mf_latency;
  unsigned idx = LOGB2(mf_latency);
  assert(idx < 32);
  mf_lat_table[idx]++;
  shader_mem_lat_log(mf->get_sid(), mf_latency);
  mf_total_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] +=
      mf_latency;
  if (mf_latency > max_mf_latency) max_mf_latency = mf_latency;

  return mf_latency;
}

void memory_stats_t::memlatstat_read_done(mem_fetch *mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    unsigned mf_latency = memlatstat_done(mf);
    if (mf_latency >
        mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk])
      mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] =
          mf_latency;
    unsigned icnt2sh_latency;
    icnt2sh_latency = (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) -
                      mf->get_return_timestamp();
    tot_icnt2sh_latency += icnt2sh_latency;
    icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
    if (icnt2sh_latency > max_icnt2sh_latency)
      max_icnt2sh_latency = icnt2sh_latency;
  }
}

void memory_stats_t::memlatstat_dram_access(mem_fetch *mf) {
  unsigned dram_id = mf->get_tlx_addr().chip;
  unsigned bank = mf->get_tlx_addr().bk;
  if (m_memory_config->gpgpu_memlatency_stat) {
    if (mf->get_is_write()) {
      if (mf->get_sid() < m_n_shader) {  // do not count L2_writebacks here
        bankwrites[mf->get_sid()][dram_id][bank]++;
        shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'w');
      }
      totalbankwrites[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
    } else {
      bankreads[mf->get_sid()][dram_id][bank]++;
      shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'r');
      totalbankreads[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
    }
    mem_access_type_stats[mf->get_access_type()][dram_id][bank] +=
        ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
  }

  if (mf->get_pc() != (unsigned)-1)
    m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_dram_traffic(
        mf->get_pc(), mf->get_data_size());
}

void memory_stats_t::memlatstat_icnt2mem_pop(mem_fetch *mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    unsigned icnt2mem_latency;
    icnt2mem_latency =
        (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) - mf->get_timestamp();
    tot_icnt2mem_latency += icnt2mem_latency;
    icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
    if (icnt2mem_latency > max_icnt2mem_latency)
      max_icnt2mem_latency = icnt2mem_latency;
  }
}

void memory_stats_t::memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk) {
  unsigned i, j, k, l, m;
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;

  if (m_memory_config->gpgpu_memlatency_stat) {
    printf("maxmflatency = %d \n", max_mf_latency);
    printf("max_icnt2mem_latency = %d \n", max_icnt2mem_latency);
    printf("maxmrqlatency = %d \n", max_mrq_latency);
    // printf("maxdqlatency = %d \n", max_dq_latency);
    printf("max_icnt2sh_latency = %d \n", max_icnt2sh_latency);
    if (num_mfs) {
      printf("averagemflatency = %lld \n", mf_total_lat / num_mfs);
      printf("avg_icnt2mem_latency = %lld \n", tot_icnt2mem_latency / num_mfs);
      if (tot_mrq_num)
        printf("avg_mrq_latency = %lld \n", tot_mrq_latency / tot_mrq_num);

      printf("avg_icnt2sh_latency = %lld \n", tot_icnt2sh_latency / num_mfs);
    }
    printf("mrq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mrq_lat_table[i]);
    }
    printf("\n");
    printf("dq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", dq_lat_table[i]);
    }
    printf("\n");
    printf("mf_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_table[i]);
    }
    printf("\n");
    printf("icnt2mem_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2mem_lat_table[i]);
    }
    printf("\n");
    printf("icnt2sh_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2sh_lat_table[i]);
    }
    printf("\n");
    printf("mf_lat_pw_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_pw_table[i]);
    }
    printf("\n");

    /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
    printf("maximum concurrent accesses to same row:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%9d ", max_conc_access2samerow[i][j]);
      }
      printf("\n");
    }

    /*MAXIMUM SERVICE TIME TO SAME ROW*/
    printf("maximum service time to same row:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%9d ", max_servicetime2samerow[i][j]);
      }
      printf("\n");
    }

    /*AVERAGE ROW ACCESSES PER ACTIVATE*/
    int total_row_accesses = 0;
    int total_num_activates = 0;
    printf("average row accesses per activate:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        total_row_accesses += row_access[i][j];
        total_num_activates += num_activates[i][j];
        printf("%9f ", (float)row_access[i][j] / num_activates[i][j]);
      }
      printf("\n");
    }
    printf("average row locality = %d/%d = %f\n", total_row_accesses,
           total_num_activates,
           (float)total_row_accesses / total_num_activates);
    /*MEMORY ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total memory accesses made:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankaccesses[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total accesses: %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*READ ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total read accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankreads[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram reads = %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*WRITE ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total write accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankwrites[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram writes = %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*AVERAGE MF LATENCY PER BANK*/
    printf("average mf latency per bank:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        k = totalbankwrites[i][j] + totalbankreads[i][j];
        if (k)
          printf("%10lld", mf_total_lat_table[i][j] / k);
        else
          printf("    none  ");
      }
      printf("\n");
    }

    /*MAXIMUM MF LATENCY PER BANK*/
    printf("maximum mf latency per bank:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%10d", mf_max_lat_table[i][j]);
      }
      printf("\n");
    }
  }

  if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
    printf(
        "\nNumber of Memory Banks Accessed per Memory Operation per Warp (from "
        "0):\n");
    unsigned long long accum_MCBs_accessed = 0;
    unsigned long long tot_mem_ops_per_warp = 0;
    for (i = 0; i < n_mem * gpu_mem_n_bk; i++) {
      accum_MCBs_accessed += i * num_MCBs_accessed[i];
      tot_mem_ops_per_warp += num_MCBs_accessed[i];
      printf("%d\t", num_MCBs_accessed[i]);
    }

    printf(
        "\nAverage # of Memory Banks Accessed per Memory Operation per "
        "Warp=%f\n",
        (float)accum_MCBs_accessed / tot_mem_ops_per_warp);

    // printf("\nAverage Difference Between First and Last Response from Memory
    // System per warp = ");

    printf("\nposition of mrq chosen\n");

    if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
      j = 1024;
    else
      j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
    k = 0;
    l = 0;
    for (i = 0; i < j; i++) {
      printf("%d\t", position_of_mrq_chosen[i]);
      k += position_of_mrq_chosen[i];
      l += i * position_of_mrq_chosen[i];
    }
    printf("\n");
    printf("\naverage position of mrq chosen = %f\n", (float)l / k);
  }
}

void memory_stats_t::memlatstat_print_file(FILE *outputfile, unsigned n_mem, unsigned gpu_mem_n_bk) {
    unsigned i, j, k, l, m;
    unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
            min_chip_accesses;

    if (m_memory_config->gpgpu_memlatency_stat) {
        fprintf(outputfile, "maxmflatency = %d \n", max_mf_latency);
        fprintf(outputfile, "max_icnt2mem_latency = %d \n", max_icnt2mem_latency);
        fprintf(outputfile, "maxmrqlatency = %d \n", max_mrq_latency);
        // printf("maxdqlatency = %d \n", max_dq_latency);
        fprintf(outputfile, "max_icnt2sh_latency = %d \n", max_icnt2sh_latency);
        if (num_mfs) {
            fprintf(outputfile, "averagemflatency = %lld \n", mf_total_lat / num_mfs);
            fprintf(outputfile, "avg_icnt2mem_latency = %lld \n", tot_icnt2mem_latency / num_mfs);
            if (tot_mrq_num)
                fprintf(outputfile, "avg_mrq_latency = %lld \n", tot_mrq_latency / tot_mrq_num);

            fprintf(outputfile, "avg_icnt2sh_latency = %lld \n", tot_icnt2sh_latency / num_mfs);
        }
        fprintf(outputfile, "mrq_lat_table:");
        for (i = 0; i < 32; i++) {
            fprintf(outputfile, "%d \t", mrq_lat_table[i]);
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "dq_lat_table:");
        for (i = 0; i < 32; i++) {
            fprintf(outputfile, "%d \t", dq_lat_table[i]);
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "mf_lat_table:");
        for (i = 0; i < 32; i++) {
            fprintf(outputfile, "%d \t", mf_lat_table[i]);
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "icnt2mem_lat_table:");
        for (i = 0; i < 24; i++) {
            fprintf(outputfile, "%d \t", icnt2mem_lat_table[i]);
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "icnt2sh_lat_table:");
        for (i = 0; i < 24; i++) {
            fprintf(outputfile, "%d \t", icnt2sh_lat_table[i]);
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "mf_lat_pw_table:");
        for (i = 0; i < 32; i++) {
            fprintf(outputfile, "%d \t", mf_lat_pw_table[i]);
        }
        fprintf(outputfile, "\n");

        /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
        fprintf(outputfile, "maximum concurrent accesses to same row:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                fprintf(outputfile, "%9d ", max_conc_access2samerow[i][j]);
            }
            fprintf(outputfile, "\n");
        }

        /*MAXIMUM SERVICE TIME TO SAME ROW*/
        fprintf(outputfile, "maximum service time to same row:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                fprintf(outputfile, "%9d ", max_servicetime2samerow[i][j]);
            }
            fprintf(outputfile, "\n");
        }

        /*AVERAGE ROW ACCESSES PER ACTIVATE*/
        int total_row_accesses = 0;
        int total_num_activates = 0;
        fprintf(outputfile, "average row accesses per activate:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                total_row_accesses += row_access[i][j];
                total_num_activates += num_activates[i][j];
                fprintf(outputfile, "%9f ", (float)row_access[i][j] / num_activates[i][j]);
            }
            fprintf(outputfile, "\n");
        }
        fprintf(outputfile, "average row locality = %d/%d = %f\n", total_row_accesses,
               total_num_activates,
               (float)total_row_accesses / total_num_activates);
        /*MEMORY ACCESSES*/
        k = 0;
        l = 0;
        m = 0;
        max_bank_accesses = 0;
        max_chip_accesses = 0;
        min_bank_accesses = 0xFFFFFFFF;
        min_chip_accesses = 0xFFFFFFFF;
        fprintf(outputfile, "number of total memory accesses made:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                l = totalbankaccesses[i][j];
                if (l < min_bank_accesses) min_bank_accesses = l;
                if (l > max_bank_accesses) max_bank_accesses = l;
                k += l;
                m += l;
                fprintf(outputfile, "%9d ", l);
            }
            if (m < min_chip_accesses) min_chip_accesses = m;
            if (m > max_chip_accesses) max_chip_accesses = m;
            m = 0;
            fprintf(outputfile, "\n");
        }
        fprintf(outputfile, "total accesses: %d\n", k);
        if (min_bank_accesses)
            fprintf(outputfile, "bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
                   (float)max_bank_accesses / min_bank_accesses);
        else
            fprintf(outputfile, "min_bank_accesses = 0!\n");
        if (min_chip_accesses)
            fprintf(outputfile, "chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
                   (float)max_chip_accesses / min_chip_accesses);
        else
            fprintf(outputfile, "min_chip_accesses = 0!\n");

        /*READ ACCESSES*/
        k = 0;
        l = 0;
        m = 0;
        max_bank_accesses = 0;
        max_chip_accesses = 0;
        min_bank_accesses = 0xFFFFFFFF;
        min_chip_accesses = 0xFFFFFFFF;
        fprintf(outputfile, "number of total read accesses:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                l = totalbankreads[i][j];
                if (l < min_bank_accesses) min_bank_accesses = l;
                if (l > max_bank_accesses) max_bank_accesses = l;
                k += l;
                m += l;
                fprintf(outputfile, "%9d ", l);
            }
            if (m < min_chip_accesses) min_chip_accesses = m;
            if (m > max_chip_accesses) max_chip_accesses = m;
            m = 0;
            fprintf(outputfile, "\n");
        }
        fprintf(outputfile, "total dram reads = %d\n", k);
        if (min_bank_accesses)
            fprintf(outputfile, "bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
                   (float)max_bank_accesses / min_bank_accesses);
        else
            fprintf(outputfile, "min_bank_accesses = 0!\n");
        if (min_chip_accesses)
            fprintf(outputfile, "chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
                   (float)max_chip_accesses / min_chip_accesses);
        else
            fprintf(outputfile, "min_chip_accesses = 0!\n");

        /*WRITE ACCESSES*/
        k = 0;
        l = 0;
        m = 0;
        max_bank_accesses = 0;
        max_chip_accesses = 0;
        min_bank_accesses = 0xFFFFFFFF;
        min_chip_accesses = 0xFFFFFFFF;
        fprintf(outputfile, "number of total write accesses:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                l = totalbankwrites[i][j];
                if (l < min_bank_accesses) min_bank_accesses = l;
                if (l > max_bank_accesses) max_bank_accesses = l;
                k += l;
                m += l;
                fprintf(outputfile, "%9d ", l);
            }
            if (m < min_chip_accesses) min_chip_accesses = m;
            if (m > max_chip_accesses) max_chip_accesses = m;
            m = 0;
            fprintf(outputfile, "\n");
        }
        fprintf(outputfile, "total dram writes = %d\n", k);
        if (min_bank_accesses)
            fprintf(outputfile, "bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
                   (float)max_bank_accesses / min_bank_accesses);
        else
            fprintf(outputfile, "min_bank_accesses = 0!\n");
        if (min_chip_accesses)
            fprintf(outputfile, "chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
                   (float)max_chip_accesses / min_chip_accesses);
        else
            fprintf(outputfile, "min_chip_accesses = 0!\n");

        /*AVERAGE MF LATENCY PER BANK*/
        fprintf(outputfile, "average mf latency per bank:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                k = totalbankwrites[i][j] + totalbankreads[i][j];
                if (k)
                    fprintf(outputfile, "%10lld", mf_total_lat_table[i][j] / k);
                else
                    fprintf(outputfile, "    none  ");
            }
            fprintf(outputfile, "\n");
        }

        /*MAXIMUM MF LATENCY PER BANK*/
        fprintf(outputfile, "maximum mf latency per bank:\n");
        for (i = 0; i < n_mem; i++) {
            fprintf(outputfile, "dram[%d]: ", i);
            for (j = 0; j < gpu_mem_n_bk; j++) {
                fprintf(outputfile, "%10d", mf_max_lat_table[i][j]);
            }
            fprintf(outputfile, "\n");
        }
    }

    if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
        fprintf(outputfile,
                "\nNumber of Memory Banks Accessed per Memory Operation per Warp (from "
                "0):\n");
        unsigned long long accum_MCBs_accessed = 0;
        unsigned long long tot_mem_ops_per_warp = 0;
        for (i = 0; i < n_mem * gpu_mem_n_bk; i++) {
            accum_MCBs_accessed += i * num_MCBs_accessed[i];
            tot_mem_ops_per_warp += num_MCBs_accessed[i];
            fprintf(outputfile, "%d\t", num_MCBs_accessed[i]);
        }

        fprintf(outputfile,
                "\nAverage # of Memory Banks Accessed per Memory Operation per "
                "Warp=%f\n",
                (float)accum_MCBs_accessed / tot_mem_ops_per_warp);

        // printf("\nAverage Difference Between First and Last Response from Memory
        // System per warp = ");

        fprintf(outputfile, "\nposition of mrq chosen\n");

        if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
            j = 1024;
        else
            j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
        k = 0;
        l = 0;
        for (i = 0; i < j; i++) {
            fprintf(outputfile, "%d\t", position_of_mrq_chosen[i]);
            k += position_of_mrq_chosen[i];
            l += i * position_of_mrq_chosen[i];
        }
        fprintf(outputfile, "\n");
        fprintf(outputfile, "\naverage position of mrq chosen = %f\n", (float)l / k);
    }
}

inline uint64_t memory_stats_t::div(uint64_t x, uint64_t y) {
    return y ? (uint64_t)x / y : 0;
}

inline float memory_stats_t::div_float(uint64_t x, uint64_t y) {
    return y ? x / static_cast<double>(y) : 0;
}

inline void memory_stats_t::newline(FILE *file) {
    fprintf(file, "\n");
}

inline void memory_stats_t::print_stat(const char *stat_name, uint64_t val, FILE *file) {
    fprintf(file, "%-40s = %-10lu\n", stat_name, val);
}

inline void memory_stats_t::print_stat_float(const char *stat_name, float val, FILE *file) {
    fprintf(file, "%-40s = %-10f\n", stat_name, val);
}

void memory_stats_t::print_app_stat(const char *stat_name, uint64_t arr_size, uint64_t *arr_1, uint64_t n, uint64_t *arr_2, FILE *file) {
    fprintf(file, "%-40s = {\t", stat_name);
    if (n == 0 && arr_2 == nullptr)
        for (uint64_t i = 1; i <= arr_size; fprintf(file, "%lu\t", arr_1[i++]))
            ;
    else {
        double_t stat_arr[arr_size];
        if (n > 0)
            for (uint64_t i = 1; i <= arr_size; stat_arr[i] = div(arr_1[i], n), i++)
                ;
        else
            for (uint64_t i = 1; i <= arr_size; stat_arr[i] = div(arr_1[i], arr_2[i]), i++)
                ;
        for (uint64_t i = 1; i <= arr_size; fprintf(file, "%-10lf\t", stat_arr[i++]))
            ;
        ;
    }
    fprintf(file, "}\n");
}

void memory_stats_t::print_essential(FILE *fout) {
    print_tlb_stat(fout);
    print_mcm_stat(fout);
}

void memory_stats_t::print_tlb_stat(FILE *fout) {
    newline(fout);
    newline(fout);
    fprintf(fout, "***Page walk stats***\n\n");
    fprintf(fout, "page walk per-level l2_cache accesses = {");
    for (int i = 1; i < 10; i++)
        fprintf(fout, "%lu, ", tlb_level_accesses[i]);
    fprintf(fout, "}\n");

    fprintf(fout, "page walk per-level l2_cache hits     = {");
    for (int i = 1; i < 10; i++)
        fprintf(fout, "%lu, ", tlb_level_hits[i]);
    fprintf(fout, "}\n");

    fprintf(fout, "page walk per-level l2_cache misses   = {");
    for (int i = 1; i < 10; i++)
        fprintf(fout, "%lu, ", tlb_level_misses[i]);
    fprintf(fout, "}\n");

    fprintf(fout, "page walk per-level l2_cache fails    = {");
    for (int i = 1; i < 10; i++)
        fprintf(fout, "%lu, ", tlb_level_fails[i]);
    fprintf(fout, "}\n");

    fprintf(fout, "page walk per-level l2_cache hit_rate = {");
    for (int i = 1; i < 10; i++)
        fprintf(fout, "%f, ", static_cast<float>(tlb_level_hits[i]) / static_cast<float>(tlb_level_hits[i] + tlb_level_misses[i]));
    fprintf(fout, "}\n");
    newline(fout);

    print_stat("page walks total num", pw_tot_num, fout);
    print_stat("page walks tot latency", pw_tot_lat, fout);
    print_stat("page walks total waiting time in queue", pwq_tot_lat, fout);
    newline(fout);

    print_stat_float("page walks avg latency", div_float((pw_tot_lat - pwq_tot_lat), pw_tot_num), fout);
    print_stat("page walks avg waiting time in queue", div(pwq_tot_lat, pw_tot_num), fout);
    newline(fout);
    print_stat("page walks cache access", pw_tot_access, fout);
    print_stat("page walks cache local access", pw_local_access, fout);
    print_stat("page walks cache remote access", pw_remote_access, fout);
    newline(fout);
    newline(fout);

    fprintf(fout, "***Page walk cache stats***\n\n");
    print_stat("pw cache accesses", pwc_tot_accesses, fout);
    print_stat("pw cache hits", pwc_tot_hits, fout);
    print_stat("pw cache misses", pwc_tot_misses, fout);
    newline(fout);

    uint64_t n_addr_lvls = m_memory_config->tlb_levels;
    print_app_stat("pw cache tot per-lvl accesses", n_addr_lvls, pwc_tot_addr_lvl_accesses, 0, nullptr, fout);
    print_app_stat("pw cache tot per-lvl hits", n_addr_lvls, pwc_tot_addr_lvl_hits, 0, nullptr, fout);
    print_app_stat("pw cache tot per-lvl misses", n_addr_lvls, pwc_tot_addr_lvl_misses, 0, nullptr, fout);
    newline(fout);
    newline(fout);

    fprintf(fout, "***TLB Stats***\n\n");
    fprintf(fout, "L1 TLB\n");
    print_stat("l1 tlb accesses", l1_tlb_tot_access, fout);
    print_stat("l1 tlb true accesses", (l1_tlb_tot_access - l1_tlb_tot_fail), fout);
    print_stat("l1 tlb hits", l1_tlb_tot_hit, fout);
    for (std::map<unsigned, uint64_t>::iterator l1_iter = l1_hit_per_size.begin(); l1_iter != l1_hit_per_size.end(); ++l1_iter){
        if (l1_iter->second != 0) {
            std::string stat = "l1 tlb hits_" + std::to_string(l1_iter->first * 4) + "KB";
            print_stat(stat.c_str(), l1_iter->second, fout);
        }
    }
    print_stat("l1 tlb misses", l1_tlb_tot_miss, fout);
    print_stat("l1 tlb mshr hits", l1_tlb_tot_hit_reserved, fout);
    print_stat("l1 tlb mshr fails", l1_tlb_tot_fail, fout);
    print_stat("l1 tlb mshr allocate fails", l1_tlb_mshr_allocate_fail, fout);
    print_stat("l1 tlb mshr merge fails", l1_tlb_mshr_merge_fail, fout);
    print_stat("l1 tlb mshr l2 stall", l1_tlb_mshr_l2_stall, fout);
    float l1_tlb_miss_rate = div_float(l1_tlb_tot_miss, (l1_tlb_tot_access - l1_tlb_tot_fail));
    print_stat_float("l1 tlb miss rate", l1_tlb_miss_rate, fout);
    newline(fout);

    fprintf(fout, "L2 TLB\n");
    print_stat("l2 tlb accesses", l2_tlb_tot_accesses, fout);
    print_stat("l2 tlb true accesses", (l2_tlb_tot_accesses - l2_tlb_tot_mshr_fails), fout);
    print_stat("l2 tlb hits", l2_tlb_tot_hits, fout);
    for (std::map<unsigned, uint64_t>::iterator l2_iter = l2_hit_per_size.begin(); l2_iter != l2_hit_per_size.end(); ++l2_iter){
        if (l2_iter->second != 0) {
            std::string stat = "l2 tlb hits_" + std::to_string(l2_iter->first * 4) + "KB";
            print_stat(stat.c_str(), l2_iter->second, fout);
        }
    }
    print_stat("l2 tlb misses", l2_tlb_tot_misses, fout);
    print_stat("l2 tlb mshr hits", l2_tlb_tot_mshr_hits, fout);
    print_stat("l2 tlb mshr fails", l2_tlb_tot_mshr_fails, fout);
    print_stat("l2 tlb mshr allocate fails", l2_tlb_mshr_allocate_fail, fout);
    print_stat("l2 tlb mshr merge fails", l2_tlb_mshr_merge_fail, fout);
    print_stat("l2 tlb mshr pwq full", l2_tlb_mshr_pwq_full, fout);
    print_stat("l2 tlb backpressure fails", l2_tlb_tot_backpressure_fails, fout);
    print_stat("l2 tlb backpressure stalls", l2_tlb_tot_backpressure_stalls, fout);
    float l2_tlb_miss_rate = div_float(l2_tlb_tot_misses, (l2_tlb_tot_accesses - l2_tlb_tot_mshr_fails));
    print_stat_float("l2 tlb miss rate", l2_tlb_miss_rate, fout);
    newline(fout);
}

/*** MCM GPU Stat ***/
void memory_stats_t::print_mcm_stat(FILE *fout) {
    newline(fout);
    newline(fout);
    fprintf(fout, "***MCM GPU Stat***\n\n");

    float remote_ratio = div_float(tot_remote_access, tot_local_access + tot_remote_access);
    print_stat_float("remote access ratio", remote_ratio, fout);
    float pw_remote_ratio = div_float(pw_leaf_remote, pw_leaf_local + pw_leaf_remote);
    print_stat_float("pw remote access ratio", pw_remote_ratio, fout);

    unsigned total_alloc_result;
    std::vector<unsigned> chiplet_alloc_result;
    m_gpu->m_gpu_alloc->copy_alloc_result(total_alloc_result, chiplet_alloc_result);
    newline(fout);
    uint64_t alloc_MB = (static_cast<uint64_t>(total_alloc_result) * static_cast<uint64_t>(4 * 1024)) / static_cast<uint64_t>(1024 * 1024);
    print_stat("total mapped data (MB)", alloc_MB, fout);
    for (unsigned i = 0; i < m_chiplet; i++) {
        uint64_t chiplet_alloc_MB = (static_cast<uint64_t>(chiplet_alloc_result[i]) * static_cast<uint64_t>(4 * 1024)) / static_cast<uint64_t>(1024 * 1024);
        std::string stat = "Chiplet" + std::to_string(i) + " (MB)";
        print_stat(stat.c_str(), chiplet_alloc_MB, fout);
    }
    newline(fout);
    newline(fout);

    fprintf(fout, "***Latency Stat***\n\n");
    float avg_lat        = div_float(data_tot_latency, data_tot_count);
    float avg_tlb_lat    = div_float(tlb_tot_latency, tlb_tot_count);
    float avg_local_lat  = div_float(data_local_tot_latency, data_local_tot_count);
    float avg_remote_lat = div_float(data_remote_tot_latency, data_remote_tot_count);

    print_stat_float("avg latency", avg_lat, fout);
    print_stat_float("avg tlb latency", avg_tlb_lat, fout);
    print_stat_float("avg local latency", avg_local_lat, fout);
    print_stat_float("avg remote latency", avg_remote_lat, fout);
    newline(fout);
    newline(fout);

    fprintf(fout, "***Cache Stat***\n\n");
    print_stat("L1$ access", l1_cache_access, fout);
    print_stat("L1$ hit", l1_cache_hit, fout);
    print_stat("L1$ hit_reserved", l1_cache_hit_reserved, fout);
    print_stat("L1$ miss", l1_cache_miss, fout);
    print_stat("L1$ reserve_fail", l1_cache_reserve_fail, fout);
    float l1_miss_rate = div_float(l1_cache_miss, l1_cache_access);
    print_stat_float("L1$ miss rate", l1_miss_rate, fout);
    newline(fout);
    print_stat("L1$ local access", l1_cache_local_access, fout);
    print_stat("L1$ local hit", l1_cache_local_hit, fout);
    print_stat("L1$ local hit_reserved", l1_cache_local_hit_reserved, fout);
    print_stat("L1$ local miss", l1_cache_local_miss, fout);
    print_stat("L1$ local reserve_fail", l1_cache_local_reserve_fail, fout);
    float l1_local_miss_rate = div_float(l1_cache_local_miss, l1_cache_local_access);
    print_stat_float("L1$ local miss rate", l1_local_miss_rate, fout);
    newline(fout);
    print_stat("L1$ remote access", l1_cache_remote_access, fout);
    print_stat("L1$ remote hit", l1_cache_remote_hit, fout);
    print_stat("L1$ remote hit_reserved", l1_cache_remote_hit_reserved, fout);
    print_stat("L1$ remote miss", l1_cache_remote_miss, fout);
    print_stat("L1$ remote reserve_fail", l1_cache_remote_reserve_fail, fout);
    float l1_remote_miss_rate = div_float(l1_cache_remote_miss, l1_cache_remote_access);
    print_stat_float("L1$ remote miss rate", l1_remote_miss_rate, fout);
    newline(fout);
    newline(fout);
    print_stat("L2$ access", l2_cache_access, fout);
    print_stat("L2$ hit", l2_cache_hit, fout);
    print_stat("L2$ hit_reserved", l2_cache_hit_reserved, fout);
    print_stat("L2$ miss", l2_cache_miss, fout);
    print_stat("L2$ reserve_fail", l2_cache_reserve_fail, fout);
    float l2_miss_rate = div_float(l2_cache_miss, l2_cache_access);
    print_stat_float("L2$ miss rate", l2_miss_rate, fout);
    newline(fout);
    print_stat("L2$ local access", l2_cache_local_access, fout);
    print_stat("L2$ local hit", l2_cache_local_hit, fout);
    print_stat("L2$ local hit_reserved", l2_cache_local_hit_reserved, fout);
    print_stat("L2$ local miss", l2_cache_local_miss, fout);
    print_stat("L2$ local reserve_fail", l2_cache_local_reserve_fail, fout);
    float l2_local_miss_rate = div_float(l2_cache_local_miss, l2_cache_local_access);
    print_stat_float("L2$ local miss rate", l2_local_miss_rate, fout);
    newline(fout);
    print_stat("L2$ remote access", l2_cache_remote_access, fout);
    print_stat("L2$ remote hit", l2_cache_remote_hit, fout);
    print_stat("L2$ remote hit_reserved", l2_cache_remote_hit_reserved, fout);
    print_stat("L2$ remote miss", l2_cache_remote_miss, fout);
    print_stat("L2$ remote reserve_fail", l2_cache_remote_reserve_fail, fout);
    float l2_remote_miss_rate = div_float(l2_cache_remote_miss, l2_cache_remote_access);
    print_stat_float("L2$ remote miss rate", l2_remote_miss_rate, fout);
    newline(fout);
    print_stat("L2$ remote gpm access", l2_cache_remote_transfer_access, fout);
    print_stat("L2$ remote gpm hit", l2_cache_remote_transfer_hit, fout);
    print_stat("L2$ remote gpm hit_reserved", l2_cache_remote_transfer_hit_reserved, fout);
    print_stat("L2$ remote gpm miss", l2_cache_remote_transfer_miss, fout);
    print_stat("L2$ remote gpm reserve_fail", l2_cache_remote_transfer_reserve_fail, fout);
    float l2_remote_gpm_miss_rate = div_float(l2_cache_remote_transfer_miss, l2_cache_remote_transfer_access);
    print_stat_float("L2$ remote gpm miss rate", l2_remote_gpm_miss_rate, fout);
    newline(fout);
}

void memory_stats_t::data_latency_stat(mem_fetch *mf) {
    new_addr_type start_time = mf->get_timestamp();
    new_addr_type data_tot_lat = mf->get_time() - start_time;
    data_tot_latency += data_tot_lat;  data_tot_count++;

    if (mf->get_tlb_time() != static_cast<new_addr_type>(-1)){
        new_addr_type tlb_lat =  mf->get_tlb_time() - start_time;
        tlb_tot_latency += tlb_lat;
        tlb_tot_count++;
    }

    if (mf->get_is_local_access()){
        data_local_tot_latency += data_tot_lat;  data_local_tot_count++;
    } else {
        data_remote_tot_latency += data_tot_lat;  data_remote_tot_count++;
    }
}

void memory_stats_t::mcm_cache_stat(cache_request_status access_status, cache_request_status probe_status,
                                    bool is_partition, mem_fetch *mf) {
    if (!is_partition && access_status != RESERVATION_FAIL) {
        bool is_local = mf->get_is_local_access();
        if (is_local) tot_local_access++;
        else tot_remote_access++;

        if (m_memory_config->enable_remote_debug) {
            if (mf->get_malloc_num() != static_cast<unsigned>(-1)) {
                unsigned malloc_num = mf->get_malloc_num();
                tot_access_malloc[malloc_num] += 1;

                if (is_local) {
                    tot_local_access_malloc[malloc_num] += 1;
                } else {
                    tot_remote_access_malloc[malloc_num] += 1;
                }

                // record write to each malloc
                if (mf->is_write()){
                    write_malloc[malloc_num] = true;
                    tot_write_malloc[malloc_num] = true;
                }
            } else {
                non_malloc_access++;
            }
        }
    }

    if (!is_partition){  // l1 cache stat
        bool is_local = mf->get_is_local_access();
        if (access_status == RESERVATION_FAIL){
            l1_cache_reserve_fail++;
            if (is_local) l1_cache_local_reserve_fail++;
            else l1_cache_remote_reserve_fail++;
            return;
        }
        l1_cache_access++;
        if (is_local) l1_cache_local_access++;
        else l1_cache_remote_access++;

        switch (probe_status) {
            case HIT:
                l1_cache_hit++;
                if (is_local) l1_cache_local_hit++;
                else l1_cache_remote_hit++;
                break;
            case HIT_RESERVED:
                l1_cache_hit_reserved++;
                if (is_local) l1_cache_local_hit_reserved++;
                else l1_cache_remote_hit_reserved++;
                break;
            case SECTOR_MISS:
            case MISS:
                l1_cache_miss++;
                if (is_local) l1_cache_local_miss++;
                else l1_cache_remote_miss++;
                break;
            default:
                assert(0);
        }

    } else {  // l2 cache stat
        bool is_local = mf->get_is_local_access();
        if (access_status == RESERVATION_FAIL){
            l2_cache_reserve_fail++;
            if (is_local) l2_cache_local_reserve_fail++;
            else if (mf->get_mcm_req_status() == 0) l2_cache_remote_reserve_fail++;
            else l2_cache_remote_transfer_reserve_fail++;
            return;
        }

        l2_cache_access++;
        if (is_local) l2_cache_local_access++;
        else if (mf->get_mcm_req_status() == 0) l2_cache_remote_access++;
        else l2_cache_remote_transfer_access++;

        switch (probe_status) {
            case HIT:
                l2_cache_hit++;
                if (is_local)
                    l2_cache_local_hit++;
                else if (mf->get_mcm_req_status() == 0)
                    l2_cache_remote_hit++;
                else
                    l2_cache_remote_transfer_hit++;
                break;
            case HIT_RESERVED:
                l2_cache_hit_reserved++;
                if (is_local)
                    l2_cache_local_hit_reserved++;
                else if (mf->get_mcm_req_status() == 0)
                    l2_cache_remote_hit_reserved++;
                else
                    l2_cache_remote_transfer_hit_reserved++;
                break;
            case SECTOR_MISS:
            case MISS:
                l2_cache_miss++;
                if (is_local)
                    l2_cache_local_miss++;
                else if (mf->get_mcm_req_status() == 0)
                    l2_cache_remote_miss++;
                else
                    l2_cache_remote_transfer_miss++ ;
                break;
            default:
                assert(0);
        }
    }
}

// Remote memory access ratio of page walks
void memory_stats_t::collect_pw_numa_stat(mem_fetch *mf) {
    if (mf->get_tlb_depth_count() != 1) return;  // collect only last level page table access
    if (mf->get_pw_origin_chiplet() == mf->get_map_chiplet()) pw_leaf_local++;
    else pw_leaf_remote++;
}

void memory_stats_t::reset_write_record() {
    write_malloc.clear();
}
