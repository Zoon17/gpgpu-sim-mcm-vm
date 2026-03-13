// ---------------------------------------------------------------------------
// Modified by: Junhyeok Park (2023-2026)
// Purpose: Add logic for address translation and handling multi-chip module
// (MCM) GPUs
// ---------------------------------------------------------------------------
// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>
#include <zlib.h>
#include <map>
#include "mem_fetch.h"
#include "gpu-cache.h"

class memory_config;
class memory_stats_t {
public:
  const static uint64_t stat_arr_size = 10;
  uint64_t div(uint64_t x, uint64_t y);
  float div_float(uint64_t x, uint64_t y);
  void print_stat(const char *, uint64_t, FILE *file);
  void print_stat_float(const char *, float, FILE *file);
  void print_app_stat(const char *, uint64_t, uint64_t *, uint64_t, uint64_t *, FILE *file);
  void newline(FILE *file);

  unsigned m_chiplet = 0;
  void set_chiplet(unsigned chiplet) { m_chiplet = chiplet; }

  void data_latency_stat(mem_fetch * mf);
  void mcm_cache_stat(cache_request_status access_status, cache_request_status probe_status, bool is_partition, mem_fetch * mf);

public:
  memory_stats_t(unsigned n_shader,
                 const class shader_core_config *shader_config,
                 const memory_config *mem_config, const class gpgpu_sim *gpu);

  unsigned memlatstat_done(class mem_fetch *mf);  // may be need to fix it unsigned -> uint64_t
  void memlatstat_read_done(class mem_fetch *mf);
  void memlatstat_dram_access(class mem_fetch *mf);
  void memlatstat_icnt2mem_pop(class mem_fetch *mf);
  void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);
  void memlatstat_print_file(FILE *outputfile, unsigned n_mem, unsigned gpu_mem_n_bk);

  void visualizer_print(gzFile visualizer_file);

  // Reset local L2 stats that are aggregated each sampling window
  void clear_L2_stats_pw();

  void print_essential(FILE *fout);
  void print_tlb_stat(FILE *fout);
  void print_mcm_stat(FILE *fout);

  unsigned m_n_shader;

  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;
  const class gpgpu_sim *m_gpu;

  unsigned max_mrq_latency;
  unsigned max_dq_latency;
  unsigned max_mf_latency;
  unsigned max_icnt2mem_latency;
  unsigned long long int tot_icnt2mem_latency;
  unsigned long long int tot_icnt2sh_latency;
  unsigned long long int tot_mrq_latency;
  unsigned long long int tot_mrq_num;
  unsigned max_icnt2sh_latency;
  unsigned mrq_lat_table[32];
  unsigned dq_lat_table[32];
  unsigned mf_lat_table[32];
  unsigned icnt2mem_lat_table[24];
  unsigned icnt2sh_lat_table[24];
  unsigned mf_lat_pw_table[32];  // table storing values of mf latency Per
                                 // Window
  unsigned mf_num_lat_pw;
  unsigned max_warps;
  unsigned mf_tot_lat_pw;  // total latency summed up per window. divide by
                           // mf_num_lat_pw to obtain average latency Per Window
  unsigned long long int mf_total_lat;
  unsigned long long int *
      *mf_total_lat_table;      // mf latency sums[dram chip id][bank id]
  unsigned **mf_max_lat_table;  // mf latency sums[dram chip id][bank id]
  unsigned num_mfs;
  unsigned int ***bankwrites;  // bankwrites[shader id][dram chip id][bank id]
  unsigned int ***bankreads;   // bankreads[shader id][dram chip id][bank id]
  unsigned int **totalbankwrites;    // bankwrites[dram chip id][bank id]
  unsigned int **totalbankreads;     // bankreads[dram chip id][bank id]
  unsigned int **totalbankaccesses;  // bankaccesses[dram chip id][bank id]
  unsigned int
      *num_MCBs_accessed;  // tracks how many memory controllers are accessed
                           // whenever any thread in a warp misses in cache
  unsigned int *position_of_mrq_chosen;  // position of mrq in m_queue chosen

  unsigned ***mem_access_type_stats;  // dram access type classification

  // AerialVision L2 stats
  unsigned L2_read_miss;
  unsigned L2_write_miss;
  unsigned L2_read_hit;
  unsigned L2_write_hit;

  // DRAM access row locality stats
  unsigned int *
      *concurrent_row_access;    // concurrent_row_access[dram chip id][bank id]
  unsigned int **num_activates;  // num_activates[dram chip id][bank id]
  unsigned int **row_access;     // row_access[dram chip id][bank id]
  unsigned int **max_conc_access2samerow;  // max_conc_access2samerow[dram chip
                                           // id][bank id]
  unsigned int **max_servicetime2samerow;  // max_servicetime2samerow[dram chip
                                           // id][bank id]

  // Power stats
  unsigned total_n_access;
  unsigned total_n_reads;
  unsigned total_n_writes;

  // collect page walk local/remote stat
  void collect_pw_numa_stat(mem_fetch * mf);

  uint64_t l1_tlb_tot_access = 0;
  uint64_t l1_tlb_tot_hit = 0;
  uint64_t l1_tlb_tot_miss = 0;
  uint64_t l1_tlb_tot_hit_reserved = 0;
  uint64_t l1_tlb_tot_fail = 0;
  // mshr detail
  uint64_t l1_tlb_mshr_allocate_fail = 0;
  uint64_t l1_tlb_mshr_merge_fail    = 0;
  uint64_t l1_tlb_mshr_l2_stall      = 0;

  // mshr detail
  uint64_t l2_tlb_mshr_allocate_fail = 0;
  uint64_t l2_tlb_mshr_merge_fail    = 0;
  uint64_t l2_tlb_mshr_pwq_full      = 0;

  // L2 cache write-back stat
  uint64_t l2_write_back_generate = 0;
  uint64_t l2_write_back_try      = 0;
  uint64_t l2_write_back_success  = 0;
  uint64_t l2_write_back_fail     = 0;

  uint64_t pt_space_size;
  uint64_t tlb_level_accesses[stat_arr_size];
  uint64_t tlb_level_hits[stat_arr_size];
  uint64_t tlb_level_misses[stat_arr_size];
  uint64_t tlb_level_fails[stat_arr_size];

  uint64_t total_num_mfs = 0;
  uint64_t tlb_total_num_mfs = 0;
  uint64_t data_total_num_mfs = 0;

  uint64_t l2_tlb_tot_hits, l2_tlb_tot_misses, l2_tlb_tot_accesses,
          l2_tlb_tot_mshr_hits, l2_tlb_tot_mshr_fails,
          l2_tlb_tot_backpressure_fails,
          l2_tlb_tot_backpressure_stalls;

  //page walk statistics
  uint64_t pwq_tot_lat;
  uint64_t pw_tot_lat, pw_tot_num;

  uint64_t pwc_tot_accesses, pwc_tot_hits, pwc_tot_misses;
  uint64_t pwc_tot_addr_lvl_accesses[stat_arr_size], pwc_tot_addr_lvl_hits[stat_arr_size], pwc_tot_addr_lvl_misses[stat_arr_size];

  uint64_t pw_tot_access    = 0;
  uint64_t pw_local_access  = 0;
  uint64_t pw_remote_access = 0;

  uint64_t pw_leaf_local  = 0;
  uint64_t pw_leaf_remote = 0;

  uint64_t data_tot_latency        = 0;
  uint64_t data_tot_count          = 0;
  uint64_t data_local_tot_latency  = 0;
  uint64_t data_local_tot_count    = 0;
  uint64_t data_remote_tot_latency = 0;
  uint64_t data_remote_tot_count   = 0;

  uint64_t tlb_tot_latency = 0;
  uint64_t tlb_tot_count   = 0;

  uint64_t tot_local_access  = 0;
  uint64_t tot_remote_access = 0;

  uint64_t l1_cache_access       = 0;
  uint64_t l1_cache_hit          = 0;
  uint64_t l1_cache_hit_reserved = 0;
  uint64_t l1_cache_miss         = 0;
  uint64_t l1_cache_reserve_fail = 0;

  uint64_t l1_cache_local_access       = 0;
  uint64_t l1_cache_local_hit          = 0;
  uint64_t l1_cache_local_hit_reserved = 0;
  uint64_t l1_cache_local_miss         = 0;
  uint64_t l1_cache_local_reserve_fail = 0;

  uint64_t l1_cache_remote_access       = 0;
  uint64_t l1_cache_remote_hit          = 0;
  uint64_t l1_cache_remote_hit_reserved = 0;
  uint64_t l1_cache_remote_miss         = 0;
  uint64_t l1_cache_remote_reserve_fail = 0;

  uint64_t l2_cache_access       = 0;
  uint64_t l2_cache_hit          = 0;
  uint64_t l2_cache_hit_reserved = 0;
  uint64_t l2_cache_miss         = 0;
  uint64_t l2_cache_reserve_fail = 0;

  uint64_t l2_cache_local_access       = 0;
  uint64_t l2_cache_local_hit          = 0;
  uint64_t l2_cache_local_hit_reserved = 0;
  uint64_t l2_cache_local_miss         = 0;
  uint64_t l2_cache_local_reserve_fail = 0;

  uint64_t l2_cache_remote_access       = 0;
  uint64_t l2_cache_remote_hit          = 0;
  uint64_t l2_cache_remote_hit_reserved = 0;
  uint64_t l2_cache_remote_miss         = 0;
  uint64_t l2_cache_remote_reserve_fail = 0;

  uint64_t l2_cache_remote_transfer_access       = 0;
  uint64_t l2_cache_remote_transfer_hit          = 0;
  uint64_t l2_cache_remote_transfer_hit_reserved = 0;
  uint64_t l2_cache_remote_transfer_miss         = 0;
  uint64_t l2_cache_remote_transfer_reserve_fail = 0;

  std::map<unsigned, uint64_t> tot_access_malloc;
  std::map<unsigned, uint64_t> tot_local_access_malloc;
  std::map<unsigned, uint64_t> tot_remote_access_malloc;
  uint64_t non_malloc_access = 0;

  std::map<unsigned, bool> write_malloc;
  std::map<unsigned, bool> tot_write_malloc;
  uint64_t wb_error          = 0;
  uint64_t tot_invalid_write = 0;
  void reset_write_record();

  std::map<unsigned, uint64_t> l1_hit_per_size;
  std::map<unsigned, uint64_t> l2_hit_per_size;

  new_addr_type m_fault_exception = 0;
};

#endif /*MEM_LATENCY_STAT_H*/
