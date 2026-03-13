// Copyright (c) 2009-2021, Tor M. Aamodt, Ivan Sham, Ali Bakhoda,
// George L. Yuan, Wilson W.L. Fung, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef DRAM_H
#define DRAM_H

#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "delayqueue.h"
#include "memory_owner.h"

#define READ 'R'  // define read and write states
#define WRITE 'W'
#define BANK_IDLE 'I'
#define BANK_ACTIVE 'A'

#define BANK_BLOCKED 'B' //LISA modify data within the bank
#define BANK_BLOCKED_PENDING 'P' //Sent inter-bank command, waiting for both banks to be idle
#define BANK_TRANSFER 'T' //Row-clone PSM Mode, blocking two banks (source, target)

#define LISA_COPY 0
#define InterSA_COPY 1
#define IntraSA_COPY 2
#define RC_PSM 3
#define RC_IntraSA 4
#define RC_zero 5
#define Channel_copy 6
#define COPY 7 //Generic copy, will figure the best timing from from/to_bank and subarrays
#define TARGET 8 //Target bank, got the command from other bank
#define ZERO 9 //Baseline zeroing a page
#define SCAN 10 //SCAN request (works similar to a read request, but does not queue in DRAM request queue from the GPU core side
#define DRAM_CMD 12345678 //Special ID for DRAM command, so that addrdec can parse dram command address

struct page;
class tlb_tag_array;
class Hub;

class dram_cmd{
public:
    dram_cmd(int cmd, int from_bk, int to_bk, int from_ch, int to_ch, int from_subarray, int to_sa,
             int pk_size, int app_ID, const struct memory_config * config);
    dram_cmd(int cmd, page * from_page, page * to_page, const struct memory_config * config);
    int command;
    int from_bank;
    int from_channel;
    int from_sa;
    int to_bank;
    int to_channel;
    int to_sa;
    int size;
    int appID;
    const struct memory_config *m_config;
};
/************************/

class dram_req_t {
 public:
  dram_req_t(class mem_fetch *data, unsigned banks,
             unsigned dram_bnk_indexing_policy, class gpgpu_sim *gpu);

  unsigned int row;
  unsigned int col;
  unsigned int bk;
  unsigned int nbytes;
  unsigned int txbytes;
  unsigned int dqbytes;
  unsigned int age;
  unsigned int timestamp;
  unsigned char rw;  // is the request a read or a write?
  unsigned long long int addr;
  unsigned int insertion_time;
  class mem_fetch *data;
  class gpgpu_sim *m_gpu;
};

struct bankgrp_t {
  unsigned int CCDLc;
  unsigned int RTPLc;
};

struct bank_t {
  unsigned int RCDc;
  unsigned int RCDWRc;
  unsigned int RASc;
  unsigned int RPc;
  unsigned int RCc;
  unsigned int WTPc;  // write to precharge
  unsigned int RTPc;  // read to precharge

  unsigned char rw;     // is the bank reading or writing?
  unsigned char state;  // is the bank active or idle?
  unsigned int curr_row;
  unsigned int curr_subarray;

  dram_req_t *mrq;

  unsigned int n_access;
  unsigned int n_writes;
  unsigned int n_idle;

  //Counter for BANK_BLOCKED and BANK_TRANSFER, if these are zeros, banks should become idle
  unsigned int blocked;
  unsigned int transfer;
  std::list<dram_cmd*> * cmd_queue;

    unsigned int bkgrpindex;
};

enum bank_index_function {
  LINEAR_BK_INDEX = 0,
  BITWISE_XORING_BK_INDEX,
  IPOLY_BK_INDEX,
  CUSTOM_BK_INDEX
};

enum bank_grp_bits_position { HIGHER_BITS = 0, LOWER_BITS };

class mem_fetch;
class memory_config;

class dram_t {
 public:
  dram_t(unsigned int parition_id, const memory_config *config,
         class memory_stats_t *stats, class memory_partition_unit *mp,
         class gpgpu_sim *gpu,
         mmu * page_manager,
         tlb_tag_array * shared_tlb);

  mmu * m_page_manager;
  tlb_tag_array * m_shared_tlb;

  unsigned compaction_bank_id;
  unsigned data_bus_busy;

  unsigned dram_bwutil();

  unsigned dram_bwutil_data();

  unsigned dram_bwutil_tlb();

  void set_miss(float m);
  void set_miss_core(float m, unsigned which_core);

  float get_miss();
  float get_miss_core(unsigned i);

  float get_rbl();
  /*****************************/

  bool full(bool is_write) const;
  void print(FILE *simFile) const;
  void visualize() const;
  void print_stat(FILE *simFile);
  unsigned que_length() const;
  bool returnq_full() const;
  unsigned int queue_limit() const;
  void visualizer_print(gzFile visualizer_file);

  class mem_fetch *return_queue_pop();
  class mem_fetch *return_queue_top();

  void push(class mem_fetch *data);
  void cycle();
  void dram_log(int task);

  class memory_partition_unit *m_memory_partition_unit;
  class gpgpu_sim *m_gpu;
  unsigned int id;

  void insert_dram_command(dram_cmd * cmd);

  // Power Model
  void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &wr_WB, unsigned &req) const;

  const memory_config *m_config;

 private:
  bankgrp_t **bkgrp;

  bank_t **bk;
  unsigned int prio;

  unsigned get_bankgrp_number(unsigned i);

  void scheduler_fifo();
  void scheduler_frfcfs();

  bool issue_col_command(int j);
  bool issue_row_command(int j);

  unsigned int RRDc;
  unsigned int CCDc;
  unsigned int RTWc;  // read to write penalty applies across banks
  unsigned int WTRc;  // write to read penalty applies across banks

  unsigned char
      rw;  // was last request a read or write? (important for RTW, WTR)

  unsigned int pending_writes;

  fifo_pipeline<dram_req_t> *rwq;
  fifo_pipeline<dram_req_t> *mrqq;
  // buffer to hold packets when DRAM processing is over
  // should be filled with dram clock and popped with l2or icnt clock
  fifo_pipeline<mem_fetch> *returnq;

  std::list<mem_fetch*> wait_list;  // A queue for TLB-related requests

  unsigned int dram_util_bins[10];
  unsigned int dram_eff_bins[10];
  unsigned int last_n_cmd, last_n_activity, last_bwutil;

  unsigned long long n_cmd;
  unsigned long long n_activity;
  unsigned long long n_nop;
  unsigned long long n_act;
  unsigned long long n_pre;
  unsigned long long n_ref;
  unsigned long long n_rd;
  unsigned long long n_rd_L2_A;
  unsigned long long n_wr;
  unsigned long long n_wr_WB;
  unsigned long long n_req;
  unsigned long long max_mrqs_temp;

  // some statistics to see where BW is wasted?
  unsigned long long wasted_bw_row;
  unsigned long long wasted_bw_col;
  unsigned long long util_bw;
  unsigned long long idle_bw;
  unsigned long long RCDc_limit;
  unsigned long long CCDLc_limit;
  unsigned long long CCDLc_limit_alone;
  unsigned long long CCDc_limit;
  unsigned long long WTRc_limit;
  unsigned long long WTRc_limit_alone;
  unsigned long long RCDWRc_limit;
  unsigned long long RTWc_limit;
  unsigned long long RTWc_limit_alone;
  unsigned long long rwq_limit;

  // row locality, BLP and other statistics
  unsigned long long access_num;
  unsigned long long read_num;
  unsigned long long write_num;
  unsigned long long hits_num;
  unsigned long long hits_read_num;
  unsigned long long hits_write_num;
  unsigned long long banks_1time;
  unsigned long long banks_acess_total;
  unsigned long long banks_acess_total_after;
  unsigned long long banks_time_rw;
  unsigned long long banks_access_rw_total;
  unsigned long long banks_time_ready;
  unsigned long long banks_access_ready_total;
  unsigned long long issued_two;
  unsigned long long issued_total;
  unsigned long long issued_total_row;
  unsigned long long issued_total_col;
  double write_to_read_ratio_blp_rw_average;
  unsigned long long bkgrp_parallsim_rw;

  unsigned int bwutil;
  unsigned int bwutil_data;
  unsigned int bwutil_tlb;
  unsigned int max_mrqs;
  unsigned int ave_mrqs;

  unsigned int bwutil_periodic;
  unsigned int bwutil_periodic_data;
  unsigned int bwutil_periodic_tlb;
  unsigned int n_cmd_blp;
  unsigned int  mem_state_blp;
  unsigned int  mem_state_blp_alarm[32];
  unsigned int  mem_state_blp_ncmd[32];
  unsigned int sanity_read;
  unsigned int sanity_write;

  float miss_rate_d;
  float miss_rate_d_core[64];

  unsigned int dram_cycles;
  unsigned int dram_cycles_active;

  class frfcfs_scheduler *m_frfcfs_scheduler;

  unsigned int n_cmd_partial;
  unsigned int n_activity_partial;
  unsigned int n_nop_partial;
  unsigned int n_act_partial;
  unsigned int n_pre_partial;
  unsigned int n_req_partial;
  unsigned int ave_mrqs_partial;
  unsigned int bwutil_partial;

  class memory_stats_t *m_stats;
  class Stats *mrqq_Dist;  // memory request queue inside DRAM

  friend class frfcfs_scheduler;
};

#endif /*DRAM_H*/
