// ---------------------------------------------------------------------------
// Modified by: Junhyeok Park (2023-2026)
// Purpose: Add logic for address translation and handling multi-chip module
// (MCM) GPUs
// ---------------------------------------------------------------------------
// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../option_parser.h"

#ifndef ADDRDEC_H
#define ADDRDEC_H

#include "../abstract_hardware_model.h"

#ifndef PT_SPACE
#define PT_SPACE 987654321
#endif

class mmu;
class memory_config;

enum partition_index_function {
  CONSECUTIVE = 0,
  BITWISE_PERMUTATION = 1,
  IPOLY = 2,
  PAE = 3,
  RANDOM = 4,
  MCM_LOCALITY_AWARE = 5,
  MCM_INTERLEAVE = 6
};

struct addrdec_t {
  void print(FILE *fp) const;

  unsigned chip;
  unsigned bk;
  unsigned row;
  unsigned col;
  unsigned burst;

  unsigned sub_partition;

  unsigned subarray;  // not used
  // requested partition
  unsigned request_sub_partition;
  unsigned map_chiplet;
};

class linear_to_raw_address_translation {
 public:
  linear_to_raw_address_translation();
  void addrdec_setoption(option_parser_t opp);
  void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel, memory_config * config);

  // accessors
  bool addrdec_tlx(new_addr_type addr, addrdec_t *tlx, unsigned appID, unsigned level, bool isRead, unsigned chiplet) const;
  new_addr_type partition_address(new_addr_type addr, unsigned appID, unsigned level, bool isRead, unsigned chiplet) const;
  void set_mmu(mmu * page_manager);

 private:
  void addrdec_parseoption(const char *option);
  void sweep_test() const;  // sanity check to ensure no overlapping

  enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, SUBARRAY = 5, N_ADDRDEC };

  const char *addrdec_option;
  int gpgpu_mem_address_mask;
  partition_index_function memory_partition_indexing;
  bool run_test;

  int ADDR_CHIP_S;
  unsigned char addrdec_mklow[N_ADDRDEC];
  unsigned char addrdec_mkhigh[N_ADDRDEC];
  new_addr_type addrdec_mask[N_ADDRDEC];
  new_addr_type sub_partition_id_mask;

  unsigned int gap;
  unsigned m_n_channel;
  int m_n_sub_partition_in_channel;
  int m_n_sub_partition_total;
  unsigned log2channel;
  unsigned log2sub_partition;
  unsigned nextPowerOf2_m_n_channel;

  mmu * m_mmu;
  memory_config * m_config;
  int m_n_bk;     // number of banks
  int m_n_bkgrp;  // number of bank groups
};

#endif
