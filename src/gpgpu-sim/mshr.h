// Created by junhyeok on 2/21/23.
// ---------------------------------------------------------------------------
// Modified by: Junhyeok Park (2023-2026)
// Purpose: Add logic for address translation and handling multi-chip module
// (MCM) GPUs
// ---------------------------------------------------------------------------
// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah, Nikos Hardavellas
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

#ifndef ACCELSIM_MSHR_H
#define ACCELSIM_MSHR_H

#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

class memory_config;
class cache_config;
class memory_stats_t;

class mshr_table {
public:
    mshr_table(unsigned num_entries, unsigned max_merged)
            : m_num_entries(num_entries), m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
            ,
              m_data(2 * num_entries)
#endif
    {
        isL1T = false;
        isL2T = false;
        isL1D = false;
        isL2D = false;
    }

    /// Checks if there is a pending request to the lower memory level already
    bool probe(new_addr_type block_addr) const;
    /// Checks if there is space for tracking a new memory access
    bool full(new_addr_type block_addr) const;
    /// Add or merge this access
    void add(new_addr_type block_addr, mem_fetch *mf);
    /// Returns true if cannot accept new fill responses
    bool busy() const { return false; }
    /// Accept a new cache fill response: mark entry ready for processing
    void mark_ready(new_addr_type block_addr, bool &has_atomic);
    /// Returns true if ready accesses exist
    bool access_ready() const { return !m_current_response.empty(); }
    /// Returns next ready access
    mem_fetch *next_access();
    void display(FILE *fp) const;
    // Returns true if there is a pending read after write
    bool is_read_after_write_pending(new_addr_type block_addr);

    void check_mshr_parameters(unsigned num_entries, unsigned max_merged) const {
        assert(m_num_entries == num_entries &&
               "Change of MSHR parameters between kernels is not allowed");
        assert(m_max_merged == max_merged &&
               "Change of MSHR parameters between kernels is not allowed");
    }

    bool full(new_addr_type block_addr, mem_fetch * mf);

    void mark_ready(new_addr_type key);
    // for l1 tlb return queue
    void mark_ready(new_addr_type key, tlb_tag_array * L1_TLB);

    void set_config(const memory_config* config);
    void set_mem_stats(memory_stats_t * mem_stat){
        m_mem_stats = mem_stat;
    }

    void set_isL1T() { this->isL1T = true; }
    void set_isL2T() { this->isL2T = true; }
    void set_isL1D() { this->isL1D = true; }
    void set_isL2D() { this->isL2D = true; }
private:
    // finite sized, fully associative table, with a finite maximum number of
    // merged requests
    const unsigned m_num_entries;
    const unsigned m_max_merged;

    struct mshr_entry {
        std::list<mem_fetch *> m_list;
        bool m_has_atomic;
        bool is_ready;
        mshr_entry()
            : m_has_atomic(false), is_ready(false) {}
    };

    typedef tr1_hash_map<new_addr_type, mshr_entry> table;
    typedef tr1_hash_map<new_addr_type, mshr_entry> line_table;
    table m_data;
    line_table pending_lines;

    // it may take several cycles to process the merged requests
    bool m_current_response_ready;
    std::list<new_addr_type> m_current_response;

    const memory_config* m_config = nullptr;
    memory_stats_t * m_mem_stats = nullptr;
    bool isL1T;
    bool isL2T;
    bool isL1D;
    bool isL2D;

public:
    bool handle_remote_access(new_addr_type block_addr);
};


#endif //ACCELSIM_MSHR_H
