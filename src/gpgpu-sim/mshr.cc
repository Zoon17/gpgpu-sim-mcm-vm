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

#include "mshr.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"

/****************************************************************** MSHR
 * ******************************************************************/
void mshr_table::set_config(const memory_config *config) {
    this->m_config = config;
}

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
    table::const_iterator a = m_data.find(block_addr);
    return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
    table::const_iterator i = m_data.find(block_addr);
    if (i != m_data.end()){
        return i->second.m_list.size() >= m_max_merged;
    }
    else { return m_data.size() >= m_num_entries; }
}

// for L1 cache mshr merge handle
bool mshr_table::full(new_addr_type block_addr, mem_fetch *mf) {
    table::iterator i = m_data.find(block_addr);
    if (i != m_data.end()){
        return i->second.m_list.size() >= m_max_merged;
    }
    else { return m_data.size() >= m_num_entries; }
}

/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
    if ((m_data[block_addr].m_list.size() != 0) && (mf->get_mcm_req_status() == 1) && (m_data[block_addr].is_ready)){
        inter_flit* new_flit = new inter_flit(mf->get_map_chiplet(), mf->get_origin_chiplet(), mf, true);
        mf->get_gpu()->get_inter_noc()->push_flit(new_flit);
        return;
    }

    m_data[block_addr].m_list.push_back(mf);

    assert(m_data.size() <= m_num_entries);
    assert(m_data[block_addr].m_list.size() <= m_max_merged);
    // indicate that this MSHR entry contains an atomic operation
    if (mf->isatomic()) {
        m_data[block_addr].m_has_atomic = true;
    }
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
    std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
    bool write_found = false;
    for (std::list<mem_fetch *>::iterator it = my_list.begin();
         it != my_list.end(); ++it) {
        if ((*it)->is_write())  // Pending Write Request
            write_found = true;
        else if (write_found)  // Pending Read Request and we found previous Write
            return true;
    }
    return false;
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
    assert(!busy());
    table::iterator a = m_data.find(block_addr);
    assert(a != m_data.end());
    m_data[block_addr].is_ready = true;

    if (isL1D){  // only for L1D data cache
        m_current_response.push_back(block_addr);
        has_atomic = a->second.m_has_atomic;
        assert(m_current_response.size() <= m_data.size());
    } else {
        // handle remote access
        bool check_empty = handle_remote_access(block_addr);
        if (check_empty){
            has_atomic = a->second.m_has_atomic;
            m_data.erase(block_addr);
        } else {
            m_current_response.push_back(block_addr);
            has_atomic = a->second.m_has_atomic;
            assert(m_current_response.size() <= m_data.size());
        }
    }
}

void mshr_table::mark_ready(new_addr_type key) {
    assert(!busy());
    table::iterator a = m_data.find(key);
    assert(a != m_data.end()); // don't remove same request twice
    m_data[key].is_ready = true;
    if (isL2T){  // page size information propagation
        unsigned page_size = m_data[key].m_list.front()->get_page_size();
        for (std::list<mem_fetch *>::iterator iter = m_data[key].m_list.begin(); iter != m_data[key].m_list.end(); ++iter) {
            (*iter)->set_page_size(page_size);
        }
    }
    m_current_response.push_back(key);
    assert(m_current_response.size() <= m_data.size());
}

void mshr_table::mark_ready(new_addr_type key, tlb_tag_array * L1_TLB) {
    assert(!busy());
    assert(isL1T);  // only for L1 TLB
    table::iterator a = m_data.find(key);
    assert(a != m_data.end()); // don't remove same request twice

    std::deque<mem_fetch*>** return_queues = L1_TLB->get_tlb_return_queue();
    while (!m_data[key].m_list.empty()) {
        mem_fetch* mf_process = m_data[key].m_list.front();
        m_data[key].m_list.pop_front();

        mf_process->set_tlb_ready_cycle(m_config->l1_tlb_latency);
        assert(mf_process->get_tlb_ready_cycle() != 0);

        unsigned bank_id = mf_process->get_bank_id();
        assert(bank_id != static_cast<unsigned>(-1));

        std::deque<mem_fetch*>* bank_queue = return_queues[bank_id];
        if (!bank_queue->empty()) {
            mem_fetch* mf_back = bank_queue->back();
            assert(mf_process->get_tlb_ready_cycle() >= mf_back->get_tlb_ready_cycle());
        }
        bank_queue->push_back(mf_process);
    }

    assert(m_data[key].m_list.empty());  // must process all request
    m_data.erase(key);  // release block
}

/// Returns next ready access
mem_fetch *mshr_table::next_access() {
    assert(access_ready());
    new_addr_type block_addr = m_current_response.front();
    assert(!m_data[block_addr].m_list.empty());
    assert(m_data[block_addr].is_ready);
    mem_fetch *result = m_data[block_addr].m_list.front();
    m_data[block_addr].m_list.pop_front();
    if (m_data[block_addr].m_list.empty()) {
        // release entry
        m_data.erase(block_addr);
        m_current_response.pop_front();
    }
    return result;
}

void mshr_table::display(FILE *fp) const {
    fprintf(fp, "MSHR contents\n");
    for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
        unsigned block_addr = e->first;
        fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
                e->second.m_has_atomic, e->second.m_list.size());
        if (!e->second.m_list.empty()) {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp, "%p :", mf);
            mf->print(fp);
        } else {
            fprintf(fp, " no memory requests???\n");
        }
    }
}

// handle remote access
bool mshr_table::handle_remote_access(new_addr_type block_addr) {
    auto iter = m_data[block_addr].m_list.begin();
    while (iter != m_data[block_addr].m_list.end()) {
        mem_fetch* mf = *iter;
        int status = mf->get_mcm_req_status();
        if (status == 0 || status == 2) {
            ++iter;
        } else {
            assert(status == 1);
            inter_flit* new_flit = new inter_flit(mf->get_map_chiplet(), mf->get_origin_chiplet(), mf, true);
            mf->get_gpu()->get_inter_noc()->push_flit(new_flit);
            iter = m_data[block_addr].m_list.erase(iter);
        }
    }
    if (m_data[block_addr].m_list.empty()) return true;
    else return false;
}