// Created by junhyeok on 2/20/23.
// Copyright (c) 2023-2026 Junhyeok Park. All rights reserved.
//
// Portions of the initial logic for virtual memory support were referenced from
// 1. Mosaic (CMU SAFARI, MICRO'17) / MASK (CMU SAFARI, ASPLOS'18)
// 2. DWS (Indian Institute of Science, HPCA'21)
// This file has been modified, optimized, and extended to improve functionality.
//
// Built for compatibility with the GPGPU-Sim framework.
// Licensed under the BSD 3-Clause License.
// See the COPYRIGHT file in the project root for full license text.

#ifndef ACCELSIM_TLB_H
#define ACCELSIM_TLB_H

#include <queue>

#include "delayqueue.h"
#include "mshr.h"

#define GLOBAL_SPACE 0
#define TEXTURE_SPACE 1
#define SHARED_SPACE 2
#define OTHER_SPACE 3

class PageWalker;
class PageWalkSubsystem;
class mem_fetch;
class mmu;
class shader_core_stats;
class memory_stats_t;
class memory_partition_unit;
class memory_config;
class dram_cmd;
class page_table;
class tlb_fetch;
class cache_config;

enum tlb_request_status {
    TLB_HIT,
    TLB_HIT_PROCESS,
    TLB_HIT_RESERVED,
    TLB_PENDING,
    TLB_MISS,
    TLB_MSHR_FAIL,
    TLB_BACKPRESSURE_MISS,
    TLB_FLUSH,
    TLB_FAULT
};

enum page_mode_status {
  PAGE_4K  = 1,
  PAGE_64K = 16,
  PAGE_2M  = 512,
};

class tlb_tag_array{
private:
    unsigned m_tot_chiplet     = -1;  // tot chiplet num
    unsigned m_own_chiplet_num = -1;  // own chiplet_num
    unsigned m_page_mode       = -1;  // used for paging chunk size selection

    struct tlb_entry {
      new_addr_type m_tag;
      new_addr_type m_shift;
      unsigned m_page_size;
      tlb_entry() {
        m_tag       = 0;
        m_shift     = 0;
        m_page_size = 0;
      }
      tlb_entry(new_addr_type tag_in , new_addr_type shift_in, unsigned size_in) {
        m_tag       = tag_in;
        m_shift     = shift_in;
        m_page_size = size_in;
      }
    };
    std::list<tlb_entry>* tag_array;
    std::list<tlb_entry>** l2_tag_array;
    mshr_table m_mshrs;

    typedef std::unordered_map<new_addr_type, unsigned> page_map;
    page_map page_mode_map;  // used to maintain page mode information for each 2MB chunk

    std::deque<mem_fetch*>** tlb_return_queue;  // add for tlb return

    PageWalkSubsystem* page_walk_subsystem;
    bool stall;

    memory_partition_unit** m_memory_partition;
    page_table* root;
    mmu* m_page_manager;
    shader_core_stats* m_stat;
    tlb_tag_array** l1_tlb;
    tlb_tag_array * m_mother_tlb;
    tlb_tag_array** chiplet_l2_tlb;
    const memory_config* m_config;
    tlb_tag_array* m_shared_tlb;
    memory_stats_t* m_mem_stats;
    fifo_pipeline<tlb_fetch> ** chiplet_request_queue;
    int m_core_id;

    unsigned m_ways, m_entries;
    bool m_isL2TLB;
    class gpgpu_sim * m_gpu;
    Hub * m_gpu_alloc;

    new_addr_type m_debug_counter = 1;

public:
    void done_tlb_req(mem_fetch * mf);

    /* Constructors and Destructor */
    tlb_tag_array(const memory_config* config, shader_core_stats* stat,
                  mmu* page_manager, tlb_tag_array* shared_tlb, int core_id, memory_stats_t * mem_stat,
                  class gpgpu_sim *gpu);
    tlb_tag_array(const memory_config* config, shader_core_stats* stat,
                  mmu* page_manager, bool isL2TLB, memory_stats_t* mem_stat,
                  memory_partition_unit** mem_partition, class gpgpu_sim * gpu);
    tlb_tag_array(const memory_config* config, shader_core_stats* stat,
                  mmu* page_manager, bool isL2TLB, memory_stats_t* mem_stat,
                  memory_partition_unit** mem_partition, class gpgpu_sim * gpu,
                  tlb_tag_array * mother_tlb, unsigned chiplet);
    ~tlb_tag_array();

    enum tlb_request_status probe(new_addr_type addr,
                                  unsigned accessor, mem_fetch * mf);
    enum tlb_request_status probe(new_addr_type addr, mem_fetch * mf, unsigned chiplet);  // TLB probe with chiplet ID
    enum tlb_request_status probe(new_addr_type addr, mem_fetch * mf);  // TLB probe with chiplet ID

    bool access(tlb_fetch* tf, unsigned chiplet);  // L2 TLB access in MCM GPUs

    new_addr_type get_tlbreq_addr(mem_fetch * mf);
    bool get_stall() const { return stall; }
    void set_stall(bool set) { this->stall = set; }

    void set_l1_tlb(int coreID, tlb_tag_array* l1);
    void set_l1_tlb(int coreID, tlb_tag_array* l1, unsigned chiplet);  // set L1 TLB for per-chiplet L2 TLB

    void fill(new_addr_type addr, mem_fetch* mf);
    void fill(new_addr_type addr, unsigned accessor, mem_fetch* mf);

    std::deque<mem_fetch*>** get_tlb_return_queue() { return tlb_return_queue; }
    bool request_shared_tlb(new_addr_type addr, unsigned accessor, mem_fetch* mf);

    /* L2 TLB specific methods */
    void cycle();  // L2 TLB access handle for MCM GPUs
    void fill_into_l1_tlb(new_addr_type addr, mem_fetch* mf);
    void l2_fill(new_addr_type addr, unsigned accessor, mem_fetch* mf);

    tlb_tag_array* get_shared_tlb() {
        return m_shared_tlb;
    }
    const memory_config* get_memory_config() const { return m_config; }
    memory_stats_t* get_mem_stat() {
        assert(m_mem_stats != nullptr);
        return m_mem_stats;
    }
    new_addr_type get_key(new_addr_type addr, unsigned appid);

    bool access_ready() const {
        return m_mshrs.access_ready();
    }
    mem_fetch *next_access() {
        return m_mshrs.next_access();
    }

    unsigned get_tlb_index(new_addr_type key);

    unsigned record_page_size(new_addr_type key, unsigned chiplet, unsigned page_mode);
    unsigned get_page_size(new_addr_type key);

    void debug();

    // shared tlb update
    unsigned get_tlb_chiplet(new_addr_type key);

    // TLB shootdwon
    void flush_TLBs(uint64_t vpn_chunk);
    void flush_private(uint64_t vpn_chunk);
    void flush_shared(uint64_t vpn_chunk);
};

class tlb_fetch {
public:
    tlb_fetch(tlb_tag_array* origin_tlb, mem_fetch* mf, new_addr_type addr,
              unsigned accessor, new_addr_type ready_cycle);
    mem_fetch* get_mf() { return mf; }
    new_addr_type get_ready_cycle() const { return ready_cycle; }
    void set_ready_cycle(unsigned delay_cycle){
        ready_cycle = delay_cycle;
    }
private:
    mem_fetch* mf;
    tlb_tag_array* origin_tlb;
    new_addr_type addr;
    unsigned accessor;
    new_addr_type ready_cycle;
};

#endif //ACCELSIM_TLB_H
