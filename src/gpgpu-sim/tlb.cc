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

#include "stat-tool.h"
#include <cassert>
#include "tlb.h"
#include "dram.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "l2cache.h"
#include <map>
#include "pagewalk.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#define APP_ID 1

// L1 TLB Constructor
tlb_tag_array::tlb_tag_array(const memory_config *config,
                             shader_core_stats *stat, mmu *page_manager,
                             tlb_tag_array * shared_tlb, int core_id, memory_stats_t * mem_stat,
                             class gpgpu_sim *gpu)
        : m_mshrs(config->L1_TLB_MSHR_entry, config->L1_TLB_MSHR_merge)
{
    m_gpu = gpu;
    m_gpu_alloc = gpu->m_gpu_alloc;
    m_config = config;
    m_tot_chiplet = m_config->chiplet_num;
    m_stat = stat;
    m_page_manager = page_manager;
    m_shared_tlb = shared_tlb;
    m_core_id = core_id;

    assert(shared_tlb != nullptr);
    m_mem_stats = mem_stat;

    m_isL2TLB = false;

    // Aan unified tlb that handles multiple page sizes
    tag_array = new std::list<tlb_entry>;

    root = page_manager->get_page_table_root();

    m_mshrs.set_isL1T();
    m_mshrs.set_config(config);
    m_mshrs.set_mem_stats(m_mem_stats);

    tlb_return_queue = new std::deque<mem_fetch*> *[m_config->l1_tlb_port];
    for (unsigned i = 0; i < m_config->l1_tlb_port; i++) {
        tlb_return_queue[i] = new std::deque<mem_fetch*>;
    }

    // init unused member variables
    l1_tlb                = nullptr;
    m_mother_tlb          = nullptr;
    chiplet_l2_tlb        = nullptr;
    chiplet_request_queue = nullptr;
    m_memory_partition    = nullptr;
    l2_tag_array          = nullptr;
    page_walk_subsystem   = nullptr;
    m_ways                = 0;
    m_entries             = 0;
    stall                 = false;
}

// L2 TLB Constructor : Mother TLB in MCM GPUs
tlb_tag_array::tlb_tag_array(const memory_config *config,
                             shader_core_stats *stat, mmu *page_manager, bool isL2TLB,
                             memory_stats_t *mem_stat, memory_partition_unit **mem_partition,
                             class gpgpu_sim * gpu)
    : m_mshrs(0, 0) {
    m_gpu = gpu;
    m_gpu_alloc = gpu->m_gpu_alloc;
    m_config = config;
    m_tot_chiplet = m_config->chiplet_num;  // tot_chiplet_cnt
    m_stat = stat;
    m_page_manager = page_manager;
    m_mem_stats = mem_stat;
    assert(m_mem_stats != nullptr);
    m_memory_partition = mem_partition;
    stall = false;
    m_isL2TLB = true;

    l1_tlb = new tlb_tag_array *[config->total_sm_num];
    m_page_manager->set_L2_tlb(this);

    unsigned per_chiplet_sm_cnt = m_config->total_sm_num / m_tot_chiplet;
    chiplet_request_queue = new fifo_pipeline<tlb_fetch> *[m_tot_chiplet];
    for (unsigned i = 0; i < m_tot_chiplet; i++){
        chiplet_request_queue[i] = new fifo_pipeline<tlb_fetch>("crq",
            0, m_config->L1_TLB_MSHR_entry * per_chiplet_sm_cnt);
    }

    // L2 TLBs for each chiplet
    chiplet_l2_tlb = new tlb_tag_array * [m_tot_chiplet];
    for (unsigned i = 0; i < m_tot_chiplet; i++) {
        chiplet_l2_tlb[i] = new tlb_tag_array(config, stat, page_manager, isL2TLB, mem_stat,
                                              mem_partition, gpu, this, i);
    }
    m_page_mode = m_config->set_page_size;

    // init unused member variables
    tag_array           = nullptr;
    tlb_return_queue    = nullptr;
    root                = nullptr;
    m_mother_tlb        = nullptr;
    m_shared_tlb        = nullptr;
    l2_tag_array        = nullptr;
    page_walk_subsystem = nullptr;
    m_core_id           = 0;
    m_ways              = 0;
    m_entries           = 0;
}

// tlb constructor for per-chiplet L2 TLB
tlb_tag_array::tlb_tag_array(const memory_config *config, shader_core_stats *stat, mmu *page_manager, bool isL2TLB,
                             memory_stats_t *mem_stat, memory_partition_unit **mem_partition,
                             class gpgpu_sim *gpu, tlb_tag_array * mother_tlb, unsigned int chiplet)
    : m_mshrs(config->L2_TLB_MSHR_entry, config->L2_TLB_MSHR_merge)
{
    m_gpu = gpu;
    m_gpu_alloc = gpu->m_gpu_alloc;
    m_config = config;
    m_tot_chiplet = m_config->chiplet_num;
    m_own_chiplet_num = chiplet;
    m_mother_tlb = mother_tlb;
    m_stat = stat;
    m_page_manager = page_manager;
    m_mem_stats = mem_stat;
    assert(m_mem_stats != nullptr);
    m_memory_partition = mem_partition;
    m_shared_tlb = nullptr;
    m_ways = m_config->l2_tlb_ways;
    m_entries = m_config->l2_tlb_ways == 0 ? 0 : m_config->l2_tlb_entries / m_config->l2_tlb_ways;

    l2_tag_array = new std::list<tlb_entry> * [m_entries];
    for (unsigned i = 0; i < m_entries; i++){
        l2_tag_array[i] = new std::list<tlb_entry>;
    }

    page_walk_subsystem =
        new PageWalkSubsystem(this, page_manager, config, mem_stat);
    stall = false;
    l1_tlb = new tlb_tag_array *[config->total_sm_num];
    m_isL2TLB = true;

    m_mshrs.set_isL2T();
    m_mshrs.set_config(config);

    // init unused member variables
    tag_array             = nullptr;
    tlb_return_queue      = nullptr;
    root                  = nullptr;
    chiplet_request_queue = nullptr;
    chiplet_l2_tlb        = nullptr;
    m_core_id             = 0;
}

tlb_tag_array::~tlb_tag_array() = default;

/* Rachata: Given an access, get the addess of the location of the tlb */
new_addr_type tlb_tag_array::get_tlbreq_addr(mem_fetch *mf) {
    new_addr_type return_addr = root->parse_pa(mf);
    return return_addr;
}

// Right now multi-page-size fill only support baseline and MASK
// Only for L1 TLB
void tlb_tag_array::fill(new_addr_type addr, mem_fetch *mf) {
    /* Does not have to fill if we always return TLB HIT(speeding things up)*/
    if (m_config->vm_config == VM_IDEAL_TLB){
        return; }

    new_addr_type key = get_key(addr, mf->get_appID());
    m_mshrs.mark_ready(key, this);

    unsigned page_size = mf->get_page_size();
    assert(page_size != 0);

    new_addr_type page_shift = (new_addr_type)(std::log2(static_cast<float>(page_size)));
    new_addr_type key_shift = (new_addr_type)(key >> page_shift);

    // this time, does not erase duplicated entries
    std::list<tlb_entry> *active_tag_array = tag_array;

    if (active_tag_array->size() >= m_config->tlb_size) {
        active_tag_array->pop_back();  // LRU
    }
    active_tag_array->emplace_front(key_shift, page_shift, page_size);
}


void tlb_tag_array::set_l1_tlb(int coreID, tlb_tag_array *l1) {
    l1_tlb[coreID] = l1;

    // propagate to the per-chiplet l2 tlb
    for (unsigned i = 0; i < m_tot_chiplet; i++){
        chiplet_l2_tlb[i]->set_l1_tlb(coreID, l1, i);
    }
}

void tlb_tag_array::set_l1_tlb(int coreID, tlb_tag_array *l1, unsigned int chiplet) {
    l1_tlb[coreID] = l1;
}

// fill into the per-chiplet L2 TLB
void tlb_tag_array::l2_fill(new_addr_type addr,
                            unsigned accessor, mem_fetch *mf) {
    m_shared_tlb->chiplet_l2_tlb[mf->get_tlb_chiplet()]->fill(addr, accessor, mf);
}

// L2 TLB fill
void tlb_tag_array::fill(new_addr_type addr, unsigned accessor,
                         mem_fetch *mf)
{
    assert(accessor == 1);
    new_addr_type key = get_key(addr, mf->get_appID());
    m_mshrs.mark_ready(key);
    fill_into_l1_tlb(addr, mf);

    unsigned page_size = mf->get_page_size();
    assert(page_size != 0);

    new_addr_type page_shift = (new_addr_type)(std::log2(static_cast<float>(page_size)));
    new_addr_type key_shift = (new_addr_type)(key >> page_shift);

    // index check, hashed_index for unified tlb
    unsigned index = get_tlb_index(key);

    std::list<tlb_entry> **correct_tag_array = l2_tag_array;

    // this time, does not erase duplicated entry
    if (correct_tag_array[index]->size() >= (m_ways)) {
        correct_tag_array[index]->pop_back();  // LRU
    }
    correct_tag_array[index]->emplace_front(key_shift, page_shift, page_size);
}

// TLB lookup function for L1 TLBs
enum tlb_request_status tlb_tag_array::probe(
        new_addr_type addr, unsigned accessor, mem_fetch *mf) {
    if (m_config->enable_remote_debug && mf->get_malloc_num() == static_cast<unsigned>(-1)) {
        mf->set_malloc_num(m_gpu_alloc->get_malloc(mf->get_original_addr()));
    }

    m_mem_stats->l1_tlb_tot_access++;
    if (m_config->vm_config == VM_IDEAL_TLB){
        assert(m_shared_tlb != nullptr);
        unsigned bank_id = mf->get_bank_id();
        assert(bank_id != static_cast<unsigned>(-1));
        mf->set_tlb_ready_cycle(m_config->l1_tlb_latency);
        if (!tlb_return_queue[bank_id]->empty()){
            mem_fetch * mf_back = tlb_return_queue[bank_id]->back();
            assert(mf->get_tlb_ready_cycle() >= mf_back->get_tlb_ready_cycle());
        }
        tlb_return_queue[bank_id]->push_back(mf);
        return TLB_HIT_PROCESS;
    }

    new_addr_type key = get_key(addr, mf->get_appID());

    for (auto iter = tag_array->begin(); iter != tag_array->end(); ++iter){
        new_addr_type key_shift = iter->m_shift;
        auto probe_key = (new_addr_type)(key >> key_shift);
        if (probe_key == iter->m_tag) {
            unsigned page_size = iter->m_page_size;
            tag_array->splice(tag_array->begin(), *tag_array, iter);  // LRU update
            m_mem_stats->l1_tlb_tot_hit++;
            m_mem_stats->l1_hit_per_size[page_size]++;
            unsigned bank_id = mf->get_bank_id();
            assert(bank_id != static_cast<unsigned>(-1));
            mf->set_tlb_ready_cycle(m_config->l1_tlb_latency);
            if (!tlb_return_queue[bank_id]->empty()){
              mem_fetch * mf_back = tlb_return_queue[bank_id]->back();
              assert(mf->get_tlb_ready_cycle() >= mf_back->get_tlb_ready_cycle());
            }
            tlb_return_queue[bank_id]->push_back(mf);
          return TLB_HIT_PROCESS;
        }
    }

    bool mshr_hit = m_mshrs.probe(key);
    bool mshr_avail = !m_mshrs.full(key);
    if (mshr_hit && mshr_avail) {
        m_mshrs.add(key, mf);
        m_mem_stats->l1_tlb_tot_hit_reserved++;
        return TLB_MISS;
    } else if (mshr_hit && !mshr_avail) {  // mhsr merge fail
        m_mem_stats->l1_tlb_tot_fail++;
        m_mem_stats->l1_tlb_mshr_merge_fail++;
        return TLB_MSHR_FAIL;
    }

    assert(!mshr_hit);

    if (!mshr_hit && mshr_avail) {
        if (request_shared_tlb(addr, accessor, mf)) {  // send request to L2 tlb
            m_mshrs.add(key, mf);
            m_mem_stats->l1_tlb_tot_miss++;
            return TLB_MISS;
        } else {
            m_mem_stats->l1_tlb_tot_fail++;
            m_mem_stats->l1_tlb_mshr_l2_stall++;
            return TLB_MSHR_FAIL;
        }
    } else {  // mshr allocate fail
        m_mem_stats->l1_tlb_tot_fail++;
        m_mem_stats->l1_tlb_mshr_allocate_fail++;
        return TLB_MSHR_FAIL;
    }
}

enum tlb_request_status tlb_tag_array::probe(new_addr_type addr, mem_fetch *mf, unsigned int chiplet) {
    return chiplet_l2_tlb[chiplet]->probe(addr, mf);  // pass to per-chiplet L2 TLB
}

//per-chiplet L2 TLB probe
enum tlb_request_status tlb_tag_array::probe(new_addr_type addr, mem_fetch *mf) {
    assert(m_shared_tlb == nullptr);
    new_addr_type key = get_key(addr, mf->get_appID());

    m_mem_stats->l2_tlb_tot_accesses++;
    if (m_config->vm_config == VM_IDEAL_L2_TLB){
        tlb_tag_array *l1_tlb = mf->get_tlb();
        l1_tlb->fill(mf->get_addr(), mf);
        m_mem_stats->l2_tlb_tot_hits++;
        return TLB_HIT;
    }

    unsigned index = get_tlb_index(key);
    std::list<tlb_entry> * probe_array = l2_tag_array[index];
    for (auto iter = probe_array->begin(); iter != probe_array->end(); ++iter){
        new_addr_type key_shift = iter->m_shift;
        new_addr_type probe_key = (new_addr_type)(key >> key_shift);
        if (probe_key == iter->m_tag) {
            /* Fill into L1 TLB */
            unsigned page_size = iter->m_page_size;
            mf->set_page_size(page_size);
            tlb_tag_array *l1_tlb = mf->get_tlb();
            l1_tlb->fill(mf->get_addr(), mf);

            probe_array->splice(probe_array->begin(), *probe_array, iter);  // LRU update
            m_mem_stats->l2_tlb_tot_hits++;
            m_mem_stats->l2_hit_per_size[page_size]++;
            return TLB_HIT;
        }
    }

    bool mshr_hit = m_mshrs.probe(key);
    bool mshr_avail = !m_mshrs.full(key);
    if (mshr_hit && mshr_avail) {
        m_mshrs.add(key, mf);
        m_mem_stats->l2_tlb_tot_mshr_hits++;
        return TLB_HIT_RESERVED;
    }

    if (!mshr_hit && mshr_avail) {
        if (m_config->enable_walk_fault) {
            if (!m_gpu_alloc->check_mapping(key)) {
                mf->trigger_fault();
            }
        }

        mf->set_page_size(m_mother_tlb->get_page_size(m_mother_tlb->get_key(mf->get_addr(), mf->get_appID())));
        mf->set_page_addr();

        if (page_walk_subsystem->enqueue(mf))   // page walk start or page walk enqueue
        {
            m_mshrs.add(key, mf);
            m_mem_stats->l2_tlb_tot_misses++;
            return TLB_MISS;
        } else {
            m_mem_stats->l2_tlb_mshr_pwq_full++;
            return TLB_BACKPRESSURE_MISS;
        }
    } else {
        m_mem_stats->l2_tlb_tot_mshr_fails++;
        // stat collect
        if (mshr_hit && !mshr_avail) {
            m_mem_stats->l2_tlb_mshr_merge_fail++;
        }
        if (!mshr_hit && !mshr_avail) {
            m_mem_stats->l2_tlb_mshr_allocate_fail++;
        }
        return TLB_MSHR_FAIL;
    }
}

bool tlb_tag_array::request_shared_tlb(new_addr_type addr,
                                       unsigned accessor, mem_fetch *mf) {

    if (m_shared_tlb->get_stall()) {
        m_mem_stats->l2_tlb_tot_backpressure_stalls++;
        return false;
    }

    new_addr_type ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config->l2_tlb_latency;
    assert(mf->get_tlb() != nullptr);
    tlb_fetch *tf = new tlb_fetch(this, mf, addr, accessor, ready_cycle);

    // push into correct chiplet queue
    unsigned req_chiplet = mf->get_origin_chiplet();
    mf->set_tlb_chiplet(req_chiplet);
    m_shared_tlb->chiplet_request_queue[req_chiplet]->push(tf);
    return true;
}

/* L2 TLB only */
void tlb_tag_array::fill_into_l1_tlb(new_addr_type addr, mem_fetch *mf) {
    while (m_mshrs.access_ready())
    {
        mem_fetch *mshr_f = m_mshrs.next_access();
        tlb_tag_array *l1_t = mshr_f->get_tlb();
        l1_t->fill(addr, mshr_f);
    }
}

bool tlb_tag_array::access(tlb_fetch *tf, const unsigned int chiplet) {
    mem_fetch *mf = tf->get_mf();
    if (mf->get_tlb() == nullptr){
        abort();
    }
    /* the probe function handles the downstream sending of page walk */
    tlb_request_status status = probe(mf->get_addr(), mf, chiplet);

    if (status == TLB_HIT || status == TLB_MISS || status == TLB_HIT_RESERVED)
    {
        return true;
    }
    else if (status == TLB_MSHR_FAIL || status == TLB_BACKPRESSURE_MISS)
    {
        return false;
    }
    else
    {
        assert(0 && "should not have reached here\n");
    }
}

void tlb_tag_array::cycle() {
    for (unsigned i = 0; i < m_tot_chiplet; i++) {
        unsigned ports = 0;
        while (!chiplet_request_queue[i]->empty() && ports < m_config->l2_tlb_ports) {
            tlb_fetch *tf = chiplet_request_queue[i]->top();
            if (tf->get_ready_cycle() > tf->get_mf()->get_time()) {
                break;
            }
            bool success = access(tf, i);
            if (success) {
                chiplet_request_queue[i]->pop();
                delete tf;
                ports++;
            }
            else {
                break;
            }
        }
        chiplet_l2_tlb[i]->page_walk_subsystem->service_page_walk_cache_queue();
    }
    debug();
}

tlb_fetch::tlb_fetch(tlb_tag_array *origin_tlb, mem_fetch *mf,
                     new_addr_type addr, unsigned accessor, new_addr_type ready_cycle) {
    this->origin_tlb = origin_tlb;
    this->mf = mf;
    this->addr = addr;
    this->accessor = accessor;
    this->ready_cycle = ready_cycle;
}

new_addr_type tlb_tag_array::get_key(new_addr_type addr, unsigned appid) {
    new_addr_type key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size() - 1];
    assert((key & (static_cast<uint64_t>(3) << 62)) == 0);
    return key;
}

// record page mode for each 2MB chunk
unsigned tlb_tag_array::record_page_size(new_addr_type key, unsigned chiplet, unsigned page_mode) {
    new_addr_type key_chunk = key >> 9;  // shift hard coded
    auto find_chunk = page_mode_map.find(key_chunk);
    if (find_chunk != page_mode_map.end()){
        return page_mode_map.at(key_chunk);
    }
    else {
        unsigned get_page_mode = page_mode;
        page_mode_map.insert(std::pair<new_addr_type, unsigned>(key_chunk, get_page_mode));  // update page mode map
        return get_page_mode;
    }
}

unsigned tlb_tag_array::get_page_size(new_addr_type key) {
    new_addr_type key_chunk = key >> 9;  // shift hard coded
    auto find_chunk = page_mode_map.find(key_chunk);
    if (find_chunk != page_mode_map.end()){
        return page_mode_map.at(key_chunk);
    } else {
        assert(0 && "should not have reached here\n");
    }
}

void tlb_tag_array::debug() {
    new_addr_type debug_insn = static_cast<new_addr_type>(100000000) * (new_addr_type)m_debug_counter;
    if ((m_gpu->gpu_tot_sim_insn + m_gpu->gpu_sim_insn) > debug_insn){
        fprintf(stderr, "insn : %12lld, Page Mode : %3d\n", (m_gpu->gpu_tot_sim_insn + m_gpu->gpu_sim_insn), m_page_mode);

        float ipc = static_cast<float>(m_gpu->gpu_tot_sim_insn + m_gpu->gpu_sim_insn) / static_cast<float>(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
        float remote_ratio = static_cast<float>(m_mem_stats->tot_remote_access) / static_cast<float>(m_mem_stats->tot_local_access + m_mem_stats->tot_remote_access);

        unsigned total_alloc_result;
        std::vector<unsigned> chiplet_alloc_result;
        m_gpu_alloc->copy_alloc_result(total_alloc_result, chiplet_alloc_result);

        time_t curr_time;
        time(&curr_time);
        unsigned long long elapsed_time = m_gpu->get_elapsed_time();
        time_t days = elapsed_time / (3600 * 24);
        time_t hrs = elapsed_time / 3600 - 24 * days;
        time_t minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        time_t sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        fprintf(stderr, "insn : %12lld, ipc : %4.2f, remote_ratio : %2.2f, total page : %10d | elapsed = %u:%u:%02u:%02u / %s",
                (m_gpu->gpu_tot_sim_insn + m_gpu->gpu_sim_insn),
                ipc,
                remote_ratio,
                total_alloc_result,
                static_cast<unsigned>(days), static_cast<unsigned>(hrs), static_cast<unsigned>(minutes),
                static_cast<unsigned>(sec), ctime(&curr_time));

        if (m_config->enable_remote_debug) {
            unsigned tot_malloc_num = m_gpu_alloc->get_tot_malloc_num();
            for (unsigned i = 0; i < tot_malloc_num; i++){
                if (m_mem_stats->tot_access_malloc[i] != 0) {
                    remote_ratio = static_cast<float>(m_mem_stats->tot_remote_access_malloc[i]) / static_cast<float>(m_mem_stats->tot_access_malloc[i]);
                    fprintf(stderr, "Remote ratio - malloc %2d = %2.3f | count = %15ld\n", i, remote_ratio, m_mem_stats->tot_access_malloc[i]);
                }
            }
            fprintf(stderr, "Non-malloc access = %5ld\n", m_mem_stats->non_malloc_access);
        }

        fprintf(stderr, "\n");
        m_debug_counter++;
    }
}

/******************************************************************************/
unsigned tlb_tag_array::get_tlb_chiplet(new_addr_type key) {
    new_addr_type va_key = (new_addr_type)(key >> 9);  // shift hard coded
    new_addr_type tot_chiplet = static_cast<new_addr_type>(m_config->chiplet_num) - 1;

    new_addr_type tlb_chiplet = va_key & tot_chiplet;
    return static_cast<unsigned>(tlb_chiplet);
}

// set-associative index hash update
unsigned tlb_tag_array::get_tlb_index(new_addr_type key) {
    std::bitset<64> addr(key);
    std::bitset<32> index;
    index.reset();

    unsigned set_cnt = static_cast<unsigned>(log2(m_entries));
    unsigned index_cnt_x = 20;
    unsigned index_cnt_y = 27;
    unsigned index_cnt_z = 34;

    for (unsigned i = 0; i < set_cnt; i++) {
        index[i] = addr[index_cnt_x+i] ^ addr[index_cnt_y+i] ^ addr[index_cnt_z+i];
        index_cnt_x += 1;
        index_cnt_y += 1;
        index_cnt_z += 1;
    }

    unsigned result_index = index.to_ulong();
    assert(result_index < m_entries);
    return result_index;
}

void tlb_tag_array::done_tlb_req(mem_fetch * mf) {
    PageWalker * pageWalker = mf->get_page_walker();
    assert(pageWalker != nullptr);

    if (mf->get_parent_tlb_request() != nullptr){
        /* If PW cache hit */
        if (mf->get_pwcache_hit() && !mf->get_pwcache_done()){
            /* Put this request in the latency queue for a PW cache hit */
            this->chiplet_l2_tlb[mf->get_tlb_chiplet()]->page_walk_subsystem->page_walk_cache_enqueue(mf);
            return;
        } else if (mf->get_pwcache_hit() && mf->get_pwcache_done()){
            done_tlb_req(mf->get_parent_tlb_request());
            return;
        }

        /* Send the actual (original) memory request to DRAM */
        mf->get_cache()->add_tlb_miss_to_queue(mf);
    } else {  /* If the memory fetch is done */
        pageWalker->page_walk_return(mf);
    }
}

// TLB shootdown
void tlb_tag_array::flush_TLBs(uint64_t vpn_chunk) {
    for (unsigned i = 0; i < m_config->total_sm_num; i++){
        l1_tlb[i]->flush_private(vpn_chunk);
    }
    for (unsigned i = 0; i < m_tot_chiplet; i++){
        chiplet_l2_tlb[i]->flush_shared(vpn_chunk);
    }
};

void tlb_tag_array::flush_private(uint64_t vpn_chunk) {
    for (auto iter = tag_array->begin(); iter != tag_array->end(); ){
        new_addr_type tag_shift = 9 - iter->m_shift;
        new_addr_type check_tag = iter->m_tag >> tag_shift;
        if (vpn_chunk == check_tag) {
            tag_array->erase(iter);
        } else {
            ++iter;
        }
    }
}

void tlb_tag_array::flush_shared(uint64_t vpn_chunk) {
    page_walk_subsystem->flush(APP_ID);
    for (unsigned index = 0; index < m_entries; index++){
        std::list<tlb_entry> * probe_array = l2_tag_array[index];
        std::list<tlb_entry>::iterator iter;
        for (iter = probe_array->begin(); iter != probe_array->end(); ){
            new_addr_type tag_shift = 9 - iter->m_shift;
            new_addr_type check_tag = iter->m_tag >> tag_shift;
            if (vpn_chunk == check_tag) {
                probe_array->erase(iter);
            } else {
                ++iter;
            }
        }
    }
}