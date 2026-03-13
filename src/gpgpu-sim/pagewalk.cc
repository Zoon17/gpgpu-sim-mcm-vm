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

#include "pagewalk.h"
#include "mem_latency_stat.h"

/* Page Walk Sub System methods */
PageWalkSubsystem::PageWalkSubsystem(tlb_tag_array *tlb, mmu *page_manager,
                                     const memory_config *config, memory_stats_t *mem_stat) {

    this->config = config;
    this->mem_stat = mem_stat;
    L2_TLB = tlb;
    this->page_manager = page_manager;

    /* This order of construction needs to be maintained for correctness,
     * else NULL pointers may creep in and creep you out.
     * */
    page_walk_cache = new PageWalkCache(this, config, mem_stat);
    page_walk_queue = new PageWalkQueue(this, config, mem_stat);
    num_walkers = config->num_page_walkers;
    for (unsigned i = 0; i < num_walkers; i++) {
        page_walkers.push_back(new PageWalker(this, config, mem_stat, i));
    }
}

bool PageWalkSubsystem::enqueue(mem_fetch *mf) {
    if (page_walk_queue->enqueue(mf)) {
        return true;
    }
    else {
        L2_TLB->set_stall(true);
        return false;
    }
}

mem_fetch *PageWalkSubsystem::dequeue(PageWalker *page_walker) {
    mem_fetch *page_walk = nullptr;
    page_walk = page_walk_queue->dequeue(page_walker);
    L2_TLB->set_stall(false);
    return page_walk;
}

unsigned PageWalkSubsystem::get_num_walkers() {
    return num_walkers;
}

struct tlb_tag_array *PageWalkSubsystem::get_tlb() {
    return L2_TLB;
}

PageWalker *PageWalkSubsystem::get_idle_page_walker() {
    /* Assuming Baseline
     * Needs to incorporate partition information for DWS
     * */
    std::vector<PageWalker *>::iterator it;
    for (it = page_walkers.begin(); it != page_walkers.end(); it++) {
        if ((*it)->get_current() == NULL) {
            return (*it);
        }
    }
    return NULL;
}

void PageWalkSubsystem::service_page_walk_cache_queue() {
    page_walk_cache->service_latency_queue();
}

void PageWalkSubsystem::page_walk_cache_enqueue(mem_fetch *mf) {
    page_walk_cache->enqueue(mf);
}

/* Page Walker methods */
PageWalker::PageWalker(PageWalkSubsystem *page_walk_subsystem,
                       const memory_config *config, memory_stats_t *mem_stat, unsigned id) {

    this->page_walk_subsystem = page_walk_subsystem;
    this->config = config;
    this->mem_stat = mem_stat;
    this->id = id;
    this->previous_translation_finish_cycle = 0;
    current = NULL;
    appid = -1;
    page_walk_cache = page_walk_subsystem->page_walk_cache;
    current_is_stolen = false;  // not used
}

void PageWalker::initiate_page_walk(mem_fetch *mf) {
    /* Initiate the page walk using the page walk mechanishm */
    mem_fetch *pw = pt_walk(mf);
    pw->set_page_walker(this);
    this->current = mf;
    pw->propagate_walker(this);

    tlb_tag_array * mother_tlb = mf->get_tlb()->get_shared_tlb();
    mother_tlb->done_tlb_req(pw);
}

mem_fetch *PageWalker::get_current() {
    return current;
}

bool PageWalker::page_walk_return(mem_fetch *mf) {
    unsigned pw_time_stamp = mf->get_pw_timestamp();

    /* Part 1: Fill into TLB and finish off the page walk */
    mf->set_been_through_tlb(true);
    mf->set_tlb_miss(false);

    // return to per-chiplet L2 TLB
    tlb_tag_array * shared_mother_tlb = mf->get_tlb()->get_shared_tlb();
    mf->set_page_size(shared_mother_tlb->get_page_size(shared_mother_tlb->get_key(mf->get_addr(), mf->get_appID())));
    mf->get_tlb()->l2_fill(mf->get_addr(), mf->get_appID(), mf);

    memory_stats_t *mem_stat = this->mem_stat;
    mem_stat->pw_tot_num++;
    unsigned cur_pw_lat = static_cast<unsigned>(mf->get_time()) - pw_time_stamp;
    mem_stat->pw_tot_lat += cur_pw_lat;
    this->current = nullptr;

    /* Part 2: Start a new page walk if there are outstanding page walks to
     * service */
    mf = page_walk_subsystem->dequeue(this);

    if (mf) {
        initiate_page_walk(mf);
    }
    return true;
}


mem_fetch *PageWalker::pt_walk(mem_fetch *mf) {
    /* Done with setting up the last request in the PT walk routine */
    if (mf->get_tlb_depth_count() >= config->tlb_levels) {
        return mf;
    } else {
        mem_fetch *child;
        /* Set a new mem_fetch for the next level subroutine */
        child = new mem_fetch(mf);

        probe_pw_cache(child);  // always probe pw cache

        mf->set_child_tlb_request(child);
        /* Then, continue performing the page table walk for
         * the next level of TLB access */
        return pt_walk(mf->get_child_tlb_request());
    }
}

void PageWalker::probe_pw_cache(mem_fetch *mf) {
    assert(mf->get_tlb_depth_count() != 0);
    if (mf->get_tlb_depth_count() == 1 || (mf->get_tlb_depth_count() == 2 && mf->get_pt_walk_skip() == 1)) {
        mf->set_pwcache_hit(false);
    }
    else {
        memory_stats_t *mem_stat = this->mem_stat;
        mem_stat->pwc_tot_accesses++;
        unsigned addr_lvl = mf->get_tlb_depth_count();

        mem_stat->pwc_tot_addr_lvl_accesses[addr_lvl]++;

        if (page_walk_cache->access(mf)) {
            mf->set_pwcache_hit(true);
            mem_stat->pwc_tot_hits++;
            mem_stat->pwc_tot_addr_lvl_hits[addr_lvl]++;
        } else {
            mf->set_pwcache_hit(false);
            mem_stat->pwc_tot_misses++;
            mem_stat->pwc_tot_addr_lvl_misses[addr_lvl]++;
        }
    }
}

/* Page Walk Cache methods */
PageWalkCacheImpl::PageWalkCacheImpl(const memory_config *config, memory_stats_t *mem_stat) {

    this->config = config;
    this->mem_stat = mem_stat;

    ports = config->pw_cache_num_ports;

    pw_cache_entries = config->tlb_pw_cache_entries;
    pw_cache_ways = config->tlb_pw_cache_ways;
    pw_cache = new std::list<new_addr_type> *[pw_cache_entries];
    for (unsigned i = 0; i < pw_cache_entries; i++)
        pw_cache[i] = new std::list<new_addr_type>;

    pw_cache_lat_queue = new std::list<mem_fetch *>;
    pw_cache_lat_time = new std::list<unsigned long long>;
}

bool PageWalkCacheImpl::access(new_addr_type key, unsigned index) {
    std::list<new_addr_type>::iterator findIter =
            std::find(pw_cache[index]->begin(), pw_cache[index]->end(), key);
    if (findIter != pw_cache[index]->end())
    {
        pw_cache[index]->remove(key);
        pw_cache[index]->push_front(key);
        return true;
    } else {
        fill(key, index);
        return false;
    }
}

bool PageWalkCacheImpl::fill(new_addr_type key, unsigned index) {
    pw_cache[index]->remove(key);

    while (pw_cache[index]->size() >= pw_cache_ways) {
        pw_cache[index]->pop_back();
    }
    pw_cache[index]->push_front(key);
    return 0;
}

void PageWalkCacheImpl::service_latency_queue() {
    unsigned ports = 0;
    while (!pw_cache_lat_queue->empty()) {
        mem_fetch *mf = pw_cache_lat_queue->front();

        unsigned long long temp = pw_cache_lat_time->front();
        if ((ports < config->pw_cache_num_ports) &&
            ((temp + config->pw_cache_latency) <
             mf->get_time()))
        {
            // Remove mf from the list
            pw_cache_lat_time->pop_front();
            pw_cache_lat_queue->pop_front();

            // Finish up the current pw cache hit routine, call the next mf
            mf->set_pwcache_done(true);
            tlb_tag_array * mother_tlb = mf->get_tlb()->get_shared_tlb();
            mother_tlb->done_tlb_req(mf);
            ports++;
        } else{
            break;
        }
    }
}

void PageWalkCacheImpl::enqueue(mem_fetch *mf) {
    pw_cache_lat_queue->push_front(mf);
    pw_cache_lat_time->push_front(mf->get_timestamp());
}

void PageWalkCacheImpl::print() {
    int count = 0;
    for (unsigned index = 0; index < pw_cache_entries; index++) {
        for (std::list<new_addr_type>::iterator it =
                pw_cache[index]->begin();
             it != pw_cache[index]->end(); ++it) {
            printf("%llu\t", *it);
            count++;
        }
        printf("\n\n");
    }
    printf("entries in pw cache: %d\n", count);
}

PageWalkCache::PageWalkCache(PageWalkSubsystem *page_walk_subsystem,
                             const memory_config *config, memory_stats_t *mem_stat) {
    this->page_walk_subsystem = page_walk_subsystem;
    this->config = config;
    this->mem_stat = mem_stat;
    this->page_manager = page_walk_subsystem->page_manager;
    pw_cache_entries = config->tlb_pw_cache_entries;
    pw_cache_ways = config->tlb_pw_cache_ways;

    // This needs to change for multiple page walk caches.
    per_app_pwc_used = false;
    pwc = new PageWalkCacheImpl(config, mem_stat);

    ports = config->pw_cache_num_ports;
}

bool PageWalkCache::access(mem_fetch *mf) {
    new_addr_type key = get_key(mf);
    unsigned index = (mf->get_original_addr() >> (config->page_size)) &
                     (pw_cache_entries - 1);
    return pwc->access(key, index);
}

bool PageWalkCache::fill(mem_fetch *mf) {
    new_addr_type key = get_key(mf);
    unsigned index = (mf->get_original_addr() >> (config->page_size)) &
                     (pw_cache_entries - 1);
    return pwc->fill(key, index);
}

new_addr_type PageWalkCache::get_key(mem_fetch *mf) {
    uint64_t key = mf->get_original_addr();

    uint64_t bitmask = page_manager->get_bitmask(mf->get_tlb_depth_count());
    key = key & bitmask;
    /* Process the offset, bitmask should be increasingly longer as depth
     * increases. Then need to shift the bitmask */

    while ((bitmask > 0) && ((bitmask & 1) == 0)) {
        key = key >> 1;
        bitmask = bitmask >> 1;
    }
    return key;
}

void PageWalkCache::service_latency_queue() {
    pwc->service_latency_queue();
}

void PageWalkCache::enqueue(mem_fetch *mf) {
    pwc->enqueue(mf);
}

void PageWalkCache::print() {
    pwc->print();
}

/* Page Walk Queue methods */
PageWalkQueue::PageWalkQueue(PageWalkSubsystem *page_walk_subsystem,
                             const memory_config *config, memory_stats_t *mem_stat) {

    this->page_walk_subsystem = page_walk_subsystem;
    this->config = config;
    this->mem_stat = mem_stat;

    /* This needs to change for QoS provisioning of page walkers. */
    size = config->page_walk_queue_depth;
    per_walker_queue_used = false;
}

bool PageWalkQueue::enqueue(mem_fetch *mf) {
    /* This might have to be changed to enforce page walker partitioning */
    PageWalker *idle_page_walker = page_walk_subsystem->get_idle_page_walker();

    if (idle_page_walker) {
        mf->set_pw_timestamp(mf->get_time());
        idle_page_walker->initiate_page_walk(mf);
        return true;
    } else {
        assert(global_queue.size() <= size);
        if (global_queue.size() >= size) {
            return false;
        } else {
            mf->set_pw_timestamp(mf->get_time());
            global_queue.push_back(mf);
        }
        return true;
    }
}

mem_fetch *PageWalkQueue::dequeue(PageWalker *page_walker) {
    mem_fetch *page_walk;
    if (global_queue.size() == 0) {
        return nullptr;
    } else {
        page_walk = global_queue.front();
        global_queue.pop_front();
    }
    memory_stats_t *mem_stat = this->mem_stat;
    unsigned pw_queueing_lat = static_cast<unsigned>(page_walk->get_time()) - page_walk->get_pw_timestamp();
    mem_stat->pwq_tot_lat += pw_queueing_lat;
    return page_walk;
}

void PageWalkQueue::print() {
    printf("Page Walk Queue\n");
    for (std::list<mem_fetch *>::iterator it = global_queue.begin();
         it != global_queue.end(); it++)
    {
        printf("Entry: %p\n", (*it));
    }
    printf("\n");
}

void PageWalkSubsystem::flush(unsigned appid) {
    page_walk_cache->flush(appid);
}

void PageWalkCacheImpl::flush(unsigned appid) {
    for (unsigned i = 0; i < pw_cache_entries; i++) {
        for (auto j = pw_cache[i]->begin(); j != pw_cache[i]->end();) {
            j = pw_cache[i]->erase(j);
        }
    }
}
