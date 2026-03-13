// Created by junhyeok on 2/17/23.
// Copyright (c) 2023-2026 Junhyeok Park. All rights reserved.
//
// Portions of the initial logic for virtual memory support were referenced from
// 1. Mosaic (CMU SAFARI, MICRO'17) / MASK (CMU SAFARI, ASPLOS'18)
// This file has been modified, optimized, and extended to improve functionality.
//
// Built for compatibility with the GPGPU-Sim framework.
// Licensed under the BSD 3-Clause License.
// See the COPYRIGHT file in the project root for full license text.

#include <string.h>
#include "memory_owner.h"
#include "gpu-sim.h"
#include <math.h>
#include "dram.h"
#include "tlb.h"
#include "mem_latency_stat.h"
#include "../cuda-sim/cuda-sim.h"

#define ALLOC_DEBUG_SHORT 0
#define ALLOC_DEBUG 0
#define ALLOC_DEBUG_LONG 0
//Enable this will print fault queue entries
#define GET_FAULT_STATUS 1
#define RC_DEBUG 1

//Print to check that page->utilization is correct
#define UTIL_CHECK 0
#define PT_DEBUG 0
#define MCM_DEBUG 0

page_table::page_table(const memory_config * config, unsigned parent_level, mmu * main_mmu) {
    fprintf(stdout, "Initializing page_table of level %d\n", parent_level);
    m_config = config;
    m_mmu = main_mmu;
    m_size = 0;
    // m_appID = appID;
    if (parent_level < m_config->tlb_levels - 1) {
        fprintf(stdout, "Setting pointers current level %d to the next page_table of level %d\n", parent_level,parent_level + 1);
        next_level = new page_table(m_config, parent_level + 1, main_mmu);
    } else {
        next_level = nullptr;
    }
    current_level = parent_level + 1;
    parse_bitmask();

    m_pte_page_address_64KB.clear();
}

// Parse bitmask for page table walk. Can use the page_sizes instead (above), but this seems more flexible. Downside is the two config (va_mask), tlb_levels and page_sizes has to match.
void page_table::parse_bitmask() {
    std::string mask(m_config->va_mask);
    std::string mask2(m_config->va_mask);
    for(unsigned i = 1; i <= m_config->tlb_levels;i++)
    {
        if (i <= current_level-1) {
            std::replace(mask.begin(),mask.end(),(char)(i+'0'),'0');
        }
        else {
            std::replace(mask.begin(),mask.end(),(char)(i+'0'),'1');
        }
        if (i == current_level) {
            std::replace(mask2.begin(),mask2.end(),(char)(i+'0'),'1');
        }
        else {
            std::replace(mask2.begin(),mask2.end(),(char)(i+'0'),'0');
        }
    }
    std::bitset<48> temp(mask);
    std::bitset<48> temp2(mask2);
    m_bitmask = temp.to_ulong();
    m_bitmask_pw = temp2.to_ulong();
    printf("Converting VA bitmask for page_table translation for level = %d, original string = %s, results = %lx, "
           "mask_string = %s, pwcache_offset_mask = %lx, mask_string = %s\n",
           current_level, m_config->va_mask, m_bitmask, mask.c_str(), m_bitmask_pw, mask2.c_str());

    // set page table offset mask
    new_addr_type total_offset = (new_addr_type)511;  // 9-bit mask
    new_addr_type offset_shift = (static_cast<new_addr_type>(current_level - 1)
        * static_cast<new_addr_type>(9)) + static_cast<new_addr_type>(12);  // page table level shift + page offset shift, shift hard coded
    m_pt_offset_shift = offset_shift;
    m_pt_offset_mask = (new_addr_type)(total_offset << offset_shift);
    printf("Page table offset shift = %lld, offset mask = %llx\n", m_pt_offset_shift, m_pt_offset_mask);

    m_pt_address_mask = ((new_addr_type)((new_addr_type)(m_bitmask >> (m_pt_offset_shift + static_cast<new_addr_type>(9))))
        << (m_pt_offset_shift + static_cast<new_addr_type>(9)));  // shift hard coded
    printf("Page table address mask = %llx\n", m_pt_address_mask);

    // support for 64KB PTE
    if (current_level == 1) {
        new_addr_type total_offset_64KB = (new_addr_type)31;  // 5-bit mask
        new_addr_type offset_shift_64KB = static_cast<new_addr_type>(12 + 4);
        m_pt_offset_shift_64KB = offset_shift_64KB;
        m_pt_offset_mask_64KB  = (new_addr_type)(total_offset_64KB << offset_shift_64KB);  // shift hard coded

        printf("Page table offset shift_64KB = %lld, offset mask_64KB = %llx\n", m_pt_offset_shift_64KB, m_pt_offset_mask_64KB);
    }
}

uint64_t page_table::get_bitmask(int level) {
    if (static_cast<unsigned>(level) == current_level) {
        return m_bitmask;
        //return m_bitmask_pw;
    }
    else if (next_level == nullptr) {
        return m_bitmask_pw;
        //return m_bitmask;
    }
    else {
        return next_level->get_bitmask(level);
    }
}

void page_table::set_page_in_DRAM(page * this_page) {
    //Only mark it as inDRAM when used for non-TLB-related pages
    if (this_page->appID != PT_SPACE){
        std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(this_page->starting_addr & m_bitmask);
        if (itr != entries.end())
        {
            itr->second->inDRAM = true;
            itr->second->appID = this_page->appID;
        }
        if (next_level != nullptr) next_level->set_page_in_DRAM(this_page);
    }
}

//This is called from memory.cc, add a mapping between virtual address to physical address
//i.e, this populates the entries (map of addr and actual entries) of each level of page table
//Each entry contain whether the page is in DRAM or not, whether this is a leaf (in case of
//a superpage, leaf node would be at an earlier level, valid flag (nodes below leaf nodes should
//not be valid.
// Once this mapping is set, parse_pa, should be able to simply return the page_table_entry
// back to mem_fetch to process its request.
#define PTE_DEBUG 0
page_table_entry * page_table::add_entry(new_addr_type address, int appID, bool isRead,
    unsigned chiplet, unsigned pte_skip, unsigned page_mode) {
    if (next_level != nullptr) {
        // Propagate entires across multiple levels
        page_table_entry* alloc_pte = next_level->add_entry(address,appID, isRead, chiplet, pte_skip, page_mode);
        if (pte_skip == current_level) return alloc_pte;
    }

    unsigned alloc_chiplet = chiplet;
    if (m_config->enable_pte_rr) {
        alloc_chiplet = get_pte_alloc_rr();
    }
    new_addr_type key = address & m_bitmask;
    page_table_entry * temp;
    std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(key);
    if (MCM_DEBUG) printf("Adding page table entry for address = %llx, current level key = %llx, bitmask = %lx\n",address, key, m_bitmask);
    if (itr == entries.end()) {  //Entry is not in the page table, page fault
        // new pte allocation
        if (page_mode == PAGE_64K && current_level == 1) {
            new_addr_type pt_address = address & m_pt_address_mask;
            std::map<new_addr_type, pt_page_info>::iterator find_pt_address =
                page_table_address_map_64KB.find(pt_address);
            if (find_pt_address == page_table_address_map_64KB.end()) {
                if (m_pte_page_address_64KB.empty()) {
                    page *new_page;
                    new_page = m_mmu->get_DRAM_layout()->allocate_free_page(
                        m_config->base_page_size, PT_SPACE, alloc_chiplet);
                    new_addr_type new_page_addr = new_page->starting_addr;
                    page_table_address_map_64KB.insert(
                        std::pair<new_addr_type, pt_page_info>(
                            pt_address,
                            std::pair<new_addr_type, unsigned>(new_page_addr, 0)));

                    for (new_addr_type i = 32; i < 512; i = i + 32) {
                        new_addr_type pte_page_address = new_page_addr + i * static_cast<new_addr_type>(8);
                        m_pte_page_address_64KB.push_back(pte_page_address);
                    }
                } else {
                    new_addr_type pte_page_addr = m_pte_page_address_64KB.front();
                    m_pte_page_address_64KB.pop_front();

                    page_table_address_map_64KB.insert(
                        std::pair<new_addr_type, pt_page_info>(
                            pt_address,
                            std::pair<new_addr_type, unsigned>(pte_page_addr, 0)));
                }
            }

            pt_page_info &get_info = page_table_address_map_64KB.at(pt_address);
            assert(get_info.second < 32);

            new_addr_type current_pt_address = get_info.first;
            // calculate page table offset
            new_addr_type pt_offset =
                (new_addr_type)((new_addr_type)(address & m_pt_offset_mask_64KB) >>
                                m_pt_offset_shift_64KB);
            assert(pt_offset < 32);
            new_addr_type pte_address =
                (new_addr_type)current_pt_address +
                static_cast<new_addr_type>(8) * pt_offset;

            if (PTE_DEBUG)
                printf(
                    "PTE level = %d, bitmask = %lx, Chiplet = %d, Page addr = %llx, Virtual addr = %llx, PTE offset = %llx, PTE addr = %llx\n",
                    current_level, m_bitmask, alloc_chiplet, current_pt_address, address,
                    pt_offset, pte_address);

            temp = new page_table_entry(key, pte_address, this);

            get_info.second++;
            temp->appID = appID;
            temp->isRead = true;
            temp->inDRAM = false;
            entries.insert(std::pair<uint64_t ,page_table_entry*>(key,temp));
        } else {
            new_addr_type pt_address = address & m_pt_address_mask;
            std::map<new_addr_type, pt_page_info>::iterator find_pt_address =
                page_table_address_map.find(pt_address);
            if (find_pt_address == page_table_address_map.end()) {
                // fetch new page
                page *new_page;
                new_page = m_mmu->get_DRAM_layout()->allocate_free_page(
                    m_config->base_page_size, PT_SPACE, alloc_chiplet);

                new_addr_type new_page_addr = new_page->starting_addr;
                page_table_address_map.insert(
                    std::pair<new_addr_type, pt_page_info>(
                        pt_address,
                        std::pair<new_addr_type, unsigned>(new_page_addr, 0)));
            }

            pt_page_info &get_info = page_table_address_map.at(pt_address);
            assert(get_info.second < 512);  // 8B * 512 = 4096 (4KB)

            new_addr_type current_pt_address = get_info.first;
            // calculate page table offset
            new_addr_type pt_offset =
                (new_addr_type)((new_addr_type)(address & m_pt_offset_mask) >>
                                m_pt_offset_shift);
            assert(pt_offset < 512);
            new_addr_type pte_address =
                (new_addr_type)current_pt_address +
                static_cast<new_addr_type>(8) * (new_addr_type)(pt_offset);

            if (PTE_DEBUG)
                printf(
                    "PTE level = %d, bitmask = %lx, Chiplet = %d, Page addr = %llx, Virtual addr = %llx, PTE offset = %llx, PTE addr = %llx\n",
                    current_level, m_bitmask, alloc_chiplet, current_pt_address, address,
                    pt_offset, pte_address);

            temp = new page_table_entry(key, pte_address, this);

            get_info.second++;
            temp->appID = appID;
            temp->isRead = true;
            temp->inDRAM = false;
            entries.insert(std::pair<uint64_t ,page_table_entry*>(key,temp));
        }
    }
    else {
        if (PT_DEBUG) printf("Found the page table entry for address = %llx, current level key = %llx, key = %llx, address = %llx\n",address, key, itr->second->m_key, itr->second->m_addr);
        temp = itr->second;
    }
    return temp;
}

// Find the address for tlb-related data by going through the page table entries of each level
new_addr_type page_table::parse_pa(mem_fetch * mf) {
    uint64_t key = mf->get_page_addr() & m_bitmask;
    if (PT_DEBUG) printf("Parsing PA for address = %llx, current level = %d, key = %lx, mf->addr = %llx\n", mf->get_page_addr(),mf->get_tlb_depth_count(),key,mf->get_addr());

    if (mf->get_page_size() >= PAGE_2M && current_level == 1) {  // In case of 2MB pages, skip last level pte
        assert(next_level != nullptr);
        return next_level->parse_pa(mf);
    }

    std::map<new_addr_type,page_table_entry*>::iterator itr = entries.find(key);
    if (itr == entries.end()) {
        assert(0);
    }
    else {
        unsigned check_level = mf->get_tlb_depth_count() + 1;
        if (mf->get_pt_walk_skip() != 0 && mf->get_tlb_depth_count() == 0) {
            check_level += mf->get_pt_walk_skip();
        }

        if (check_level == current_level) {
            return itr->second->m_addr;
        }
        else {
            assert(next_level != nullptr);
            return next_level->parse_pa(mf);
        }
    }
}

////////////// Physical DRAM layout (used be MMU) ///////////////
DRAM_layout::DRAM_layout(const class memory_config * config, page_table * root, class gpgpu_sim * gpu) {
    m_gpu = gpu;
    //Parse page size into m_size_count and page_size
    if (ALLOC_DEBUG)  printf("Initialing DRAM physical structure\n");
    m_config = config;
    DRAM_size = (uint64_t)m_config->DRAM_size;

    m_chiplet = m_config->chiplet_num;

    m_page_size = m_config->page_sizes;

    m_pt_root = root;

    m_page_root = new page();
    m_page_root->starting_addr = 0;
    m_page_root->used = false;
    m_page_root->size = m_config->DRAM_size;
    m_page_root->dataPresent = false;
    m_page_root->appID = NOAPP;
    m_page_root->sub_pages = new std::deque<page*>();

    // define channel/bank/sa mapping
    addrdec_t from_raw_addr;

    m_config->m_address_mapping.addrdec_tlx(0,&from_raw_addr, NOAPP, DRAM_CMD, 0, 0); //m_page_root
    m_page_root->channel_id = from_raw_addr.chip;
    m_page_root->bank_id = from_raw_addr.bk;
    m_page_root->sa_id = -1;  // not used

    //Initialize interface to DRAM memory channels
    dram_channel_interface = new dram_t*[m_config->m_n_mem];

    for (unsigned i = 0; i < m_page_size->size(); i++) {
        free_pages[i] = new std::list<page*>();
        all_page_list[i] = new std::list<page*>();
    }

    chiplet_free_pages = new std::map<uint64_t, std::list<page*>*> [m_chiplet];
    chiplet_active_huge_block = new std::map<uint64_t,page*> [m_chiplet];
    for (auto i = 0u; i < m_chiplet; i++){
        for (auto p = 0u; p < m_page_size->size(); p++){
            chiplet_free_pages[i][p] = new std::list<page*>();
        }
    }

    free_pages[0] = new std::list<page*>();
    all_page_list[0] = new std::list<page*>();
    free_pages[0]->push_front(m_page_root);
    all_page_list[0]->push_front(m_page_root);

    //Populate all the possible mapping
    initialize_pages(m_page_root, 1);

    for(uint64_t i = 0; i < static_cast<uint64_t>(1); i++)  // only one application support for page table
    {
        occupied_pages[i] = new std::list<page*>();
        active_huge_block[i] = free_pages[m_page_size->size()-2]->front();
        free_pages[m_page_size->size()-2]->pop_front();
        //Grant the first n huge blocks to each app

        for (auto c = 0u; c < m_chiplet; c++){
            chiplet_active_huge_block[c][i] = chiplet_free_pages[c][m_page_size->size()-2]->front();
            chiplet_free_pages[c][m_page_size->size()-2]->pop_front();
        }
    }
    active_huge_block[PT_SPACE] = free_pages[m_page_size->size()-2]->front();
    free_pages[m_page_size->size()-2]->pop_front();
    occupied_pages[PT_SPACE] = new std::list<page*>();
    // List of bloated pages
    occupied_pages[NOAPP] = new std::list<page*>();
    occupied_pages[MIXAPP] = new std::list<page*>();

    for (auto i = 0u; i < m_chiplet; i++){
        chiplet_active_huge_block[i][PT_SPACE] = chiplet_free_pages[i][m_page_size->size()-2]->front();
        chiplet_free_pages[i][m_page_size->size()-2]->pop_front();  // need this line?
    }
    if (ALLOC_DEBUG)  printf("Done initialing DRAM physical structure\n");
}

void DRAM_layout::initialize_pages(page * this_page, unsigned size_index) {
    // Propagate this sub_page
    page * temp;
    for (new_addr_type i = 0; i < ((new_addr_type)(*m_page_size)[size_index-1] / (new_addr_type)(*m_page_size)[size_index]); i++) //Initialize all sub-pages under this page
    {
        temp = new page();
        // Physical data -- Can't be changed

        temp->starting_addr = this_page->starting_addr + ((new_addr_type)i * (new_addr_type)((*m_page_size)[size_index]));
        // Metadata, can be changed
        temp->used = false;
        temp->size = (*m_page_size)[size_index];
        temp->dataPresent = false;
        temp->appID = NOAPP;
        temp->sub_pages = new std::deque<page*>();

        // define channel/bank/sa mapping -- Also physical data -- Can't be changed
        addrdec_t from_raw_addr;
        m_config->m_address_mapping.addrdec_tlx(temp->starting_addr,&from_raw_addr, NOAPP, DRAM_CMD, 0, 0);
        temp->channel_id = from_raw_addr.chip;
        temp->bank_id = from_raw_addr.bk;
        temp->sa_id = -1;  // not used

        //Add this page to the free page list
        if (ALLOC_DEBUG)  printf("Initialing free page list of page of size %lu, starting address = %llx, parent_page = %llx\n",(*m_page_size)[size_index],temp->starting_addr, this_page->starting_addr);
        free_pages[size_index]->push_front(temp);
        all_page_list[size_index]->push_front(temp);
        if (temp->size == (*m_page_size)[m_page_size->size()-1]) //If this is the smallest page, add this page to a map of all small pages
        {
            if (ALLOC_DEBUG)  printf("Adding a leaf page of size %lu, starting address = %llx, parent_page = %llx to the all page map\n",(*m_page_size)[size_index],temp->starting_addr, this_page->starting_addr);
            all_page_map[temp->starting_addr] = temp; //Used by gpgpu-sim to find Page * based on PA
        }

        // chiplet free page mapping
        uint64_t base_DRAM_chunk = (uint64_t)DRAM_size / m_chiplet;
        uint64_t index = (temp->starting_addr / base_DRAM_chunk);
        assert(index < m_chiplet);
        chiplet_free_pages[index][size_index]->push_front(temp);

        this_page->sub_pages->push_front(temp); // add subpages
        this_page->utilization = 0.0;
        if (size_index >= (m_page_size->size()-  1)) { //If this is a leaf node
            temp->used = true;
            temp->utilization = 0;
        }
        else {
            initialize_pages(temp,size_index + 1);
        }
    }
}

page * DRAM_layout::find_page_from_pa(new_addr_type pa) {
    new_addr_type pa_base = (pa / (*m_page_size)[m_page_size->size()-1]) * (*m_page_size)[m_page_size->size()-1];  // remove page offset
    page * res = all_page_map[pa_base];
    if (ALLOC_DEBUG){
        if (res == NULL) printf("Searching for a page using PA: Cannot find the page for PA = %llx, searched key = %llx\n", pa, pa_base);
        else printf("Searching for a page using PA: Found the page for PA = %llx, searched key = %llx, VA = %llx, appID = %d, size = %lu\n", pa, pa_base, res->va_page_addr, res->appID, res->size);
    }
    return res;
}

void DRAM_layout::set_DRAM_channel(dram_t * dram_channel, int channel_id) {
    //fprintf(stdout, "Setting DRAM interface in the MMU for channel id %d\n", channel_id);
    dram_channel_interface[channel_id] = dram_channel;
}

void DRAM_layout::set_stat(memory_stats_t * stat) {
    //fprintf(stdout, "Setting stat object in DRAM layout\n");
    m_stats = stat;
}

// Return a free page within this huge block. Called from allocate_free_page
page * DRAM_layout::get_free_base_page(page * parent) {
    page * free = nullptr;
    std::deque<page*> * temp = parent->sub_pages;
    if (temp == nullptr) {
        assert(0 && "Never reach here !!");
    }
    for (std::deque<page*>::iterator itr = temp->begin(); itr!=temp->end(); ++itr) {
        if ((*itr)->appID == NOAPP) {
            free = *itr;
            break;
        }
    }
    return free;
}

//Return a free page if a certain size
//This part now cause wierd syscall error
page * DRAM_layout::allocate_free_page(unsigned size, int appID, unsigned chiplet) {
    assert(size = 4096);
    page *return_page = nullptr;
    if (MCM_DEBUG)
        printf(
            "Allocating a page of size = %u, for appID = %d, free page exist, free page size = %lu\n",
            size, appID, chiplet_free_pages[chiplet][m_page_size->size()-1]->size());
    if ((chiplet_free_pages[chiplet][m_page_size->size()-1]->size()) > 0) {
        if (ALLOC_DEBUG)
            printf(
                "Trying to grab the front of free page list, size = %lu, front entry is at %llx, back is at %llx\n",
                chiplet_free_pages[chiplet][m_page_size->size()-1]->size(),
                chiplet_free_pages[chiplet][m_page_size->size()-1]->front()->starting_addr,
                chiplet_free_pages[chiplet][m_page_size->size()-1]->back()->starting_addr);
        // Grab the free page of a certain size
        if (appID == PT_SPACE)
        {
            m_stats->pt_space_size =
                m_stats->pt_space_size + m_config->base_page_size;
        }
        // For a normal request, first get the current parent page that handle this appID current huge block range
        page *parent =
            chiplet_active_huge_block[chiplet][appID];  // large page block - 2MB
        // Then grab a free page within this huge block
        return_page = get_free_base_page(parent);
        if (return_page == nullptr)  // If there is no more free page in this huge range
        {
            chiplet_active_huge_block[chiplet][appID] = chiplet_free_pages[chiplet][m_page_size->size()-2]->front();
            chiplet_free_pages[chiplet][m_page_size->size()-2]->pop_front();
            parent = chiplet_active_huge_block[chiplet][appID];
            return_page = get_free_base_page(parent);  // Get the free page
        }
        // Remove this page from the free page list
        chiplet_free_pages[chiplet][m_page_size->size()-1]->pop_front();
        if (MCM_DEBUG)
            printf(
                "Returning a page of size = %u, page starting address = %llx, free page size is now at = %lu, appID = %d, freepage_list_front is %llx, back is %llx\n",
                size, return_page->starting_addr, chiplet_free_pages[chiplet][m_page_size->size()-1]->size(),
                appID, chiplet_free_pages[chiplet][m_page_size->size()-1]->front()->starting_addr,
                chiplet_free_pages[chiplet][m_page_size->size()-1]->back()->starting_addr);

        // Add this page to app
        return_page->used = true;
        return_page->dataPresent = true;
        return_page->appID = appID;
        return_page->utilization = 1.0;
        occupied_pages[appID]->push_back(return_page);

        if (MCM_DEBUG)
            printf("Setting return page as used and data present\n");

        // Mark the entry in page table that the page is in DRAM
        m_pt_root->set_page_in_DRAM(
            return_page);  // Config parameters can have discrepancy between va_mask and page_sizes
    } else {
        assert(0);
    }
    return return_page;  // Return null if there are no more free page of this size
}

////////////// MMU ///////////////
mmu::mmu() {
    need_init = true;
}

void mmu::init(const class memory_config * config) {
    printf("Initializing MMU object - init2\n");
    m_config = config;
}

void mmu::init2(const class memory_config * config) {
    printf("Initializing MMU object - init\n");
    m_config = config;

    //Initialize the page table object
    printf("Setting up page tables objects\n");
    m_pt_root = new page_table(m_config, 0, this);
    printf("Done setting up page tables, setting up DRAM physical layout\n");

    va_to_pa_mapping = new std::map<new_addr_type,new_addr_type>();
    va_to_page_mapping = new std::map<new_addr_type,page*>();

    //Initialize page mapping. All pages are now free, all leaf are at the base page size
    assert(m_gpu != nullptr);
    m_DRAM = new DRAM_layout(m_config, m_pt_root, m_gpu);

    printf("Done setting up MMU\nSending pending promotion/demotion requests to TLBs\n");
}

void mmu::set_ready() {
    printf("Setting the MMU object as ready\n");
    need_init = false;
}

void mmu::set_m_gpu(class gpgpu_sim *gpu) {
    m_gpu = gpu;
    this->m_gpu_alloc = gpu->m_gpu_alloc;
}

new_addr_type mmu::get_pa(new_addr_type addr, int appID, bool isRead, unsigned chiplet){
    bool fault;
    return get_pa(addr, appID, &fault, isRead, chiplet);
}

bool mmu::allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID,
    unsigned int chiplet, unsigned page_mode) {
    page * target_page = m_DRAM->allocate_PA(va_base, pa_base, appID, chiplet, page_mode);

    if (target_page != nullptr){
        unsigned searchID = va_base | appID;
        (*va_to_page_mapping)[searchID] = target_page;
        return true;
    }
    assert(0 && "this never happen !!\n");  // manages all physical pages at initialization
}

page * DRAM_layout::allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID,
    unsigned chiplet, unsigned page_mode) {
    //Grab this physical page, check if it is occupied by PT
    page * target_page = find_page_from_pa(pa_base);

    if (target_page == nullptr) {
        // Never reach here in normal cases
        assert(0 && "Never happen : memory allocation error !!");
    }

    //If so, return false, do nothing
    if (target_page->appID == PT_SPACE) {
        assert(0 && "Never happen : memory allocation error !!");
    }

    //Otherwise, allocate the page, update page metadata, create the PTE entry for this page
    target_page->va_page_addr = va_base;
    target_page->appID = appID;
    target_page->utilization = 0.0;
    target_page->dataPresent = false;
    target_page->used = true;
    target_page->last_accessed_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    target_page->page_mode = page_mode;

    unsigned pte_skip;
    if (page_mode < PAGE_2M) {
        pte_skip = 0;
    } else if (page_mode == PAGE_2M) {
        pte_skip = 1;
    } else {
        assert(0 && "Not supported yet!");
    }

    //Create PTE entry for this page
    m_pt_root->add_entry(va_base, appID, true, chiplet, pte_skip, page_mode);

    //Return the page to MMU so it can get the mapping between VA and page
    return target_page;
}

new_addr_type mmu::get_pa(new_addr_type addr, int appID, bool * fault, bool isRead, unsigned chiplet) {
    page * this_page =  nullptr;
    unsigned searchID = ((addr >> m_config->page_size) << m_config->page_size) | appID;

    //Check if this page has their own PT entries setup or not. (VA page seen before, VA not seen)
    if (va_to_page_mapping->find(searchID) != va_to_page_mapping->end()) {
        if (ALLOC_DEBUG || ALLOC_DEBUG_SHORT) printf("Searching the map (searchID = %x) for physical page for VA = %llx, app = %d. Not the first time access.\n",searchID, addr, appID);
    }
    else {
        m_gpu_alloc->translate(appID, reinterpret_cast<void *>(addr), chiplet); //Note that at this point, addr should exist because addrdec.cc should have already handled any additional allocations
    }

    this_page = (*va_to_page_mapping)[searchID];
    //if (this_page != nullptr) {
    //    // Updates the base page's metadata
    //    this_page->last_accessed_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle; //Update last accessed time
    //    this_page->dataPresent = true;
    //    this_page->utilization = 1.0; //Might want to use a bit vector in the future to represent each cache line in the small page range. Might be an overkiltar
    //}

    new_addr_type result = reinterpret_cast<new_addr_type>(m_gpu_alloc->translate(appID,
        reinterpret_cast<void *>(addr), chiplet));

    return result; //Get the correct physical address
}

void mmu::set_L2_tlb(tlb_tag_array * L2TLB) {
    printf("Setting L2 TLB for the MMU at address = %p\n", L2TLB);
    l2_tlb = L2TLB;
}

uint64_t mmu::get_bitmask(int level) {
    return m_pt_root->get_bitmask(level);
}

void mmu::set_stat(memory_stats_t * stat) {
    m_stats = stat;
    printf("Setting stat object in MMU\n");
    m_DRAM->set_stat(stat);
}

void mmu::set_DRAM_channel(dram_t * dram_channel, int channel_id) {
    m_DRAM->set_DRAM_channel(dram_channel,channel_id);
}

unsigned page_table::get_pte_alloc_rr() {
    unsigned current_chiplet = m_pte_chiplet;
    m_pte_chiplet++;
    if (m_pte_chiplet >= m_config->chiplet_num) {
        m_pte_chiplet = 0;
    }
    return current_chiplet;
}