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

#ifndef ACCELSIM_MEMORY_OWNER_H
#define ACCELSIM_MEMORY_OWNER_H

//This number represent no app is occupying the page
#define NOAPP 9999999
#define MIXAPP 123456789
#define PT_SPACE 987654321

#include <list>
#include <map>
#include <utility>

#include <unordered_map>
#include "../abstract_hardware_model.h"

class Hub;
class dram_t;
class tlb_tag_array;
class mem_fetch;
class memory_stats_t;

struct page_metadata{
    page_metadata(unsigned long long parent_last_accessed_time, unsigned long long child_last_accessed_time,
        int appID, float util) {
        parent_last_accessed = parent_last_accessed_time;
        child_last_accessed = child_last_accessed_time;
        accessed_app = appID;
        utilization = util;
    };
    unsigned long long child_last_accessed;
    unsigned long long parent_last_accessed;
    unsigned accessed_app; //If multiple apps are in this range, return MIXAPP
    float utilization; //number of pages touched / (total possible num pages within the large page)
};

class page_table;

struct page_table_entry {
    page_table_entry(new_addr_type key, new_addr_type address, page_table * parent) {
        m_key = key;
        m_addr = address;
        appID = NOAPP;
        isRead = true;
        inDRAM = false;
        parent_pt = parent; //So that leaf nodes (hence, page object) can access every entry/level of the page table if needed
    }
    new_addr_type m_key;
    new_addr_type m_addr;
    int appID;
    bool isRead;
    bool inDRAM;
    page_table * parent_pt;
};

class mmu;
struct page;

class dram_cmd;

class page_table {
public:
    page_table(const class memory_config * config, unsigned parent_level, /*int appID, */ mmu * main_mmu);
    ~page_table();

    void parse_bitmask();
    uint64_t get_bitmask(int level);  // 4 level extension
    new_addr_type parse_pa(mem_fetch * mf); //Need to make sure parse_pa is aware of tlb-related addr

    // Used by DRAM_layout, whenever a page is grabbed from a free-page list, mark this page as in-DRAM
    void set_page_in_DRAM(page * this_page);

    // PTE page allocation
    page_table_entry * add_entry(new_addr_type address, int appID, bool isRead, unsigned chiplet, unsigned pte_level,
        unsigned page_mode);
    unsigned get_pte_alloc_rr();

private:
    unsigned m_pte_chiplet = 0;

    //Update page table after a DRAM copy
    std::map<new_addr_type, page_table_entry*> entries;
    page_table * next_level;
    unsigned current_level;
    const memory_config * m_config;
    uint64_t m_bitmask;  // bitmask extension
    uint64_t m_bitmask_pw;  // bitmask extension

    typedef std::pair<new_addr_type, unsigned> pt_page_info;
    std::map<new_addr_type, pt_page_info> page_table_address_map;
    std::map<new_addr_type, pt_page_info> page_table_address_map_64KB;

    new_addr_type m_pt_address_mask = 0;  // page table address mask
    new_addr_type m_pt_offset_shift = 0;  // page table offset shift
    new_addr_type m_pt_offset_mask  = 0;  // page table offset mask

    new_addr_type m_pt_offset_shift_64KB = 0;
    new_addr_type m_pt_offset_mask_64KB  = 0;
    std::deque<new_addr_type> m_pte_page_address_64KB;


    unsigned m_size; //Current size of the page table entries for this level
    new_addr_type current_fillable_address; //Current entries that a new entry can fill into

    unsigned address_space_ID;
    unsigned m_appID;

    unsigned addr_count;

    mmu * m_mmu;
};

// physical page structure
struct page{
    page() {
        dataPresent = false;
        size = 0;
        starting_addr = 0;
        va_page_addr = 0;
        used = false;
        appID = NOAPP;
        page_mode = 0;
        sub_pages = nullptr;
        utilization = 0.0;
        channel_id = -1;
        bank_id = -1;
        sa_id = -1;
        last_accessed_time = 0;
    }

    new_addr_type starting_addr; //Starting physical address
    new_addr_type va_page_addr; //VA page address. Used to check when we want to find the page
    uint64_t size;
    bool dataPresent; //True means not a free page, actual data present
    bool used; //Actual page node is here, false means the page is not active page (a subpage of an active superpage)
    int appID;
    unsigned page_mode;

    std::deque<page*> * sub_pages;
    //Point to the page table level that represent this page size
    //Help make it a bit easier to coalesce a page and when we want to update the content of page table
    //This is initialized in the beginning
    // For stat collection, keep track of how many of the super page are actually being used
    float utilization;

    //Easy to track these here instead of calling a function to find the mapping everytime.
    unsigned channel_id;
    unsigned bank_id;
    unsigned sa_id;

    unsigned long long last_accessed_time;
};

// DRAM layout object takes care of the physical mapping of pages in DRAM
class DRAM_layout{
    unsigned m_chiplet;

    std::map<uint64_t ,std::list<page*>*> free_pages;  // Set of free page, each entry is for differing page size
    std::map<uint64_t, std::list<page*>*>* chiplet_free_pages;

    std::map<uint64_t ,std::list<page*>*> occupied_pages;  // List of occupied pages, each entry denote the appID. Note that occupied_pages[NOAPP] means free pages within the coalesce range (bloat)

    std::map<uint64_t ,std::list<page*>*> all_page_list;  // This is a map that contains an ordered list for all the pages in DRAM.
    std::map<uint64_t ,page*> all_page_map;  // This is a map for PA_base to all small pages.

    // This map point to the current free huge block for each app (so that they hand out pages in huge block region for different apps.
    std::map<uint64_t,page*> active_huge_block;
    std::map<uint64_t,page*>* chiplet_active_huge_block;

    page_table * m_pt_root;
    std::vector<uint64_t> * m_page_size; //Array containing list of possible page sizes
    int m_page_size_count;

    page * m_page_root;
    uint64_t DRAM_size;

    class gpgpu_sim * m_gpu;
    memory_stats_t * m_stats;
    const memory_config * m_config;

    dram_t ** dram_channel_interface;
public:
    // Initialize free page list and occupied page
    DRAM_layout(const class memory_config * config, page_table * root, class gpgpu_sim * gpu);
    void initialize_pages(page * this_page, unsigned size_index);

    // Find a page using PA
    page * find_page_from_pa(new_addr_type pa);

    // Search parent for a free page
    page * get_free_base_page(page * parent);

    // Grab a free page of a certain size
    // If there is no more free page, then send DRAM command based on eviction policy.
    page * allocate_free_page(unsigned size, int appID, unsigned chiplet);

    void set_DRAM_channel(dram_t * dram_channel, int channel_id);
    void set_stat(memory_stats_t * stat);
    page * allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID, unsigned chiplet, unsigned page_mode);
};

class mmu {
public:
    mmu();

    void init(const class memory_config * config);
    void init2(const class memory_config * config);
    void set_ready(); //Called when gpgpu-sim is done initialization
    void set_m_gpu(class gpgpu_sim * gpu);

    page_table * get_page_table_root() { return m_pt_root; }

    new_addr_type get_pa(new_addr_type addr, int appID, bool isRead, unsigned chiplet);
    new_addr_type get_pa(new_addr_type addr, int appID, bool * fault, bool isRead, unsigned chiplet);

    DRAM_layout * get_DRAM_layout() { return m_DRAM; }
    class gpgpu_sim* get_gpu() { return m_gpu; }
    bool get_need_init() const { return need_init; }
    const memory_config * get_mem_config() const { return m_config; }
    void set_stat(memory_stats_t * stat);

    uint64_t get_bitmask(int level);

    void set_DRAM_channel(dram_t * dram_channel, int channel_id);

    void set_L2_tlb(tlb_tag_array * L2TLB);
    tlb_tag_array * get_L2_tlb() { return l2_tlb; }
    bool allocate_PA(new_addr_type va_base, new_addr_type pa_base, int appID, unsigned chiplet, unsigned page_mode);
private:
    class gpgpu_sim * m_gpu;
    Hub* m_gpu_alloc;
    tlb_tag_array * l2_tlb;

    std::map<new_addr_type,new_addr_type> * va_to_pa_mapping;
    std::map<new_addr_type,page*> * va_to_page_mapping;

    bool need_init;
    DRAM_layout * m_DRAM;
    page_table * m_pt_root;
    const memory_config * m_config;
    memory_stats_t * m_stats;
};


#endif //ACCELSIM_MEMORY_OWNER_H
