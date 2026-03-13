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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include <bitset>
#include "../abstract_hardware_model.h"
#include "addrdec.h"

class PageWalker;
class dram_t;
class tlb_tag_array;
class memory_config;
class data_cache;
class memory_partition_unit;

enum cache_gpu_level {
  L1_GPU_CACHE = 0,
  L2_GPU_CACHE,
  OTHER_GPU_CACHE,
  NUM_CACHE_GPU_LEVELS
};

enum cache_event_type {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT
};

struct evicted_block_info {
    new_addr_type m_block_addr;
    unsigned m_modified_size;
    mem_access_byte_mask_t m_byte_mask;
    mem_access_sector_mask_t m_sector_mask;
    evicted_block_info() {
        m_block_addr = 0;
        m_modified_size = 0;
        m_byte_mask.reset();
        m_sector_mask.reset();
    }
    void set_info(new_addr_type block_addr, unsigned modified_size,
                  mem_access_byte_mask_t byte_mask,
                  mem_access_sector_mask_t sector_mask) {
        m_block_addr = block_addr;
        m_modified_size = modified_size;
        m_byte_mask = byte_mask;
        m_sector_mask = sector_mask;
    }
};

struct cache_event {
    enum cache_event_type m_cache_event_type;
    evicted_block_info m_evicted_block;  // if it was write_back event, fill the
    // the evicted block info

    cache_event(enum cache_event_type m_cache_event) {
        m_cache_event_type = m_cache_event;
    }

    cache_event(enum cache_event_type cache_event,
                evicted_block_info evicted_block) {
        m_cache_event_type = cache_event;
        m_evicted_block = evicted_block;
    }
};

enum mf_type {
  READ_REQUEST = 0,
  WRITE_REQUEST,
  READ_REPLY,  // send to shader
  WRITE_ACK
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

class memory_config;

class mem_fetch {
 public:
  void set_tlb_time(new_addr_type cycle){
        if (m_tlb_time == static_cast<new_addr_type>(-1)){
          m_tlb_time = cycle;
        }
  }
  new_addr_type get_tlb_time() const { return m_tlb_time; }

  // Page Walk Chain memory fetch constructor
  mem_fetch(mem_fetch * parent);
  mem_fetch(const mem_access_t &access, const warp_inst_t *inst, unsigned long long streamID,
            unsigned ctrl_size, unsigned wid, unsigned sid, unsigned tpc,
            const memory_config *config, unsigned long long cycle,
            mem_fetch *original_mf = nullptr, mem_fetch *original_wr_mf = nullptr);
  ~mem_fetch();

  void set_status(enum mem_fetch_status status, unsigned long long cycle);
  void set_reply() {
    assert(m_access.get_type() != L1_WRBK_ACC &&
           m_access.get_type() != L2_WRBK_ACC);
    if (m_type == READ_REQUEST) {
      assert(!get_is_write());
      m_type = READ_REPLY;
    } else if (m_type == WRITE_REQUEST) {
      assert(get_is_write());
      m_type = WRITE_ACK;
    }
  }
  void do_atomic();

  void print(FILE *fp, bool print_inst = true) const;

  const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
  void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }
  void set_partition(unsigned sub_partition_id) {
    m_raw_addr.sub_partition = sub_partition_id;
  }
  unsigned get_data_size() const { return m_data_size; }
  void set_data_size(unsigned size) { m_data_size = size; }
  unsigned get_ctrl_size() const { return m_ctrl_size; }
  unsigned size() const { return m_data_size + m_ctrl_size; }
  bool is_write() const;
  void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
  new_addr_type get_addr() const { return m_access.get_addr(); }
  unsigned get_access_size() const { return m_access.get_size(); }
  new_addr_type get_partition_addr() const { return m_partition_addr; }  // maybe not used function
  unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
  unsigned get_request_sub_partition_id() const { return m_raw_addr.request_sub_partition; }
  bool get_is_write() const;
  unsigned get_request_uid() const { return m_request_uid; }
  unsigned get_sid() const { return m_sid; }
  unsigned get_tpc() const { return m_tpc; }
  unsigned get_wid() const { return m_wid; }
  bool istexture() const;
  bool isconst() const;
  enum mf_type get_type() const { return m_type; }
  bool isatomic() const;

  unsigned get_malloc_num() const { return m_malloc_num; }
  void set_malloc_num(unsigned num) { m_malloc_num = num; }

  // MCM gpu support
  unsigned compute_origin_chiplet() const;
  unsigned get_origin_chiplet() const { return m_origin_chiplet; }
  unsigned get_map_chiplet() const { return m_map_chiplet; }
  unsigned get_pw_origin_chiplet() const { return m_pw_origin_chiplet; }

  void set_tlb_chiplet(unsigned chiplet) { m_tlb_chiplet = chiplet; }
  unsigned get_tlb_chiplet() const { return m_tlb_chiplet; }

  void handle_write_request(unsigned sid);

  unsigned get_mcm_req_status() const { return m_mcm_req_status; }
  void set_remote_access() {
    assert(m_mcm_req_status == 0);
    m_mcm_req_status = 1; }
  void set_reply_access() {
    assert(m_mcm_req_status == 1);
    m_mcm_req_status = 2; }

  void set_is_local_access() { m_is_local_mf = true; }
  bool get_is_local_access() const { return m_is_local_mf; }

  new_addr_type get_key(new_addr_type addr) const;
  void set_original_addr(new_addr_type addr) { m_original_addr = addr; }
  new_addr_type get_original_addr() const { return m_original_addr; }
  void set_page_fault(bool val);
  bool get_page_fault() const { return m_page_fault; }
  void set_tlb_miss(bool val) { m_tlb_miss = val; }
  bool get_tlb_miss() const { return m_tlb_miss; }

  unsigned get_appID() const { return m_appID; }
  mem_access_t get_access() { return m_access; }

  // page table walk
  mem_fetch * get_parent_tlb_request() { return m_parent_tlb_request; }
  void set_parent_tlb_request(mem_fetch * mf) { m_parent_tlb_request = mf; }

  mem_fetch * get_child_tlb_request() { return m_child; }
  void set_child_tlb_request(mem_fetch * mf) { m_child = mf; }

  unsigned get_tlb_depth_count() const { return m_tlb_depth_count; }
  void propagate_walker(PageWalker * pw);

  void set_tlb(tlb_tag_array * tlb) { m_tlb = tlb;}
  tlb_tag_array * get_tlb() { return m_tlb; }

  void set_cache(data_cache * cache);
  data_cache * get_cache();
  int get_core() const;
  std::list<cache_event> & get_events() { return m_events; }
  void set_events(std::list<cache_event> &ev) { m_events = ev; }

  void set_bank_id(unsigned bank_id) { m_bank_id = bank_id; }
  unsigned get_bank_id() const { return m_bank_id; }

  void set_tlb_ready_cycle(unsigned latency);
  unsigned get_tlb_ready_cycle() const { return m_tlb_ready_cycle; }

  new_addr_type get_tlb_base_key_cache() const {
      return m_base_key;
  }

  void set_block_addr(new_addr_type block_addr) { m_block_addr = block_addr; }
  new_addr_type get_block_addr() const {
      assert(m_block_addr != 0);
      return m_block_addr; }

  void set_return_timestamp(unsigned long long t) { m_timestamp2 = t; }
  void set_timestamp(unsigned long long timestamp) { m_timestamp = timestamp; }
  void set_pw_timestamp(unsigned long long t) { m_pw_timestamp = t; }
  unsigned long long get_pw_timestamp() const { return m_pw_timestamp; }
  unsigned long long get_timestamp() const { return m_timestamp; }
  unsigned long long get_return_timestamp() const { return m_timestamp2; }
  unsigned long long get_streamID() const { return m_streamID; }

  enum mem_access_type get_access_type() const { return m_access.get_type(); }
  const active_mask_t &get_access_warp_mask() const {
    return m_access.get_warp_mask();
  }
  mem_access_byte_mask_t get_access_byte_mask() const {
    return m_access.get_byte_mask();
  }
  mem_access_sector_mask_t get_access_sector_mask() const {
    return m_access.get_sector_mask();
  }

  address_type get_pc() const { return m_inst.empty() ? -1 : m_inst.pc; }
  const warp_inst_t &get_inst() { return m_inst; }
  enum mem_fetch_status get_status() const { return m_status; }

  const memory_config *get_mem_config() { return m_mem_config; }

  unsigned get_num_flits(bool simt_to_mem);

  mem_fetch *get_original_mf() { return original_mf; }
  mem_fetch *get_original_wr_mf() { return original_wr_mf; }

  class gpgpu_sim * get_gpu() {
      assert(m_gpu != nullptr);
      return m_gpu;
  }

  bool get_beenThroughL1() const { return beenThroughL1; }
  void set_beenThroughL1(bool set) { beenThroughL1 = set; }
  bool get_tlb_related_req() const { return m_tlb_related_req; }
  bool get_been_through_tlb() const { return been_through_tlb; }
  void set_been_through_tlb(bool set) { been_through_tlb = set; }

  bool get_pwcache_hit() const { return pwcache_hit; }
  void set_pwcache_hit(bool set) { pwcache_hit = set; }
  bool get_pwcache_done() const { return pwcache_done; }
  void set_pwcache_done(bool set) { pwcache_done = set; }

  PageWalker* get_page_walker() { return page_walker; }
  void set_page_walker(PageWalker *pw) { page_walker = pw; }

  new_addr_type get_time() const;
  void set_page_size(unsigned page_size) { m_page_size = page_size; }
  unsigned get_page_size() const { return m_page_size; }

  void set_page_addr();
  new_addr_type get_page_addr() const { return m_page_addr; };

  unsigned get_pt_walk_skip() const { return m_pt_walk_skip; }

  unsigned get_uid() const {
      return m_inst.get_uid();
  }

  bool m_handle_fault = false;
  void trigger_fault();

  new_addr_type m_walk_dequeue = 0;
  void set_walk_dequeue(new_addr_type cycle){
      m_walk_dequeue = cycle;
  }

  bool check_mem_access_type(enum mem_access_type type) const;

 private:
  unsigned m_page_size = 0;
  unsigned m_mcm_req_status = 0;
  /*
   * 0 : first access  - local access
   * 1 : remote access - handle in home node partition
   * 2 : reply access  - reply to requested node
   */

  unsigned m_malloc_num = -1;
  unsigned m_origin_chiplet = -1; // request generated chiplet
  unsigned m_map_chiplet = -1; // physical page mapped chiplet
  unsigned m_pw_origin_chiplet = -1; // request generated chiplet of page walk requested mf, not tlb chiplet
  unsigned m_tlb_chiplet = -1;

  bool m_is_local_mf = false;

  unsigned m_request_uid;
  unsigned m_sid;
  unsigned m_tpc;
  unsigned m_wid;

  class gpgpu_sim * m_gpu;
  std::list<cache_event> m_events;
  uint32_t m_appID;
  unsigned m_tlb_depth_count;
  bool m_page_fault;
  bool m_tlb_miss;

  bool beenThroughL1 = false; // Used to identify L2 requests in gpu_-cache.cc and gpu-cache.h
  bool m_tlb_related_req;
  bool been_through_tlb;

  bool pwcache_hit;
  bool pwcache_done; // For PW cache hit request, is this done through the latency queue

  PageWalker *page_walker;

  new_addr_type m_original_addr;
  new_addr_type m_page_addr;  // vpn + page_offset, total 48-bit
  unsigned m_pt_walk_skip;
  mem_fetch * m_parent_tlb_request;
  mem_fetch * m_child;
  tlb_tag_array * m_tlb;
  data_cache * m_cache = nullptr;
  int m_core_id;

  unsigned m_bank_id         = -1;
  unsigned m_tlb_ready_cycle = 0;

  new_addr_type m_base_key   = static_cast<new_addr_type>(-1);  // set 4KB base key
  new_addr_type m_block_addr = 0;

  // where is this request now?
  enum mem_fetch_status m_status;
  unsigned long long m_status_change;

  // request type, address, size, mask
  mem_access_t m_access;
  unsigned m_data_size;  // how much data is being written
  unsigned
      m_ctrl_size;  // how big would all this metadata be in hardware (does not
                    // necessarily match actual size of mem_fetch)
  new_addr_type
      m_partition_addr;  // linear physical address *within* dram partition
                         // (partition bank select bits squeezed out)
  addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
                         // chip-row-bank-column address)
  enum mf_type m_type;

  // statistics
  unsigned long long
      m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
  unsigned long long m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
                          // onto icnt to shader; only used for reads
  unsigned long long m_pw_timestamp = 0;
  new_addr_type m_tlb_time = static_cast<new_addr_type>(-1);
  // requesting instruction (put last so mem_fetch prints nicer in gdb)
  warp_inst_t m_inst;

  unsigned long long m_streamID;

  static unsigned sm_next_mf_request_uid;

  const memory_config *m_mem_config;
  unsigned icnt_flit_size;

  mem_fetch
      *original_mf;  // this pointer is set up when a request is divided into
                     // sector requests at L2 cache (if the req size > L2 sector
                     // size), so the pointer refers to the original request
  mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
                              // when fetch-on-write policy is used
};

#endif
