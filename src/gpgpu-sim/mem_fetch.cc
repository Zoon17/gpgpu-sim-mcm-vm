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

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "pagewalk.h"
#include "../cuda-sim/cuda-sim.h"
#define APP_ID 1

unsigned mem_fetch::sm_next_mf_request_uid = 1;

// Page table walk constructor
mem_fetch::mem_fetch(mem_fetch *parent)
    : m_access(parent->get_access()){
    m_streamID = parent->get_streamID();

    pwcache_hit  = false;
    pwcache_done = false;

    m_request_uid = sm_next_mf_request_uid++;

    m_access = parent->get_access();
    if (parent->get_pt_walk_skip() != 0 && parent->get_tlb_depth_count() == 0) {
        m_tlb_depth_count = parent->get_pt_walk_skip() + parent->get_tlb_depth_count() + 1;
    } else {
        m_tlb_depth_count = parent->get_tlb_depth_count() + 1;
    }
    page_walker = parent->page_walker;
    m_tlb = parent->get_tlb();
    new_addr_type new_addr = (new_addr_type)m_tlb->get_tlbreq_addr(parent);
    set_addr(new_addr);
    m_inst = parent->get_inst();
    assert(parent->get_wid() == m_inst.warp_id());
    m_data_size = parent->get_data_size();
    m_ctrl_size = parent->get_ctrl_size();
    m_sid = parent->get_sid();
    m_tpc = parent->get_tpc();
    m_wid = parent->get_wid();

    m_origin_chiplet = parent->get_tlb_chiplet();  // set tlb chiplet to origin chiplet
    m_tlb_chiplet    = parent->get_tlb_chiplet();
    // request generated chiplet of page walk requested mf, not tlb chiplet
    m_pw_origin_chiplet = parent->m_pw_origin_chiplet;

    m_appID = parent->get_appID();
    assert(m_appID == APP_ID);
    m_tlb_related_req = true;
    m_tlb_miss        = false;
    m_original_addr   = parent->get_original_addr();
    m_page_addr       = parent->get_page_addr();
    m_page_size       = parent->get_page_size();
    m_pt_walk_skip    = parent->get_pt_walk_skip();
    m_mem_config      = parent->get_mem_config();

    this->set_page_fault(m_mem_config->m_address_mapping.addrdec_tlx(m_access.get_addr(),&m_raw_addr, PT_SPACE, m_tlb_depth_count, !is_write(), m_origin_chiplet));
    m_partition_addr =
      m_mem_config->m_address_mapping.partition_address(m_access.get_addr(), PT_SPACE, m_tlb_depth_count,!is_write(), m_origin_chiplet);

    m_map_chiplet = m_raw_addr.map_chiplet;
    if (m_tlb_chiplet == m_map_chiplet){  // set page walk local access
        this->set_is_local_access();
    }

    m_type = READ_REQUEST;
    m_gpu = parent->get_gpu();
    assert(m_gpu != nullptr);

    m_timestamp = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    m_timestamp2 = 0;
    m_status  = MEM_FETCH_INITIALIZED;
    m_status_change = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    icnt_flit_size = m_mem_config->icnt_flit_size;
    beenThroughL1 = false;
    been_through_tlb = false;

    if (parent->get_cache()==nullptr) {
        assert(0 && "should not reach here!");
    } else {
        m_cache = parent->get_cache();
        m_core_id = m_cache->get_core_id();
    }

    m_parent_tlb_request = parent;
    m_child = nullptr;
    m_page_fault = false;
    page_walker = nullptr;

    m_handle_fault = true;

    // init unused member variables
    original_mf         = nullptr;
    original_wr_mf      = nullptr;
}

// Original mem_fetch constructor
mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned long long streamID,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc, const memory_config *config,
                     unsigned long long cycle, mem_fetch *m_original_mf,
                     mem_fetch *m_original_wr_mf)
    : m_access(access)
{
  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_streamID  = streamID;
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  m_tlb = nullptr;
  m_core_id = -1;

  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
  if (m_original_mf) {
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
  }

  if (m_sid != static_cast<unsigned>(-1)) m_origin_chiplet = compute_origin_chiplet();  // handle write back request

  m_gpu = nullptr;
  m_original_addr = m_access.get_addr();
  m_page_size = 0;
  m_page_addr = 0;
  m_pt_walk_skip = 0;

  m_tlb_related_req = false;
  m_cache = nullptr;

  m_appID = APP_ID;

  // handle write request
  if (m_access.get_type() != INST_ACC_R && ((!m_mem_config->enable_walk_fault) || check_mem_access_type(m_access.get_type()))) {
    if (m_sid != static_cast<unsigned>(-1)) m_page_fault = (config->m_address_mapping.addrdec_tlx(access.get_addr(),&m_raw_addr, m_appID, 0,!is_write(), m_origin_chiplet));
    if (m_sid != static_cast<unsigned>(-1)) m_partition_addr = config->m_address_mapping.partition_address(access.get_addr(), m_appID, 0, !is_write(), m_origin_chiplet);

    if (m_sid != static_cast<unsigned>(-1)) m_map_chiplet = m_raw_addr.map_chiplet;
    if (m_origin_chiplet == m_raw_addr.map_chiplet && m_origin_chiplet != static_cast<unsigned>(-1)){
          this->set_is_local_access();
    }

    m_handle_fault = true;
  }

  m_pw_origin_chiplet = m_origin_chiplet;

  m_tlb_depth_count = 0;
  m_tlb_miss = false;
  m_page_fault = false;
  m_parent_tlb_request = nullptr;

  pwcache_hit = false;
  pwcache_done = false;
  beenThroughL1 = false;
  been_through_tlb = false;
  page_walker= nullptr;

  // init unused member variables
  m_child = nullptr;
}

mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }

#define MF_TUP_BEGIN(X) static const char *Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

new_addr_type mem_fetch::get_time() const {
  return (new_addr_type)(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
}

void mem_fetch::handle_write_request(unsigned sid) {
  unsigned base_chiplet_cluster = static_cast<unsigned>(m_mem_config->total_sm_num) / static_cast<unsigned>(m_mem_config->chiplet_num);
  unsigned chiplet_num = (unsigned)sid / base_chiplet_cluster;
  assert(chiplet_num < m_mem_config->chiplet_num);

  m_origin_chiplet = chiplet_num;

  m_page_fault = (m_mem_config->m_address_mapping.addrdec_tlx(m_access.get_addr(),&m_raw_addr, m_appID, 0,!is_write(), m_origin_chiplet));
  m_partition_addr = m_mem_config->m_address_mapping.partition_address(m_access.get_addr(), m_appID, 0, !is_write(), m_origin_chiplet);

  m_map_chiplet = m_raw_addr.map_chiplet;
}

unsigned mem_fetch::compute_origin_chiplet() const {
  unsigned base_chiplet_cluster = static_cast<unsigned>(m_mem_config->total_sm_num) / static_cast<unsigned>(m_mem_config->chiplet_num);
  unsigned chiplet_num = static_cast<unsigned>(m_sid) / base_chiplet_cluster;
  assert(chiplet_num < m_mem_config->chiplet_num);
  return chiplet_num;
}

void mem_fetch::set_page_fault(bool val) {
    m_page_fault = val;
}

void mem_fetch::set_tlb_ready_cycle(unsigned int latency) {
    m_tlb_ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + latency;
}

bool mem_fetch::is_write() const {
    if (m_tlb_depth_count > 0){
        return false;
    } else {
        return m_access.is_write();
    }
}

bool mem_fetch::get_is_write() const {
    if (m_tlb_depth_count > 0){
        return false;
    } else {
        return m_access.is_write();
    }
}

void mem_fetch::set_cache(data_cache *cache) {
    m_cache = cache;
    m_core_id = m_cache->get_core_id();
    if (m_gpu == nullptr){
        m_gpu = cache->get_m_gpu();
    }
}

data_cache * mem_fetch::get_cache() { return m_cache; }

int mem_fetch::get_core() const {
    return m_core_id;
}

void mem_fetch::print(FILE *fp, bool print_inst) const {
  fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid,
          m_wid, m_raw_addr.chip);
  m_access.print(fp);
  if ((unsigned)m_status < NUM_MEM_REQ_STAT)
    fprintf(fp, " status = %s (%llu), ", Status_str[m_status], m_status_change);
  else
    fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
  if (!m_inst.empty() && print_inst)
    m_inst.print(fp);
  else
    fprintf(fp, "\n");
}

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}

new_addr_type mem_fetch::get_key(new_addr_type addr) const {
    new_addr_type key = addr / (*(m_mem_config->page_sizes))[m_mem_config->page_sizes->size() - 1];
    assert((key & (static_cast<uint64_t>(3) << 62)) == 0);
    return key;
}

void mem_fetch::propagate_walker(PageWalker *pw) {
    mem_fetch * parent = this->get_parent_tlb_request();
    if (parent != nullptr) {
        parent->page_walker = pw;
        return parent->propagate_walker(pw);
    } else {
        return;
    }
}

void mem_fetch::trigger_fault() {
    if (m_handle_fault) return;  // does not handle twice

    if (m_sid != static_cast<unsigned>(-1)) m_page_fault = (m_mem_config->m_address_mapping.addrdec_tlx(m_access.get_addr(),&m_raw_addr, m_appID, 0,!is_write(), m_origin_chiplet));
    if (m_sid != static_cast<unsigned>(-1)) m_partition_addr = m_mem_config->m_address_mapping.partition_address(m_access.get_addr(), m_appID, 0, !is_write(), m_origin_chiplet);
    if (m_sid != static_cast<unsigned>(-1)) m_map_chiplet = m_raw_addr.map_chiplet;

    if (m_origin_chiplet == m_raw_addr.map_chiplet && m_origin_chiplet != static_cast<unsigned>(-1)){
        this->set_is_local_access();
    }
    m_handle_fault = true;
}

void mem_fetch::set_page_addr() {
    assert(m_page_size != 0);
    new_addr_type page_bit_mask = (static_cast<new_addr_type>(4096) << static_cast<new_addr_type>(std::log2(static_cast<double>(m_page_size)))) - 1;
    m_page_addr = m_original_addr & ~page_bit_mask;

    // Need to update this part; enable to handle page size over 512MB...
    if (m_page_size < PAGE_2M) m_pt_walk_skip = 0;
    else m_pt_walk_skip = 1;
}


bool mem_fetch::check_mem_access_type(enum mem_access_type type) const {
    if (type == L1_WRBK_ACC || type == L2_WRBK_ACC ||
        type == L1_WR_ALLOC_R || type == L2_WR_ALLOC_R ) {
        return true;
    } else {
        if (this->isatomic()) return true;  // atomic error handling
    }
    return false;
}
