// Created by junhyeok on 9/27/23.
// Copyright (c) 2023-2026 Junhyeok Park. All rights reserved.
//
// Built for compatibility with the GPGPU-Sim framework.
// Licensed under the BSD 3-Clause License.
// See the COPYRIGHT file in the project root for full license text.

#include "inter_chip_interconnect.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"

inter_link::inter_link(const memory_config* config, class gpgpu_sim *gpu) {
    m_gpu = gpu;
    m_config = config;

    m_latency = config->link_latency;

    //double bandwidth = static_cast<double>(config->link_bandwidth * 1e9);  // GBps to Bps
    //double freqency = static_cast<double>(config->link_frequency * 1e9);  // GHz to Hz
    double bandwidth = static_cast<double>(config->link_bandwidth);  // GBps
    double freqency = static_cast<double>(config->link_frequency);   // GHz

    m_fpc = bandwidth / (freqency * static_cast<double>(FLIT_BYTES));

    m_token = 0;
    m_tx_queue = new fifo_pipeline<inter_flit>("txq", 0, config->link_tx_depth);
    m_received_queue = new fifo_pipeline<inter_flit>("arq", 0, config->link_arrive_depth);
}

bool inter_link::is_available() const {
    return !m_tx_queue->full();
}

bool inter_link::push(inter_flit *flit) {
    m_tx_queue->push(flit);
    return true;
}

void inter_link::cycle() {
    new_addr_type current_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;

    // accumulate bandwidth
    m_token += m_fpc;

    // move from m_tx to inflight
    while (m_token > 1 && !m_tx_queue->empty()) {
        inter_flit *get_flit = m_tx_queue->top();
        m_tx_queue->pop();
        m_token--;
        m_inflight_queue.push_back(inflight(current_cycle + m_latency, get_flit));
    }

    // move arrived flits to arrive
    while (!m_inflight_queue.empty() && m_inflight_queue.front().m_cycle < current_cycle) {
        if (m_received_queue->full()) {
            break;
        }

        inter_flit *get_flit = m_inflight_queue.front().m_flit;
        m_received_queue->push(get_flit);
        m_inflight_queue.pop_front();
    }
}

bool inter_link::is_received() const {
    return !m_received_queue->empty();
}

inter_flit* inter_link::get_received() const {
    return m_received_queue->top();
}

void inter_link::pop_received() {
    m_received_queue->pop();
}

inter_noc::inter_noc(const memory_config* config, class gpgpu_sim *gpu) {
    m_gpu = gpu;
    m_config = config;

    auto [w, h] = select_mesh_dims(config->chiplet_num, 0, 0);
    m_mesh_w = w;
    m_mesh_h = h;
    build_mesh(m_mesh_w, m_mesh_h);
}

unsigned inter_noc::mesh_w() const {
    return m_mesh_w;
}

unsigned inter_noc::mesh_h() const {
    return m_mesh_h;
}

unsigned inter_noc::get_num_chip() const {
    return static_cast<unsigned>(m_gateway.size());
}

void inter_noc::push_flit(inter_flit *flit) {
    assert(flit->m_src_chip_id < m_gateway.size());
    if (!flit->m_is_refly) {
        assert(flit->m_mf->get_mcm_req_status() == 0);
    }
    m_gateway[flit->m_src_chip_id]->push_back(flit);
}

flit_wrapper* inter_noc::pop_received(unsigned dst) {
    assert(dst < m_receive_queue.size());
    if (m_receive_queue[dst]->empty()) return nullptr;

    flit_wrapper* flit_wrap = m_receive_queue[dst]->top();
    m_receive_queue[dst]->pop();
    return flit_wrap;
}

void inter_noc::cycle() {
    new_addr_type current_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;

    // forward one flit
    for (unsigned i = 0; i < m_gateway.size(); ++i) {
        forward(i);
    }

    // link cycle
    for (unsigned i = 0; i < m_edge.size(); ++i) {
        m_edge[i].m_link->cycle();
    }

    // link queue handling
    for (unsigned i = 0; i < m_edge.size(); ++i) {
        while (m_edge[i].m_link->is_received()) {
            inter_flit* get_flit = m_edge[i].m_link->get_received();

            if (get_flit->m_dst_chip_id == m_edge[i].m_chip_id_to) {
                flit_wrapper* new_wrap = new flit_wrapper(current_cycle, get_flit);
                if (!m_receive_queue[m_edge[i].m_chip_id_to]->full()) {
                    m_receive_queue[m_edge[i].m_chip_id_to]->push(new_wrap);
                } else {
                    break;
                }
            } else {
                if (!m_push_queue[m_edge[i].m_chip_id_to]->full()) {
                    m_push_queue[m_edge[i].m_chip_id_to]->push(get_flit);
                } else {
                    break;
                }
            }
            m_edge[i].m_link->pop_received();
        }
    }
}

// check whether total chip cnt is power of two
bool inter_noc::check_num_chip(unsigned total_chip) {
    return total_chip && ((total_chip & (total_chip - 1)) == 0);
}

std::pair<unsigned, unsigned> inter_noc::select_mesh_dims(unsigned total_chip, unsigned w, unsigned h) {
    assert(total_chip > 2);
    assert(check_num_chip(total_chip));

    if (w > 0 || h > 0) {
        assert(w > 0 || h > 0);
        assert((w * h) == total_chip);
        return std::make_pair(w, h);
    }

    double s = std::sqrt(static_cast<double>(total_chip));
    unsigned exp = static_cast<unsigned>(std::floor(std::log2(s)));
    unsigned get_w = 1 << exp;
    unsigned get_h = total_chip / get_w;
    return std::make_pair(get_w, get_h);
}

void inter_noc::add_link(unsigned chip_id_from, unsigned chip_id_to, enum DIRECTION port_from, enum DIRECTION port_to) {
    inter_link* new_link = new inter_link(m_config, m_gpu);
    m_edge.push_back(edge(new_link, chip_id_from, chip_id_to, port_from, port_to));
    m_link[chip_id_from][static_cast<unsigned>(port_from)] = new_link;
}

void inter_noc::build_mesh(unsigned w, unsigned h) {
    unsigned total_chip = w * h;

    m_gateway.resize(total_chip);
    m_push_queue.resize(total_chip);
    m_receive_queue.resize(total_chip);
    m_link.resize(total_chip);
    m_coord.resize(total_chip);

    for (unsigned i = 0; i < total_chip; ++i) {
        //m_gateway[i] = new fifo_pipeline<inter_flit>("gw", 0, m_config->noc_gateway_depth);
        m_gateway[i] = new std::deque<inter_flit*>;
        m_push_queue[i] = new fifo_pipeline<inter_flit>("pq", 0, m_config->noc_push_depth);
        m_receive_queue[i] = new fifo_pipeline<flit_wrapper>("rq", 0, m_config->noc_receive_depth);
        m_link[i].fill(nullptr);
    }

    for (unsigned y = 0; y < h; ++y) {
        for (unsigned x = 0; x < w; ++x) {
            m_coord[y * w + x] = Coord(x, y);
        }
    }

    for (unsigned y = 0; y < h; ++y) {
        for (unsigned x = 0; x < w; ++x) {
            unsigned chip_id = y * w + x;
            if (x + 1 < w) {
                unsigned chip_id_next = y * w + x + 1;
                add_link(chip_id, chip_id_next, DIRECTION::E, DIRECTION::W);
                add_link(chip_id_next, chip_id, DIRECTION::W, DIRECTION::E);
            }

            if (y + 1 < h) {
                unsigned chip_id_next = (y + 1) * w + x;
                add_link(chip_id, chip_id_next, DIRECTION::S, DIRECTION::N);
                add_link(chip_id_next, chip_id, DIRECTION::N, DIRECTION::S);
            }
        }
    }
}

enum DIRECTION inter_noc::route(unsigned src, unsigned dst) const {
    if (src == dst) { return DIRECTION::L; }

    const Coord& c_src = m_coord[src];
    const Coord& c_dst = m_coord[dst];

    if (c_dst.x > c_src.x) { return DIRECTION::E; }
    if (c_dst.x < c_src.x) { return DIRECTION::W; }
    if (c_dst.y > c_src.y) { return DIRECTION::S; }
    if (c_dst.y < c_src.y) { return DIRECTION::N; }

    return DIRECTION::L;
}

void inter_noc::forward(unsigned chip_id) {
    new_addr_type current_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;

    // handle traffic
    while (true) {
        if (!m_push_queue[chip_id]->empty()) {
            inter_flit* get_flit = m_push_queue[chip_id]->top();
            DIRECTION direction = route(chip_id, get_flit->m_dst_chip_id);
            if (direction == DIRECTION::L) {
                flit_wrapper* new_wrapper = new flit_wrapper(current_cycle, get_flit);
                if (!m_receive_queue[chip_id]->full()) {
                    m_receive_queue[chip_id]->push(new_wrapper);
                    m_push_queue[chip_id]->pop();
                } else {
                    break;
                }
            }

            inter_link* get_link = m_link[chip_id][static_cast<unsigned>(direction)];
            if (get_link == nullptr) { break; }
            if (get_link->is_available()) {
                get_link->push(get_flit);
                m_push_queue[chip_id]->pop();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // push new flits
    while (true) {
        if (!m_gateway[chip_id]->empty()) {
            inter_flit* get_flit = m_gateway[chip_id]->front();
            DIRECTION direction = route(chip_id, get_flit->m_dst_chip_id);
            if (direction == DIRECTION::L) {
                flit_wrapper* new_wrapper = new flit_wrapper(current_cycle, get_flit);
                if (!m_receive_queue[chip_id]->full()) {
                    m_receive_queue[chip_id]->push(new_wrapper);
                    m_gateway[chip_id]->pop_front();
                } else {
                    break;
                }
            }

            inter_link* get_link = m_link[chip_id][static_cast<unsigned>(direction)];
            if (get_link == nullptr) { break; }
            if (get_link->is_available()) {
                get_link->push(get_flit);
                m_gateway[chip_id]->pop_front();
            } else {
                break;
            }
        } else {
            break;
        }
    }
}
