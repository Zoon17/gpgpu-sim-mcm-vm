// Created by junhyeok on 9/27/23.
// Copyright (c) 2023-2026 Junhyeok Park. All rights reserved.
//
// Built for compatibility with the GPGPU-Sim framework.
// Licensed under the BSD 3-Clause License.
// See the COPYRIGHT file in the project root for full license text.

#ifndef INTER_CHIP_INTERCONNECT_H
#define INTER_CHIP_INTERCONNECT_H

#include "delayqueue.h"
#include <array>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

class memory_config;
class mem_fetch;

enum DIRECTION {
    N = 0,
    E = 1,
    S = 2,
    W = 3,
    L = 4
};

#define NUM_PORT 5
#define FLIT_BYTES 32

struct inter_flit {
    unsigned m_src_chip_id = 0;
    unsigned m_dst_chip_id = 0;
    mem_fetch *m_mf = nullptr;
    bool m_is_refly = false;
    inter_flit() {
        m_src_chip_id = -1;
        m_dst_chip_id = -1;
        m_mf = nullptr;
        m_is_refly = false;
    }
    inter_flit(unsigned src_chip_id, unsigned dst_chip_id, mem_fetch* mf, bool is_reply) {
        m_src_chip_id = src_chip_id;
        m_dst_chip_id = dst_chip_id;
        m_mf = mf;
        m_is_refly = is_reply;
    }
};

struct flit_wrapper {
    uint64_t m_cycle = 0;
    inter_flit * m_flit = nullptr;
    flit_wrapper() {
        m_cycle = 0;
        m_flit = nullptr;
    }
    flit_wrapper(uint64_t cycle, inter_flit * flit) {
        m_cycle = cycle;
        m_flit = flit;
    }
};

class inter_link {
public:
    class gpgpu_sim * m_gpu;
    const memory_config * m_config;

    struct inflight {
        uint64_t m_cycle = 0;
        inter_flit *m_flit = nullptr;
        inflight() {
            m_cycle = 0;
            m_flit = nullptr;
        }
        inflight(uint64_t cycle, inter_flit *flit) {
            m_cycle = cycle;
            m_flit = flit;
        }
    };

    inter_link(const memory_config* config, class gpgpu_sim * gpu);
    bool is_available() const;
    bool push(inter_flit *flit);

    void cycle();

    bool is_received() const;
    inter_flit* get_received() const;
    void pop_received();

private:
    uint64_t m_latency = 1;
    double m_fpc = 1;
    double m_token = 0;

    fifo_pipeline<inter_flit>* m_tx_queue;
    fifo_pipeline<inter_flit>* m_received_queue;
    std::deque<inflight> m_inflight_queue;
};

class inter_noc {
public:
    class gpgpu_sim * m_gpu;
    const memory_config * m_config;

    inter_noc(const memory_config* config, class gpgpu_sim * gpu);

    unsigned mesh_w() const;
    unsigned mesh_h() const;
    unsigned get_num_chip() const;

    void push_flit(inter_flit* flit);
    flit_wrapper* pop_received(unsigned dst);

    void cycle();

private:
    struct Coord {
        unsigned x = 0;
        unsigned y = 0;
        Coord() {
            x = -1;
            y = -1;
        }
        Coord(unsigned x, unsigned y):
            x(x), y(y) { }
    };

    struct edge {
        inter_link *m_link = nullptr;
        unsigned m_chip_id_from = 0;
        unsigned m_chip_id_to = 0;
        enum DIRECTION m_port_from = DIRECTION::L;
        enum DIRECTION m_port_to = DIRECTION::L;
        edge() {
            m_link = nullptr;
            m_chip_id_from = -1;
            m_chip_id_to = -1;
            m_port_from = DIRECTION::L;
            m_port_to = DIRECTION::L;
        }
        edge(inter_link *link, unsigned chip_id_from, unsigned chip_id_to,
            enum DIRECTION port_from, enum DIRECTION port_to):
            m_link(link), m_chip_id_from(chip_id_from), m_chip_id_to(chip_id_to),
            m_port_from(port_from), m_port_to(port_to) { }
    };

    static bool check_num_chip(unsigned total_chip);
    static std::pair<unsigned, unsigned> select_mesh_dims(unsigned total_chip, unsigned w, unsigned h);

    void add_link(unsigned chip_id_from, unsigned chip_id_to, enum DIRECTION port_from, enum DIRECTION port_to);
    void build_mesh(unsigned w, unsigned h);
    enum DIRECTION route(unsigned src, unsigned dst) const;
    void forward(unsigned chip_id);

private:
    unsigned m_mesh_w = 0;
    unsigned m_mesh_h = 0;

    //std::vector<fifo_pipeline<inter_flit>*> m_gateway;
    std::vector<std::deque<inter_flit*>*> m_gateway;
    std::vector<fifo_pipeline<inter_flit>*> m_push_queue;
    std::vector<fifo_pipeline<flit_wrapper>*> m_receive_queue;

    std::vector<std::array<inter_link*, NUM_PORT>> m_link;
    std::vector<Coord> m_coord;
    std::vector<edge> m_edge;
};

#endif //INTER_CHIP_INTERCONNECT_H
