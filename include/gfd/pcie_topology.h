#pragma once

#include "gfd/log.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <unistd.h>
#endif

namespace gfd {

struct GpuTopology {
    int gpu_id;
    int numa_node;
    int pcie_bus;
    int cpu_start;
    int cpu_end;
    int ht_offset;
    int num_physical_cores;
};

struct TopologyConfig {
    int num_gpus;
    int total_numa_nodes;
    int cpus_per_numa;
    int physical_cores_per_numa;
    std::vector<GpuTopology> gpus;

    std::vector<int> gpus_per_numa;

    int recommended_ce_channels(int active_gpus_on_same_numa) const {
        if (active_gpus_on_same_numa <= 2) return 3;
        if (active_gpus_on_same_numa <= 4) return 2;
        return 1;
    }

    void get_exclusive_cores(int gpu_id, int& out_base_cpu, int& out_num_cores,
                             int& out_stride) const {
        auto& g = gpus[gpu_id];
        int numa = g.numa_node;
        int ngpus_on_numa = gpus_per_numa[numa];
        int cores_per_gpu = physical_cores_per_numa / ngpus_on_numa;

        int intra_numa_idx = 0;
        for (int i = 0; i < gpu_id; i++) {
            if (gpus[i].numa_node == numa) intra_numa_idx++;
        }

        out_stride = 1;
        out_base_cpu = g.cpu_start + intra_numa_idx * cores_per_gpu;
        out_num_cores = cores_per_gpu;

        if (out_base_cpu + out_num_cores - 1 > g.cpu_end) {
            out_num_cores = g.cpu_end - out_base_cpu + 1;
        }
    }
};

static inline TopologyConfig discover_topology(int num_gpus) {
    TopologyConfig topo;
    topo.num_gpus = num_gpus;
    topo.gpus.resize(num_gpus);

#ifdef __linux__
    int max_numa = 0;
    FILE* f = fopen("/sys/devices/system/node/online", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f)) {
            char* dash = strchr(buf, '-');
            if (dash) max_numa = atoi(dash + 1);
        }
        fclose(f);
    }
    topo.total_numa_nodes = max_numa + 1;

    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node0/cpulist");
    f = fopen(path, "r");
    int node0_start = 0, node0_end = 63, ht_start = 128;
    if (f) {
        char buf[256];
        if (fgets(buf, sizeof(buf), f)) {
            sscanf(buf, "%d-%d,%d-", &node0_start, &node0_end, &ht_start);
        }
        fclose(f);
    }
    int cpus_per_range = node0_end - node0_start + 1;
    topo.cpus_per_numa = cpus_per_range * 2;
    topo.physical_cores_per_numa = cpus_per_range;

    for (int g = 0; g < num_gpus; g++) {
        topo.gpus[g].gpu_id = g;

        // Query actual NUMA node from PCIe topology via sysfs
        int gpu_numa = -1;
        // Try reading from CUDA device's PCI bus address
        char pci_bus_id[32] = {};
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), g) == cudaSuccess) {
            // pci_bus_id format: "0000:XX:YY.Z" - convert to sysfs path
            // Lowercase the hex for sysfs lookup
            for (char* p = pci_bus_id; *p; p++) {
                if (*p >= 'A' && *p <= 'F') *p = *p - 'A' + 'a';
            }
            snprintf(path, sizeof(path),
                     "/sys/bus/pci/devices/%s/numa_node", pci_bus_id);
            FILE* nf = fopen(path, "r");
            if (nf) {
                char nbuf[16];
                if (fgets(nbuf, sizeof(nbuf), nf)) {
                    gpu_numa = atoi(nbuf);
                }
                fclose(nf);
            }
        }
        // Fallback: naive heuristic if sysfs query fails
        if (gpu_numa < 0 || gpu_numa >= topo.total_numa_nodes) {
            gpu_numa = (num_gpus > 1 && g >= num_gpus / 2) ? 1 : 0;
        }
        topo.gpus[g].numa_node = gpu_numa;

        int numa = topo.gpus[g].numa_node;
        topo.gpus[g].cpu_start = numa * cpus_per_range;
        topo.gpus[g].cpu_end = (numa + 1) * cpus_per_range - 1;
        topo.gpus[g].ht_offset = ht_start - node0_start;
        topo.gpus[g].num_physical_cores = cpus_per_range;
        topo.gpus[g].pcie_bus = g;
    }
#else
    topo.total_numa_nodes = 1;
    topo.cpus_per_numa = 16;
    topo.physical_cores_per_numa = 8;
    for (int g = 0; g < num_gpus; g++) {
        topo.gpus[g].gpu_id = g;
        topo.gpus[g].numa_node = 0;
        topo.gpus[g].cpu_start = 0;
        topo.gpus[g].cpu_end = 15;
        topo.gpus[g].ht_offset = 0;
        topo.gpus[g].num_physical_cores = 8;
        topo.gpus[g].pcie_bus = g;
    }
#endif

    topo.gpus_per_numa.resize(topo.total_numa_nodes, 0);
    for (int g = 0; g < num_gpus; g++) {
        topo.gpus_per_numa[topo.gpus[g].numa_node]++;
    }

    return topo;
}

static inline void print_topology(const TopologyConfig& topo) {
    GFD_LOG_INFO("Topology: %d GPUs, %d NUMA nodes, %d phys cores/node\n",
                 topo.num_gpus, topo.total_numa_nodes, topo.physical_cores_per_numa);
    for (int g = 0; g < topo.num_gpus; g++) {
        auto& gpu = topo.gpus[g];
        int base, ncores, stride;
        topo.get_exclusive_cores(g, base, ncores, stride);
        GFD_LOG_INFO("  GPU %d: NUMA %d, cores %d-%d (stride %d, %d cores)\n",
                     g, gpu.numa_node, base, base + (ncores - 1) * stride, stride, ncores);
    }
    for (int n = 0; n < topo.total_numa_nodes; n++) {
        int ngpu = topo.gpus_per_numa[n];
        int ce_rec = topo.recommended_ce_channels(ngpu);
        GFD_LOG_INFO("  NUMA %d: %d GPUs, recommended %d CE channels/GPU\n",
                     n, ngpu, ce_rec);
    }
}

}  // namespace gfd
