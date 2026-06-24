/**
 * benches/throughput_bench.cu — Microkernel Performance Benchmarks
 *
 * Build: nvcc -arch=sm_80 -rdc=true -lcudadevrt throughput_bench.cu
 *             -I ../cuda/include -O3 --use_fast_math -o bench
 * Run:   ./bench
 */

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include "../cuda/include/task_queue.cuh"

static constexpr int BENCH_ITERS      = 1000000;
static constexpr int WARMUP_ITERS     = 10000;
static constexpr int LATENCY_SAMPLES  = 10000;
static constexpr int CDP_BENCH_BLOCKS = 108;

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARK 1: Queue Throughput
// ─────────────────────────────────────────────────────────────────────────────

__device__ unsigned long long g_queue_ops = 0;
__device__ unsigned long long g_cas_retries = 0;

__global__ void queue_throughput_bench(TaskQueue* q, int iters) {
    int seg = blockIdx.x % TASK_QUEUE_SEGMENTS;
    Task t  = {TASK_COMPUTE, 0, (uint64_t)threadIdx.x, 0};
    Task out;

    for (int i = 0; i < WARMUP_ITERS / gridDim.x; ++i) {
        queue_try_enqueue(q, seg, &t);
        queue_try_dequeue(q, seg, &out);
    }
    __syncthreads();

    uint64_t start = clock64();
    int ops = 0;

    for (int i = 0; i < iters; ++i) {
        if (queue_try_enqueue(q, seg, &t)) ops++;
        if (queue_try_dequeue(q, seg, &out)) ops++;
    }

    uint64_t elapsed = clock64() - start;

    if (threadIdx.x == 0) {
        atomicAdd(&g_queue_ops, (unsigned long long)ops);
    }
}

void run_queue_throughput_bench(TaskQueue* d_queue) {
    printf("── Benchmark 1: Queue Throughput ───────────────────────────\n");

    cudaMemset(d_queue, 0, sizeof(TaskQueue));
    // BUG FIX: g_queue_ops is a __device__ variable; assigning to it directly
    // from host code is undefined behaviour. Use cudaMemcpyToSymbol instead.
    {
        unsigned long long zero = 0ULL;
        cudaMemcpyToSymbol(g_queue_ops, &zero, sizeof(zero));
    }

    int grid  = CDP_BENCH_BLOCKS;
    int block = 256;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    queue_throughput_bench<<<grid, block>>>(d_queue, BENCH_ITERS / (grid * block));
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    unsigned long long total_ops;
    cudaMemcpyFromSymbol(&total_ops, g_queue_ops, sizeof(unsigned long long));

    double throughput_mops = (double)total_ops / (ms / 1000.0) / 1e6;
    printf("  Total ops:    %llu\n", total_ops);
    printf("  Elapsed:      %.2f ms\n", ms);
    printf("  Throughput:   %.1f Million ops/sec\n\n", throughput_mops);
}

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARK 2: CDP Launch Latency
// ─────────────────────────────────────────────────────────────────────────────

__global__ void cdp_noop_child(uint64_t* start_times, int slot) {
    if (threadIdx.x == 0) {
        start_times[slot] = clock64();
    }
}

__global__ void cdp_launch_latency_bench(uint64_t* launch_times,
                                          uint64_t* start_times,
                                          int samples) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int i = 0; i < samples; ++i) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

        launch_times[i] = clock64();
        cdp_noop_child<<<1, 32, 0, s>>>(start_times, i);

        cudaStreamDestroy(s);
        cudaDeviceSynchronize();
    }
}

void run_cdp_latency_bench() {
    printf("── Benchmark 2: CDP Launch Latency ─────────────────────────\n");

    uint64_t* d_launch_times;
    uint64_t* d_start_times;
    cudaMalloc(&d_launch_times, LATENCY_SAMPLES * sizeof(uint64_t));
    cudaMalloc(&d_start_times,  LATENCY_SAMPLES * sizeof(uint64_t));

    cdp_launch_latency_bench<<<1, 1>>>(d_launch_times, d_start_times, LATENCY_SAMPLES);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_launch(LATENCY_SAMPLES), h_start(LATENCY_SAMPLES);
    cudaMemcpy(h_launch.data(), d_launch_times, LATENCY_SAMPLES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_start.data(),  d_start_times,  LATENCY_SAMPLES * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // BUG FIX: store clock rate as double so division is floating-point,
    // not integer — integer division silently truncates sub-μs latencies to 0.
    int clock_khz_int;
    cudaDeviceGetAttribute(&clock_khz_int, cudaDevAttrClockRate, 0);
    double clock_mhz = (double)clock_khz_int / 1000.0;  // kHz → MHz

    std::vector<double> latencies_us(LATENCY_SAMPLES);
    for (int i = 0; i < LATENCY_SAMPLES; ++i) {
        uint64_t cycles = h_start[i] - h_launch[i];
        latencies_us[i] = (double)cycles / clock_mhz;
    }
    std::sort(latencies_us.begin(), latencies_us.end());

    printf("  Samples:  %d\n", LATENCY_SAMPLES);
    printf("  P50:      %.1f μs\n", latencies_us[LATENCY_SAMPLES * 50 / 100]);
    printf("  P95:      %.1f μs\n", latencies_us[LATENCY_SAMPLES * 95 / 100]);
    printf("  P99:      %.1f μs\n", latencies_us[LATENCY_SAMPLES * 99 / 100]);
    printf("  Max:      %.1f μs\n", latencies_us[LATENCY_SAMPLES - 1]);
    printf("  Min:      %.1f μs\n\n", latencies_us[0]);

    cudaFree(d_launch_times);
    cudaFree(d_start_times);
}

// ─────────────────────────────────────────────────────────────────────────────
// BENCHMARK 3: End-to-End Task Latency
// ─────────────────────────────────────────────────────────────────────────────

__device__ uint64_t g_e2e_enqueue_times[LATENCY_SAMPLES];
__device__ uint64_t g_e2e_child_times[LATENCY_SAMPLES];
__device__ int      g_e2e_sample_idx = 0;

__global__ void e2e_child_kernel(int sample_idx) {
    if (threadIdx.x == 0) {
        g_e2e_child_times[sample_idx] = clock64();
    }
}

__global__ void e2e_latency_bench(TaskQueue* q, int samples) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    q->segments[0].head_packed.store(0, cuda::memory_order_relaxed);
    q->segments[0].tail_packed.store(0, cuda::memory_order_relaxed);
    for (int i = 0; i < SEGMENT_CAPACITY; ++i)
        q->segments[0].slot_state[i].store(SLOT_EMPTY, cuda::memory_order_relaxed);

    for (int i = 0; i < samples; ++i) {
        g_e2e_enqueue_times[i] = clock64();

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        e2e_child_kernel<<<1, 32, 0, s>>>(i);
        cudaStreamDestroy(s);
        cudaDeviceSynchronize();
    }
}

void run_e2e_latency_bench(TaskQueue* d_queue) {
    printf("── Benchmark 3: End-to-End Task Latency ────────────────────\n");

    e2e_latency_bench<<<1, 1>>>(d_queue, LATENCY_SAMPLES);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_enq(LATENCY_SAMPLES), h_child(LATENCY_SAMPLES);
    cudaMemcpyFromSymbol(h_enq.data(),   g_e2e_enqueue_times, LATENCY_SAMPLES * sizeof(uint64_t));
    cudaMemcpyFromSymbol(h_child.data(), g_e2e_child_times,   LATENCY_SAMPLES * sizeof(uint64_t));

    // BUG FIX: same integer-division truncation fix as in run_cdp_latency_bench.
    int clock_khz_int2;
    cudaDeviceGetAttribute(&clock_khz_int2, cudaDevAttrClockRate, 0);
    double clock_mhz_e2e = (double)clock_khz_int2 / 1000.0;

    std::vector<double> e2e_us(LATENCY_SAMPLES);
    for (int i = 0; i < LATENCY_SAMPLES; ++i)
        e2e_us[i] = (double)(h_child[i] - h_enq[i]) / clock_mhz_e2e;
    std::sort(e2e_us.begin(), e2e_us.end());

    printf("  P50:  %.1f μs\n", e2e_us[LATENCY_SAMPLES * 50 / 100]);
    printf("  P95:  %.1f μs\n", e2e_us[LATENCY_SAMPLES * 95 / 100]);
    printf("  P99:  %.1f μs\n\n", e2e_us[LATENCY_SAMPLES - 1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("=== GPU Microkernel Performance Benchmarks ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM %d.%d, %d SMs, %.0f MHz)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.clockRate / 1000.0);

    TaskQueue* d_queue;
    cudaMalloc(&d_queue, sizeof(TaskQueue));

    run_queue_throughput_bench(d_queue);
    run_cdp_latency_bench();
    run_e2e_latency_bench(d_queue);

    cudaFree(d_queue);
    printf("Benchmarks complete.\n");
    return 0;
}
