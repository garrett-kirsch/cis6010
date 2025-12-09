// TODO: before you submit on Canvas, include here:
//   The following is with gpu_1x_a100_sxm4
//   On the above GPU here are the performances for size = 2048, reps = 1
//      0) cublas Algorithm: 13465.87 GFLOPS
//      1) basic Algorithm: 226.45 GFLOPS
//      2) gmem_coalesced Algorithm: 2150.06 GFLOPS
//      3) sharedmem Algorithm: 4097.31 GFLOPS (F = 32)
//      4) sharedmem_multioutput Algorithm: 9740.39 (F = 64, G = 4)
//      5) smem_multioutput_1stream Algorithm: 2501.87 (F = 64, G = 4)
//      6) smem_multioutput_multistream Algorithm: 3239.22 (F = 64, G = 4, NUM_STREAMS = 8)

//      7) base_tensor_core: 8635.81 GFLOPS
//      8) optimized_tensor_core: 9046.76 GFLOPS

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N)-1) / (N))

enum Algo
{
    cublas = 0,
    basic,
    gmem_coalesced,
    smem,
    smem_multioutput,
    smem_multioutput_1stream,
    smem_multioutput_multistream,
    naive_tensor_core,
    optimized_tensor_core,
    numAlgos
};

const char *algo2str(Algo a)
{
    switch (a)
    {
    case cublas:
        return "cublas";
    case basic:
        return "basic";
    case gmem_coalesced:
        return "gmem_coalesced";
    case smem:
        return "sharedmem";
    case smem_multioutput:
        return "sharedmem_multioutput";
    case smem_multioutput_1stream:
        return "sharedmem_multioutput_1stream";
    case smem_multioutput_multistream:
        return "sharedmem_multioutput_multistream";
    case naive_tensor_core:
        return "naive_tensor_core";
    case optimized_tensor_core:
        return "optimized_tensor_core";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_matrix(float *mat, int N);
void const_init_matrix(float *mat, int N, float F);
bool verify_matrix(float *expected, float *actual, int M, int N);
void print_matrix(const float *A, int M, int N, std::ostream &outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C, uint NUM_STREAMS, float* hA, float* hB, float* hC);
void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

const std::string errLogFile = "gemmValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)", cxxopts::value<uint16_t>()->default_value("128"))                //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))                           //
        ("algo", "GEMM algorithm to use, a number in [0,6], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0")) //
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))                     //
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))                                              //
        ("streams", "number of CUDA streams to use", cxxopts::value<uint>()->default_value("1"))                          //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    if (SIZE % 32 != 0)
    {
        std::cout << "--size must be a multiple of 32" << std::endl;
        exit(EXIT_FAILURE);
    }
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }
    const uint NUM_STREAMS = clFlags["streams"].as<uint>();

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    generator.seed(SEED);
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Setup cublas
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = alpha*AB+beta*C

    // just do pure A*B (for simpler debugging)
    float alpha = 1.0, beta = 1.0, initC = 1.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    cudaMallocHost(&A, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&B, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&C, sizeof(float) * SIZE * SIZE);
    cudaMallocHost(&C_ref, sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);
    // print_matrix(A, SIZE, SIZE, std::cout);
    // print_matrix(B, SIZE, SIZE, std::cout);
    // print_matrix(C, SIZE, SIZE, std::cout);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));

    printf("dimensions(m=n=k) %u, alpha: %f, beta: %f\n", m, alpha, beta);

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE)
    {
        printf("disabled validation\n");
    }
    else
    {
        // run cublas to get correct answer in dC_ref
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC, NUM_STREAMS, A, B, C);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, n, m))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile, std::ios::out | std::ios::trunc);
            fs << "alpha=" << alpha << " beta=" << beta << std::endl;
            fs << "C matrix initialized to " << initC << std::endl << std::endl;
            fs << "A:" << std::endl;
            print_matrix(A, m, n, fs);
            fs << "B:" << std::endl;
            print_matrix(B, m, n, fs);
            fs << "C:" << std::endl;
            print_matrix(C, m, n, fs);
            fs << "Expected:" << std::endl;
            print_matrix(C_ref, m, n, fs);
            fs.close();
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC, NUM_STREAMS, A, B, C);
        cudaCheck(cudaDeviceSynchronize());
    }

    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = (double)2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        m);

    // free CPU and GPU memory
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to random values. */
void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(float *mat, int N, float F)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = F;
    }
}

/** Print the given MxN matrix `mat` to the provided output stream. */
void print_matrix(const float *A, int M, int N, std::ostream &outs)
{
    outs << "[";
    for (int i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
        {
            outs << std::fixed << std::setprecision(3) << A[i];
        }
        else
        {
            outs << std::fixed << std::setprecision(3) << A[i] << ", ";
        }
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                outs << ";" << std::endl;
        }
    }
    outs << "]" << std::endl << std::endl;
}

bool verify_matrix(float *expected, float *actual, int M, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float fexp = (expected[(i * N) + j]);
            float fact = (actual[(i * N) + j]);
            double diff = std::fabs(fexp - fact);
            if (diff > 1.5)
            {
                printf("Divergence! Should be %5.3f, is %5.3f (diff %5.3f) at [%d,%d]\n",
                       fexp, fact, diff, i, j);
                return false;
            }
        }
    }
    return true;
}

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C)
{
    // cuBLAS uses *column-major* order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
    //                                  N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, /*CUBLAS_COMPUTE_16F*/ CUBLAS_COMPUTE_16F_PEDANTIC,
    //                                  CUBLAS_GEMM_DEFAULT);
    cublasStatus_t ok = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cublasCheck(ok);
}

__global__ void runBasic(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        // C = alpha*(AxB)+beta*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(x * K) + i] * B[(i * N) + y];
        }
        // __C__[x][y]
        C[(x * N) + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}

__global__ void runGmemCoalesced(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW1 TODO: copy runBasic() code here and update to avoid uncoalesced accesses to global memory.
    // Note, you are also free to change the grid dimensions in the kernel launch below.

    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // variable per warp
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y; // constant per warp

    if (x < N && y < M)
    {
        float tmp = 0.0;
        // C = alpha*(AxB)+beta*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(y * K) + i] * B[(i * N) + x];
        }
        // __C__[x][y]
        C[(y * N) + x] = (alpha * tmp) + (beta * C[y * N + x]);
    }

}

const uint F = 64;

__global__ void runSharedMem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW2 TODO: Use shared memory to cache square FxF tiles of the A and B matrices in shared memory 
    // (SA and SB, respectively, provided below). Each thread should compute the result for one cell 
    // of the output matrix C.

    // Note, you will also need to change the grid dimensions in the kernel launch below to take into account the value
    // of F (which is a constant, defined above). You should experiment with different values of F to see how it 
    // affects performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // variable per warp
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y; // constant per warp

    float cumulative_result = 0.0;

    // Iterate over however many tiles needed to cover the Kth dimension
    // for C[y][x] we need A[y][k] + B[i][k]
    for (int i = 0; i < K; i += F) {
        int a_col_element = i + threadIdx.x;
        // if not within range, save 0: x + a*0 = x
        if (a_col_element < K && y < M) {
            // save piece of col of A
            SA[threadIdx.y][threadIdx.x] = A[K * y + a_col_element];
        } else {SA[threadIdx.y][threadIdx.x] = 0.0;}

        int b_row_element = i + threadIdx.y;
        if (x < N && b_row_element < K) {
            // save piece of row of b
            SB[threadIdx.y][threadIdx.x] = B[N * b_row_element + x];
        } else {SB[threadIdx.y][threadIdx.x] = 0.0;}

        // Ensure smem is stable
        __syncthreads();

        // accumulate partial sum of products for curr matrix output
        for (int k = 0; k < F; k++) {
            cumulative_result += SA[threadIdx.y][k] * SB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < N && y < M) {
        C[(y * N) + x] = (alpha * cumulative_result) + (beta * C[y * N + x]);
    }

}

const uint G = 4;

__global__ void runSharedMemMultiOutput(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW3 TODO: Copy your runSharedMem() code here and update it so that each thread computes the result for GxG cells 
    // of the output matrix C. Each thread should accumulate temporary results in the local LC matrix, provided below,
    // before writing them to C in global memory.

    // Note, you will also need to change the grid dimensions in the kernel launch below. You should experiment 
    // with different values of F and G to see how they affect performance.

    float LC[G][G] = {0.0};

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x; // variable per warp
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y; // constant per warp

    const unsigned xG = threadIdx.x;
    const unsigned yG = threadIdx.y;

    // Iterate over however many tiles needed to cover the Kth dimension
    // for C[y][x] we need A[y][k] + B[i][k]
    for (int i = 0; i < K; i += F) {
        for (int row_lc = 0; row_lc < G; row_lc++) {
            for (int col_lc = 0; col_lc < G; col_lc++) {
                int a_row_source = y * G + row_lc;
                int a_row_shared = yG * G + row_lc;
                int b_row_source = i + yG * G + row_lc;
                int b_row_shared = yG * G + row_lc;
                int a_col_source = i + xG * G + col_lc;
                int a_col_shared = xG * G + col_lc;
                int b_col_source = x * G + col_lc;
                int b_col_shared = xG * G + col_lc;

                if  (a_row_source < M && a_col_source < K) {
                    SA[a_row_shared][a_col_shared] = A[a_row_source * K + a_col_source];
                }

                if (b_row_source < K && b_col_source < N) {
                    SB[b_row_shared][b_col_shared] = B[b_row_source * N + b_col_source];
                }
            }
        }

        // Ensure smem is stable
        __syncthreads();

        // accumulate partial sum of products for curr matrix output
        for (int k = 0; k < F; k++) {
            for (int row_lc = 0; row_lc < G; row_lc++) {
                for (int col_lc = 0; col_lc < G; col_lc++) {
                    int a_row_shared = yG * G + row_lc;
                    int b_col_shared = xG * G + col_lc;
                    LC[row_lc][col_lc] += SA[a_row_shared][k] * SB[k][b_col_shared];
                }
            }
        }

        __syncthreads();
    }

    for (int row_lc = 0; row_lc < G; row_lc++) {
        for (int col_lc = 0; col_lc < G; col_lc++) {
            int c_col_source = x * G + col_lc;
            int c_row_source = y * G + row_lc;

            C[c_row_source * N + c_col_source] = alpha * LC[row_lc][col_lc] + beta * C[c_row_source * N + c_col_source];
        }
    }

}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void convert_FP32_to_FP16(const float* in, half* out, int numElems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElems) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void naive_tensor_core_kernel(int M, int N, int K, float alpha, const half* A16, const half* B16, float beta, float* C) {
    int tileCol = blockIdx.x;
    int tileRow = blockIdx.y;

    int row = tileRow * WMMA_M;
    int col = tileCol * WMMA_N;

    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>   a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>   b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const half* tileA = A16 + row * K + k; // (row, k)
        const half* tileB = B16 + k * N + col; // (k, col)

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
    }

    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// optimized kernel with tile accumulation and memory coalescing

const int BM = 64;
const int BN = 64;
const int BK = 32;

// optimized kernel with tile accumulation and memory coalescing
__global__ void optimized_tensor_core_kernel(int M, int N, int K, float alpha, 
                                              const half* A16, const half* B16, 
                                              float beta, float* C) {
    // block tile indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // warp and lane ids
    int warpM = (threadIdx.x / 32) / (BN / WMMA_N);
    int warpN = (threadIdx.x / 32) % (BN / WMMA_N);
    
    // shared memory for prefetching tiles
    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];
    
    // fragments for this warp
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // each warp handles one wmma tile
    int warp_row = by * BM + warpM * WMMA_M;
    int warp_col = bx * BN + warpN * WMMA_N;
    
    // initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // loop over k dimension in blocks
    for (int k_block = 0; k_block < K; k_block += BK) {
        // cooperative loading of a and b tiles into shared memory
        // each thread loads multiple elements
        int tid = threadIdx.x;
        int num_threads = blockDim.x;
        
        // load a tile (bm x bk)
        int A_tile_elems = BM * BK;
        for (int i = tid; i < A_tile_elems; i += num_threads) {
            int row = i / BK;
            int col = i % BK;
            int global_row = by * BM + row;
            int global_col = k_block + col;
            As[i] = (global_row < M && global_col < K) ? 
                    A16[global_row * K + global_col] : __float2half(0.0f);
        }
        
        // load b tile (bk x bn)
        int B_tile_elems = BK * BN;
        for (int i = tid; i < B_tile_elems; i += num_threads) {
            int row = i / BN;
            int col = i % BN;
            int global_row = k_block + row;
            int global_col = bx * BN + col;
            Bs[i] = (global_row < K && global_col < N) ? 
                    B16[global_row * N + global_col] : __float2half(0.0f);
        }
        
        __syncthreads();
        
        // compute wmma tiles from shared memory
        for (int k = 0; k < BK; k += WMMA_K) {
            int local_row = warpM * WMMA_M;
            int local_col = warpN * WMMA_N;
            
            wmma::load_matrix_sync(a_frag, &As[local_row * BK + k], BK);
            wmma::load_matrix_sync(b_frag, &Bs[k * BN + local_col], BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        __syncthreads();
    }
    
    // apply alpha and beta, then store result
    if (warp_row < M && warp_col < N) {
        // scale by alpha
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] *= alpha;
        }
        
        // add beta * c if beta != 0
        if (beta != 0.0f) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
            wmma::load_matrix_sync(c_old, C + warp_row * N + warp_col, N, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] += beta * c_old.x[i];
            }
        }
        
        wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, wmma::mem_row_major);
    }
}


void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha,
             float *A, float *B, float beta, float *C,
             uint NUM_STREAMS, float *hA, float* hB, float* hC)
{
    switch (algo)
    {
    case cublas:
        runCublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case basic:
    {
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runBasic<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case gmem_coalesced:
    {
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockDim(32, 32);
        runGmemCoalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
        dim3 blockDim(F,F);
        runSharedMem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem_multioutput:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        assert(0 == F % G);
        assert((F*F) / (G*G) >= F);
        dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
        dim3 blockDim(F / G, F/ G);
        runSharedMemMultiOutput<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem_multioutput_1stream:
    {
      assert(0 == M % F);
      assert(0 == N % F);
      assert(0 == K % F);
      assert(0 == F % G);
      assert((F*F) / (G*G) >= F);
      cudaCheck(cudaMemcpy(A, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(B, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(C, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice));

      dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
      dim3 blockDim(F / G, F/ G);
      runSharedMemMultiOutput<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

      cudaMemcpy(hC, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
      break;
    }
    case smem_multioutput_multistream:
    {
      assert(0 == M % F);
      assert(0 == N % F);
      assert(0 == K % F);
      assert(0 == F % G);
      assert((F*F) / (G*G) >= F);
      assert(0 == (N/F) % NUM_STREAMS);

      dim3 gridDim(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N / NUM_STREAMS, F));
      dim3 blockDim(F / G, F/ G);

      cudaStream_t streams[NUM_STREAMS];
      for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheck(cudaStreamCreate(&streams[i]));
      }

      cudaMemcpy(B, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice);

      for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(A + i * M / NUM_STREAMS * K, hA + i * M / NUM_STREAMS * K, sizeof(float) * K * M / NUM_STREAMS, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(C + i * M / NUM_STREAMS * N, hC + i * M / NUM_STREAMS * N, sizeof(float) * N * M / NUM_STREAMS, cudaMemcpyHostToDevice, streams[i]);

        runSharedMemMultiOutput<<<gridDim, blockDim, 0, streams[i]>>>(M / NUM_STREAMS, N, K, alpha, A + i * M / NUM_STREAMS * K, B, beta, C + i * M / NUM_STREAMS * N);

        cudaMemcpyAsync(hC + i * M / NUM_STREAMS * N, C + i * M / NUM_STREAMS * N, sizeof(float) * N * M / NUM_STREAMS, cudaMemcpyDeviceToHost, streams[i]);
      }
      
      for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaCheck(cudaStreamSynchronize(streams[i]));
            cudaCheck(cudaStreamDestroy(streams[i]));
      }

      break;
    }
    case naive_tensor_core:
    {
        half *A16, *B16;
        cudaCheck(cudaMalloc(&A16, sizeof(half) * M * K));
        cudaCheck(cudaMalloc(&B16, sizeof(half) * K * N));

        // Convert FP32 A,B → FP16 on GPU
        int numElemsA = M * K;
        int numElemsB = K * N;

        int threads = 256;
        int blocksA = ROUND_UP_TO_NEAREST(numElemsA, threads);
        int blocksB = ROUND_UP_TO_NEAREST(numElemsB, threads);

        convert_FP32_to_FP16<<<blocksA, threads>>>(A, A16, numElemsA);
        convert_FP32_to_FP16<<<blocksB, threads>>>(B, B16, numElemsB);

        // Launch tensor core GEMM kernel
        dim3 gridDim( (N + WMMA_N - 1) / WMMA_N,
                    (M + WMMA_M - 1) / WMMA_M );
        dim3 blockDim(32, 1);  
        // Note: WMMA kernels typically use 32 threads / warp per tile.

        naive_tensor_core_kernel<<<gridDim, blockDim>>>(
            M, N, K, alpha, A16, B16, beta, C
        );

        // Free temporary FP16 matrices
        cudaCheck(cudaFree(A16));
        cudaCheck(cudaFree(B16));
        break;
    }
    case optimized_tensor_core:
    {
        half *A16, *B16;
        cudaCheck(cudaMalloc(&A16, sizeof(half) * M * K));
        cudaCheck(cudaMalloc(&B16, sizeof(half) * K * N));

        // Convert FP32 A,B → FP16 on GPU
        int numElemsA = M * K;
        int numElemsB = K * N;

        int threads = 256;
        int blocksA = ROUND_UP_TO_NEAREST(numElemsA, threads);
        int blocksB = ROUND_UP_TO_NEAREST(numElemsB, threads);

        convert_FP32_to_FP16<<<blocksA, threads>>>(A, A16, numElemsA);
        convert_FP32_to_FP16<<<blocksB, threads>>>(B, B16, numElemsB);

        int WARPS_M = BM / WMMA_M;
        int WARPS_N = BN / WMMA_N;
        int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
        
        dim3 grid(ROUND_UP_TO_NEAREST(N, BN), ROUND_UP_TO_NEAREST(M, BM));
        dim3 block(WARPS_PER_BLOCK * 32);
        
        optimized_tensor_core_kernel<<<grid, block>>>(
            M, N, K, alpha, A16, B16, beta, C
        );
        break;
    }
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}