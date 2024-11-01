#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

__global__ void calc_account(int *changes, int *account, int *sum, int clients, int periods, int *row_counter) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
	
    int row = blockIdx.y * TILE_SIZE_Y + ty;
    int col = blockIdx.x * TILE_SIZE_X + tx;

    __shared__ int tile[TILE_SIZE_Y][TILE_SIZE_X];

    // Ensure that each block row waits for the previous row to finish
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Wait until the counter allows this row to proceed
        while (atomicAdd(row_counter, 0) != blockIdx.y);
    }
    __syncthreads();

    int prev_block_val = blockIdx.y == 0 ? 0 : account[(row - 1) * clients + col];

    // Load data into shared memory
    if (row < periods && col < clients) {
        tile[ty][tx] = changes[row * clients + col] + prev_block_val;
    } else {
        tile[ty][tx] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < TILE_SIZE_Y; stride *= 2) {
        int val = ty >= stride ? tile[ty - stride][tx] : 0;
        __syncthreads();
        tile[ty][tx] += val;
        __syncthreads();
    }

    account[row * clients + col] = tile[ty][tx];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(row_counter, 1);
    }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 gridDim(clients / blockDim.x);

    for (int i = 0; i < periods / TILE_SIZE_Y; i++)
        calc_account<<<gridDim, blockDim>>>(changes, account, sum, clients, periods, i);
}

__global__ void calc_sum__parallel(int *account, int *sum, int clients, int periods) {
    // partial sum within one block (one row)
    __shared__ int partial_sum[128];

    int period = blockIdx.x;
    int tid = threadIdx.x;

    partial_sum[tid] = 0;

    for (int col = tid; col < clients; col += blockDim.x) {
        partial_sum[tid] += account[period * clients + col];  // Accumulate sum for this thread's chunk
    }
    __syncthreads();

    // Perform parallel reduction to sum the row elements using shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads(); // Ensure all threads have completed their work before proceeding
    }

    // The first thread in the block writes the result to the output row sum
    if (tid == 0) {
        sum[period] = partial_sum[0];  // Final sum for this row
    }
}

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void calc_sum__parallel_warp(int *account, int *sum, int clients, int periods) {
    int period = blockIdx.x;
    int tid = threadIdx.x;

    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    __shared__ int warp_sums[32];  // Max 32 warps per block

    int local_sum = 0;

    // Each thread accumulates a portion of the row's elements
    for (int col = tid; col < clients; col += blockDim.x) {
        local_sum += account[period * clients + col];
    }

    // Perform warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Store the result of each warp's reduction in shared memory
    if (lane == 0) {
        warp_sums[warpId] = local_sum;
    }
    __syncthreads();

    // The first warp in the block reduces the partial sums from each warp
    if (warpId == 0) {
        local_sum = (tid < blockDim.x / warpSize) ? warp_sums[lane] : 0;
        if (lane == 0) {
            for (int i = 1; i < blockDim.x / warpSize; i++) {
                local_sum += warp_sums[i];
            }
            sum[period] = local_sum; // Write the final sum to the output
        }
    }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 gridDim(clients / blockDim.x, periods / blockDim.y);

    int row_counter = 0;
    calc_account<<<gridDim, blockDim>>>(changes, account, sum, clients, periods, &row_counter);

    int BLOCK_SIZE = 128;
    int N_BLOCKS = periods;

    calc_sum__parallel<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
}
