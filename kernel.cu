#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

__global__ void calc_account(int *changes, int *account, int *sum, int clients, int periods, int tile_y) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
	
    int row = tile_y * TILE_SIZE_Y + ty;
    int col = blockIdx.x * TILE_SIZE_X + tx;

    __shared__ int tile[TILE_SIZE_Y][TILE_SIZE_X];
    int prev_block_val = tile_y == 0 ? 0 : account[(row - 1) * clients + col];

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
}

__global__ void calc_sum(int *account, int *sum, int clients, int periods) {
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

__global__ void calc_account_8192(int *changes, int *account, int *sum, int clients, int periods) {
    int clientIdx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared_sum[8192];
    for (int i = 0; i < 64; i++)
        shared_sum[threadIdx.x * 64 + i] = 0;

    for (int j = 0; j < periods; j++) {
	    int accountIdx = j * clients + clientIdx;
        int deposit = 0;

        if (j == 0) {
            deposit = changes[accountIdx];
        } else {
            deposit = account[(j - 1) * clients + clientIdx] + changes[accountIdx];
        }

        account[accountIdx] = deposit;
        atomicAdd(&shared_sum[j], deposit);
    }
    __syncthreads();

    for (int i = 0; i < 64; i++) {
        int idx = threadIdx.x * 64 + i;
        atomicAdd(&sum[idx], shared_sum[idx]);
    }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    dim3 blockDim(128);
	dim3 gridDim((clients + blockDim.x - 1) / blockDim.x);
	calc_account_8192<<<gridDim, blockDim>>>(changes, account, sum, clients, periods);
}
