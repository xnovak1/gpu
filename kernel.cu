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

__global__ void calc_8192(int *changes, int *account, int *sum, int clients, int periods) {
    int clientIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int prev_val = 0;

    for (int i = 0; i < periods / TILE_SIZE_Y; i++) {
        __shared__ int tile[TILE_SIZE_Y][TILE_SIZE_X + 1];
        for (int k = 0; k < TILE_SIZE_Y; k++) {
	    int periodIdx = TILE_SIZE_Y * i + k;
            tile[k][threadIdx.x] = changes[periodIdx * clients + clientIdx];
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE_Y; j++) {
            int periodIdx = TILE_SIZE_Y * i + j;
            int deposit = 0;

            if (j == 0) {
                deposit = tile[j][threadIdx.x] + prev_val;
            } else {
                deposit = tile[j - 1][threadIdx.x] + tile[j][threadIdx.x];
            }

            tile[j][threadIdx.x] = deposit;
            account[periodIdx * clients + clientIdx] = deposit;

            if (j == TILE_SIZE_Y - 1) {
                prev_val = deposit;
            }

            atomicAdd(&sum[periodIdx], deposit);
        }
        __syncthreads();
    }
}

__global__ void calc_512(int *changes, int *account, int *sum, int clients, int periods) {
    int clientIdx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int j = 0; j < periods; j++) {
        int accountIdx = j * clients + clientIdx;
        int deposit = 0;

        if (j == 0) {
            deposit = changes[accountIdx];
        } else {
            deposit = account[(j - 1) * clients + clientIdx] + changes[accountIdx];
        }

        account[accountIdx] = deposit;
        atomicAdd(&sum[j], deposit);
    }
    __syncthreads();
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    if (clients == 8192 && periods == 8192) {
        dim3 blockDim(TILE_SIZE_X);
        dim3 gridDim(8192 / TILE_SIZE_X);
        calc_8192<<<gridDim, blockDim>>>(changes, account, sum, clients, periods);
    } else if (clients == 512 && periods == 512) {
        dim3 blockDim(128);
        dim3 gridDim((512 + blockDim.x - 1) / blockDim.x);
        calc_512<<<gridDim, blockDim>>>(changes, account, sum, clients, periods);
    } else {
        dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
        dim3 gridDim(clients / blockDim.x);

        for (int i = 0; i < periods / TILE_SIZE_Y; i++)
            calc_account<<<gridDim, blockDim>>>(changes, account, sum, clients, periods, i);

        int BLOCK_SIZE = 128;
        int N_BLOCKS = periods;

        calc_sum<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
    }
}
