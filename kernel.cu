#define TILE_SIZE_X 32
#define TILE_SIZE_Y 32

__global__ void calc_account(int *changes, int *account, int *sum, int clients, int periods, int tile_y) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
	
    int row = tile_y * TILE_SIZE_Y + ty;
    int col = blockIdx.x * TILE_SIZE_X + tx;

    __shared__ int tile[TILE_SIZE_X][TILE_SIZE_Y];
    int prev_block_val = tile_y == 0 ? 0 : changes[row * clients - clients + col];

    // load data into shared memory
    if (row < periods && col < clients) {
        tile[ty][tx] = changes[row * clients + col] + prev_block_val;
    } else { // remove unnecessary else?
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

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    dim3 blockDim(TILE_SIZE_X, TILE_SIZE_Y);
    dim3 gridDim(clients / blockDim.x);

    for (int i = 0; i < periods / TILE_SIZE_Y; i++)
        calc_account<<<gridDim, blockDim>>>(changes, account, sum, clients, periods, i);

    BLOCK_SIZE = 128;
    N_BLOCKS = periods;

    calc_sum<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
}
