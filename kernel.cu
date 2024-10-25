#define TILE_SIZE 32

__global__ void calc_account(int *changes, int *account, int *sum, int clients, int periods) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ int tile[TILE_SIZE][TILE_SIZE];

    // load data into shared memory
    if (row < periods && col < clients) {
        tile[threadIdx.y][threadIdx.x] = changes[row * clients + col];
    } else {
	tile[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    // accumulate across rows within this tile in shared memory
    if (row < periods && col < clients) {
	int acc_sum = tile[threadIdx.y][threadIdx.x];

	// accumulate verically within the tile (for the given client)
	for (int i = 1; i <= threadIdx.y; i++) {
            acc_sum += tile[threadIdx.y - i][threadIdx.x];
	}

	account[row * clients + col] = acc_sum;
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
    int BLOCK_SIZE = TILE_SIZE * TILE_SIZE; 
    int N_BLOCKS = (clients / TILE_SIZE) * (periods / TILE_SIZE);

    calc_account<<<N_BLOCKS, BLOCK_SIZE>>>(changes, account, sum, clients, periods);

    BLOCK_SIZE = 128;
    N_BLOCKS = periods;

    calc_sum<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
}
