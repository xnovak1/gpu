__global__ void calc_account(int *changes, int *account, int *sum, int clients, int periods) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    int client_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (client_id < clients) {
        int acc_sum = 0;
        for (int period = 0; period < periods; period++) {
            int idx = period * clients + client_id; // Flattened index for (period, client)
            acc_sum += changes[idx];
            account[idx] = acc_sum;
        }
    }
}

__global__ void calc_sum(int *account, int *sum, int clients, int periods) {
    // partial sum within one block (one row)
    __shared__ int partial_sum[256];

    int period = blockIdx.x;
    int tid = threadIdx.x;

    partial_sum[tid] = 0;

    for (int col = tid; col < clients; col += blockDim.x) {
        partial_sum[tid] += account[period * clients + col];  // Accumulate sum for this thread's chunk
    }
    __syncthreads();

    // Perform parallel reduction to sum the row elements using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
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
    int BLOCK_SIZE = 128; // 128
    int N_BLOCKS = (clients + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calc_account<<<N_BLOCKS, BLOCK_SIZE>>>(changes, account, sum, clients, periods);

    BLOCK_SIZE = 128; // 1024 max
    N_BLOCKS = periods;

    calc_sum<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
}
