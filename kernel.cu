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

    if (period_id < periods) {
        int acc_sum = 0;
        for (int client = 0; client < clients; client++) {
            int idx = period_id * periods + client;
            acc_sum += account[idx];
        }

        sum[period_id] = acc_sum;
    }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    int BLOCK_SIZE = 16 * 16;
    int N_BLOCKS = (clients + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calc_account<<<N_BLOCKS, BLOCK_SIZE>>>(changes, account, sum, clients, periods);

    N_BLOCKS = (periods + BLOCK_SIZE - 1) / BLOCK_SIZE;

    calc_sum<<<N_BLOCKS, BLOCK_SIZE>>>(account, sum, clients, periods);
}
