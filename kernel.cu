// divide into 2 kernels? 1 for account calculation and 1 for sum?

__global__ void solve(int *changes, int *account, int *sum, int clients, int periods) {
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    int client_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (client_id < clients) {
        int accumulated_sum = 0;
        for (int period = 0; period < periods; period++) {
            int idx = day * clients + client_id; // Flattened index for (period, client)
            accumulated_sum += changes[idx];
            account[idx] = accumulated_sum;
        }
    }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
    int BLOCK_SIZE = 16 * 16;
    int N_BLOCKS = (clients + BLOCK_SIZE - 1) / BLOCK_SIZE;

    solve<<<N_BLOCKS, BLOCK_SIZE>>>(changes, account, sum, clients, periods);
}
