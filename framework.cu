#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

// nvcc -o framework framework.cu

// airacuda:  20 000 Mv/s
// barracuda: 40 000 Mv/s

// input divisible by 128

#include "kernel.cu"
#include "kernel_CPU.C"

int test_correctness() {
    int ret_val = 0;

    // CPU data
    int *changes, *account, *sum, *account_gpu, *sum_gpu;
    changes = account = sum = account_gpu = sum_gpu = NULL;
    // GPU counterparts
    int *dchanges, *daccount, *dsum;
    dchanges = daccount = dsum = NULL;

    // 3*3 = 9 iterations
    for (int CLIENTS = 2048; CLIENTS <= 8192; CLIENTS *= 2) {
        for (int PERIODS = 2048; PERIODS <= 8192; PERIODS *= 2) {
            // allocate and set host memory
            changes = (int*)malloc(CLIENTS*PERIODS*sizeof(changes[0]));
            account = (int*)malloc(CLIENTS*PERIODS*sizeof(account[0]));
            sum = (int*)malloc(PERIODS*sizeof(sum[0]));
            account_gpu = (int*)malloc(CLIENTS*PERIODS*sizeof(account_gpu[0]));
            sum_gpu = (int*)malloc(PERIODS*sizeof(sum[0]));

            for (int i = 0; i < CLIENTS*PERIODS; i++)
                changes[i] = int(100.0f*(float)rand() / float(RAND_MAX));
        
            // allocate and set device memory
            if (cudaMalloc((void**)&dchanges, CLIENTS*PERIODS*sizeof(dchanges[0])) != cudaSuccess
            || cudaMalloc((void**)&daccount, CLIENTS*PERIODS*sizeof(daccount[0])) != cudaSuccess 
            || cudaMalloc((void**)&dsum, PERIODS*sizeof(dsum[0])) != cudaSuccess){
                fprintf(stderr, "Device memory allocation error!\n");
                goto cleanup;
            }
            cudaMemcpy(dchanges, changes, CLIENTS*PERIODS*sizeof(dchanges[0]), cudaMemcpyHostToDevice);
            cudaMemset(daccount, 0, CLIENTS*PERIODS*sizeof(daccount[0]));
            cudaMemset(dsum, 0, PERIODS*sizeof(dsum[0]));

            solveCPU(changes, account, sum, CLIENTS, PERIODS);
            solveGPU(dchanges, daccount, dsum, CLIENTS, PERIODS);

            // check GPU results
            cudaMemcpy(account_gpu, daccount, CLIENTS*PERIODS*sizeof(daccount[0]), cudaMemcpyDeviceToHost);
            for (int j = 0; j < PERIODS; j++)
                for (int i = 0; i < CLIENTS; i++)
                    if (account[j*CLIENTS+i] != account_gpu[j*CLIENTS+i]) {
                        ret_val = -1;
                        fprintf(stderr, "Account data mismatch at index %i, %i: %i should be %i :-(\n", i, j, account_gpu[j*CLIENTS+i], account[j*CLIENTS+i]);
                        goto cleanup;
                    }
            cudaMemcpy(sum_gpu, dsum, PERIODS*sizeof(dsum[0]), cudaMemcpyDeviceToHost);
            for (int i = 0; i < PERIODS; i++)
                if (sum[i] != sum_gpu[i]) {
                    ret_val = -1;
                    fprintf(stderr, "Sum data mismatch at index %i: %i should be %i :-(\n", i, sum_gpu[i], sum[i]);
                    goto cleanup;
                }
            
            if (dchanges) cudaFree(dchanges);
            if (daccount) cudaFree(daccount);
            if (dsum) cudaFree(dsum);
            if (changes) free(changes);
            if (account) free(account);
            if (sum) free(sum);
            if (account_gpu) free(account_gpu);
            if (sum_gpu) free(sum_gpu);
	    printf("loop\n");
        }
    }

cleanup:
    if (dchanges) cudaFree(dchanges);
    if (daccount) cudaFree(daccount);
    if (dsum) cudaFree(dsum);
    if (changes) free(changes); // seg fault?!
    if (account) free(account);
    if (sum) free(sum);
    if (account_gpu) free(account_gpu);
    if (sum_gpu) free (sum_gpu);

    return ret_val;
}

void print_matrices(int *cpu, int *gpu, int rows, int cols) {
    printf("\n");
    printf("Correct matrix:\n\n");

    for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
	    printf("%d ", cpu[i * cols + j]);
	}

	printf("|\n");
    }

    printf("\n");
    printf("GPU matrix:\n\n");

    for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {
	    printf("%d ", gpu[i * cols + j]);
	}

	printf("|\n");
    }
}

void test_performance() {
    const int CLIENTS = 8192;
    const int PERIODS = 8192;

    // CPU data
    int *changes, *account, *sum, *account_gpu, *sum_gpu;
    changes = account = sum = account_gpu = sum_gpu = NULL;
    // GPU counterparts
    int *dchanges, *daccount, *dsum;
    dchanges = daccount = dsum = NULL;

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate and set host memory
    changes = (int*)malloc(CLIENTS*PERIODS*sizeof(changes[0]));
    account = (int*)malloc(CLIENTS*PERIODS*sizeof(account[0]));
    sum = (int*)malloc(PERIODS*sizeof(sum[0]));
    account_gpu = (int*)malloc(CLIENTS*PERIODS*sizeof(account_gpu[0]));
    sum_gpu = (int*)malloc(PERIODS*sizeof(sum[0]));

    for (int i = 0; i < CLIENTS*PERIODS; i++)
        changes[i] = int(100.0f*(float)rand() / float(RAND_MAX));
 
    // allocate and set device memory
    if (cudaMalloc((void**)&dchanges, CLIENTS*PERIODS*sizeof(dchanges[0])) != cudaSuccess
    || cudaMalloc((void**)&daccount, CLIENTS*PERIODS*sizeof(daccount[0])) != cudaSuccess 
    || cudaMalloc((void**)&dsum, PERIODS*sizeof(dsum[0])) != cudaSuccess){
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(dchanges, changes, CLIENTS*PERIODS*sizeof(dchanges[0]), cudaMemcpyHostToDevice);
    cudaMemset(daccount, 0, CLIENTS*PERIODS*sizeof(daccount[0]));
    cudaMemset(dsum, 0, PERIODS*sizeof(dsum[0]));

    // solve on CPU
    printf("Solving on CPU...\n");
    cudaEventRecord(start, 0);
    solveCPU(changes, account, sum, CLIENTS, PERIODS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megavalues/s\n",
        float(CLIENTS)*float(PERIODS)/time/1e3f);

    // solve on GPU
    printf("Solving on GPU...\n");
    cudaEventRecord(start, 0);
    // for(int i =0; i < 100; i++) 
        solveGPU(dchanges, daccount, dsum, CLIENTS, PERIODS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU performance: %f megavalues/s\n",
        float(CLIENTS)*float(PERIODS)/time/1e3f);

    // check GPU results
    cudaMemcpy(account_gpu, daccount, CLIENTS*PERIODS*sizeof(daccount[0]), cudaMemcpyDeviceToHost);
    for (int j = 0; j < PERIODS; j++)
        for (int i = 0; i < CLIENTS; i++)
            if (account[j*CLIENTS+i] != account_gpu[j*CLIENTS+i]) { 
                fprintf(stderr, "Account data mismatch at index %i, %i: %i should be %i :-(\n", i, j, account_gpu[j*CLIENTS+i], account[j*CLIENTS+i]);
                goto cleanup;
            }
    cudaMemcpy(sum_gpu, dsum, PERIODS*sizeof(dsum[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < PERIODS; i++)
        if (sum[i] != sum_gpu[i]) {
            fprintf(stderr, "Sum data mismatch at index %i: %i should be %i :-(\n", i, sum_gpu[i], sum[i]);
                goto cleanup;
        }

    cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //  print_matrices(account, account_gpu, PERIODS, CLIENTS);

    if (dchanges) cudaFree(dchanges);
    if (daccount) cudaFree(daccount);
    if (dsum) cudaFree(dsum);
    if (changes) free(changes);
    if (account) free(account);
    if (sum) free(sum);
    if (account_gpu) free(account_gpu);
    if (sum_gpu) free (sum_gpu);
}

int main(int argc, char **argv){
    // parse command line
    int device = 0;
    if (argc == 2) 
        device = atoi(argv[1]);
    if (cudaSetDevice(device) != cudaSuccess){
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

    // test_correctness();
    // printf("Tests OK.\n\n");
    test_performance();
    

    return 0;
}
