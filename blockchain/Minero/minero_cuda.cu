// minero_cuda.cu  (con contador de intentos)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "md5.cu" // asegúrate que cuda_md5(...) exista

// -------------------- Device helpers --------------------
__device__ void int_to_str_device(int num, char *str) {
    int i = 0;
    if (num == 0) { str[0] = '0'; str[1] = '\0'; return; }
    unsigned int n = (num < 0) ? (unsigned int)(-num) : (unsigned int)num;
    char tmp[32];
    while (n > 0) { tmp[i++] = '0' + (n % 10); n /= 10; }
    int j = 0;
    if (num < 0) str[j++] = '-';
    while (i > 0) str[j++] = tmp[--i];
    str[j] = '\0';
}

__device__ int d_strlen(const char* s) {
    int i = 0; while (s[i] != '\0') ++i; return i;
}

__device__ bool starts_with_device(const char* hash, const char* prefix, int prefix_len) {
    for (int i = 0; i < prefix_len; ++i) if (hash[i] != prefix[i]) return false;
    return true;
}

__device__ void bytes_to_hex_device(const uint8_t* bytes, char* hex_out) {
    const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) {
        hex_out[i * 2]     = digits[(bytes[i] >> 4) & 0x0F];
        hex_out[i * 2 + 1] = digits[bytes[i] & 0x0F];
    }
    hex_out[32] = '\0';
}

// -------------------- Kernel con conteo de intentos --------------------
__global__ void mine_kernel_with_attempts(
    const char* input, int input_len,
    const char* prefix, int prefix_len,
    int from, int to,
    int* found_flag, int* out_nonce, char* out_hash,
    unsigned long long* attempts_global  // contador global (device pointer)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int BUFFER_MAX = 2048;
    char nonce_str[32];
    char buffer[BUFFER_MAX];
    uint8_t md5_raw[16];
    char md5_hex[33];

    // contador local para reducir contención de atomicAdd
    unsigned long long local_count = 0;
    const unsigned long long FLUSH_EVERY = 1024ULL; // ajustar si querés

    for (int nonce = from + idx; nonce <= to; nonce += stride) {
        // chequeo temprano si ya encontraron
        if (*found_flag) {
            // sumar lo local antes de salir
            if (local_count) atomicAdd(attempts_global, local_count);
            return;
        }

        // construir nonce_str y concatenar
        int_to_str_device(nonce, nonce_str);
        int nonce_len = d_strlen(nonce_str);

        if (nonce_len + input_len >= BUFFER_MAX) {
            // input demasiado grande para buffer local: saltar
            continue;
        }

        memcpy(buffer, nonce_str, nonce_len);
        memcpy(buffer + nonce_len, input, input_len);
        int total_len = nonce_len + input_len;

        // calcular MD5
        cuda_md5((uint8_t*)buffer, total_len, md5_raw);
        bytes_to_hex_device(md5_raw, md5_hex);

        // incrementamos local_count (1 intento hecho)
        local_count++;

        // si acumulamos suficiente, volcamos al contador global
        if (local_count >= FLUSH_EVERY) {
            atomicAdd(attempts_global, local_count);
            local_count = 0;
        }

        // comparar prefijo
        if (starts_with_device(md5_hex, prefix, prefix_len)) {
            // intentamos ser el primer ganador
            int prev = atomicCAS(found_flag, 0, 1);
            // volcar lo local antes de terminar
            if (local_count) atomicAdd(attempts_global, local_count);
            // registrar resultado si fuimos primeros
            if (prev == 0) {
                *out_nonce = nonce;
                for (int i = 0; i < 33; ++i) out_hash[i] = md5_hex[i];
            }
            return;
        }
    }

    // fin del loop: sumar lo que queda
    if (local_count) atomicAdd(attempts_global, local_count);
    return;
}

// -------------------- Host main --------------------
int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Uso: %s <from> <to> <prefix> <input> <output>\n", argv[0]);
        return 1;
    }

    int from = atoi(argv[1]); int to = atoi(argv[2]);
    const char* prefix = argv[3];
    const char* input = argv[4];
    const char* output = argv[5];

    int input_len = (int)strlen(input);
    int prefix_len = (int)strlen(prefix);

    // device pointers
    char *d_input = NULL, *d_prefix = NULL, *d_hash = NULL;
    int *d_found = NULL, *d_nonce = NULL;
    unsigned long long *d_attempts = NULL;

    cudaMalloc(&d_input, (input_len + 1) * sizeof(char));
    cudaMalloc(&d_prefix, (prefix_len + 1) * sizeof(char));
    cudaMalloc(&d_hash, 33 * sizeof(char));
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_nonce, sizeof(int));
    cudaMalloc(&d_attempts, sizeof(unsigned long long));

    cudaMemcpy(d_input, input, (input_len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, (prefix_len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, sizeof(int));
    cudaMemset(d_attempts, 0, sizeof(unsigned long long));

    // kernel config: ajustar parametros según GPU
    int threads = 256;
    int blocks = 4096;

    mine_kernel_with_attempts<<<blocks, threads>>>(
        d_input, input_len,
        d_prefix, prefix_len,
        from, to,
        d_found, d_nonce, d_hash,
        d_attempts
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        // cleanup
        if (d_input) cudaFree(d_input);
        if (d_prefix) cudaFree(d_prefix);
        if (d_hash) cudaFree(d_hash);
        if (d_found) cudaFree(d_found);
        if (d_nonce) cudaFree(d_nonce);
        if (d_attempts) cudaFree(d_attempts);
        return 1;
    }

    int found = 0;
    int nonce = 0;
    char hash_host[33]; memset(hash_host, 0, sizeof(hash_host));
    unsigned long long attempts_host = 0ULL;

    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&attempts_host, d_attempts, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    if (found) {
        cudaMemcpy(&nonce, d_nonce, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hash_host, d_hash, 33 * sizeof(char), cudaMemcpyDeviceToHost);
    }

    // escribir salida: "<nonce> <hash> <attempts>"
    FILE* f = fopen(output, "w");
    if (!f) {
        fprintf(stderr, "No se pudo abrir archivo %s\n", output);
    } else {
        if (found) {
            fprintf(f, "%d %s %llu", nonce, hash_host, attempts_host);
        } else {
            // sin resultado: nonce 0, hash zeros, attempts = attempts_host
            fprintf(f, "0 ");
            for (int i = 0; i < 32; ++i) fputc('0', f);
            fprintf(f, " %llu", attempts_host);
        }
        fclose(f);
    }

    // cleanup
    if (d_input) cudaFree(d_input);
    if (d_prefix) cudaFree(d_prefix);
    if (d_hash) cudaFree(d_hash);
    if (d_found) cudaFree(d_found);
    if (d_nonce) cudaFree(d_nonce);
    if (d_attempts) cudaFree(d_attempts);

    return 0;
}
