%%writefile minero_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>

constexpr int RANGE_FROM = 0;
constexpr int RANGE_TO   = 200000;

// ---------------- MD5 ----------------
__constant__ uint32_t dev_shifts[16] = { 7,12,17,22, 5,9,14,20, 4,11,16,23, 6,10,15,21 };
__constant__ uint32_t dev_sines[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

// rotación izquierda
__device__ __forceinline__ uint32_t dev_left_rotate(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

// transform que procesa un bloque de 64 bytes
__device__ void md5_transform_device(const uint8_t* chunk, uint32_t* h) {
    uint32_t M[16];
    for (int i = 0; i < 16; ++i) {
        M[i] = (uint32_t)chunk[i*4]
             | ((uint32_t)chunk[i*4+1] << 8)
             | ((uint32_t)chunk[i*4+2] << 16)
             | ((uint32_t)chunk[i*4+3] << 24);
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];

    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if (i < 16) {
            F = (b & c) | ((~b) & d);
            g = i;
        } else if (i < 32) {
            F = (d & b) | ((~d) & c);
            g = (5*i + 1) & 15;
        } else if (i < 48) {
            F = b ^ c ^ d;
            g = (3*i + 5) & 15;
        } else {
            F = c ^ (b | (~d));
            g = (7*i) & 15;
        }

        uint32_t tmp = a + F + dev_sines[i] + M[g];
        a = d; d = c; c = b;

        uint32_t sh;
        if (i < 16) sh = dev_shifts[i % 4];
        else if (i < 32) sh = dev_shifts[4 + (i % 4)];
        else if (i < 48) sh = dev_shifts[8 + (i % 4)];
        else sh = dev_shifts[12 + (i % 4)];

        b = b + dev_left_rotate(tmp, sh);
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
}


__device__ void cuda_md5_singleblock(const uint8_t* initial_msg, int initial_len, uint8_t* digest_out) {
    // estados iniciales
    uint32_t h[4];
    h[0] = 0x67452301;
    h[1] = 0xefcdab89;
    h[2] = 0x98badcfe;
    h[3] = 0x10325476;

    // preparar bloque en stack
    uint8_t chunk[64];
    // inicializar con ceros
    for (int i = 0; i < 64; ++i) chunk[i] = 0;

    // copiar mensaje
    for (int i = 0; i < initial_len; ++i) chunk[i] = initial_msg[i];

    // padding
    chunk[initial_len] = 0x80;

    // longitud en bits, little-endian
    uint64_t bits_len = (uint64_t)initial_len * 8ULL;
    for (int i = 0; i < 8; ++i) chunk[56 + i] = (uint8_t)((bits_len >> (8 * i)) & 0xFF);

    // procesar un bloque
    md5_transform_device(chunk, h);

    // escribir digest_out en formato little-endian
    for (int i = 0; i < 4; ++i) {
        uint32_t v = h[i];
        digest_out[i*4 + 0] = (uint8_t)(v & 0xFF);
        digest_out[i*4 + 1] = (uint8_t)((v >> 8) & 0xFF);
        digest_out[i*4 + 2] = (uint8_t)((v >> 16) & 0xFF);
        digest_out[i*4 + 3] = (uint8_t)((v >> 24) & 0xFF);
    }
}

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
    unsigned long long* attempts_global
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const int BUFFER_MAX = 2048;
    char nonce_str[32];
    char buffer[BUFFER_MAX];
    uint8_t md5_raw[16];
    char md5_hex[33];

    unsigned long long local_count = 0;
    const unsigned long long FLUSH_EVERY = 1024ULL;

    for (int nonce = from + idx; nonce <= to; nonce += stride) {
        if (*found_flag) {
            if (local_count) atomicAdd(attempts_global, local_count);
            return;
        }

        int_to_str_device(nonce, nonce_str);
        int nonce_len = d_strlen(nonce_str);

        // construir buffer: nonce + input
        if (nonce_len + input_len >= BUFFER_MAX) {
            // evitar overflow
            continue;
        }
        // copiar nonce
        for (int i = 0; i < nonce_len; ++i) buffer[i] = nonce_str[i];
        // copiar input despues
        for (int i = 0; i < input_len; ++i) buffer[nonce_len + i] = input[i];
        int total_len = nonce_len + input_len;

        // single-block requiere total_len <= 55
        if (total_len > 55) {
            // contar intento tambien
            local_count++;
            if (local_count >= FLUSH_EVERY) { atomicAdd(attempts_global, local_count); local_count = 0; }
            continue;
        }

        // calcular MD5
        cuda_md5_singleblock((const uint8_t*)buffer, total_len, md5_raw);
        bytes_to_hex_device(md5_raw, md5_hex);

        local_count++;
        if (local_count >= FLUSH_EVERY) {
            atomicAdd(attempts_global, local_count);
            local_count = 0;
        }

        if (starts_with_device(md5_hex, prefix, prefix_len)) {
            int prev = atomicCAS(found_flag, 0, 1);
            if (local_count) atomicAdd(attempts_global, local_count);
            if (prev == 0) {
                *out_nonce = nonce;
                for (int i = 0; i < 33; ++i) out_hash[i] = md5_hex[i];
            }
            return;
        }
    }

    if (local_count) atomicAdd(attempts_global, local_count);
    return;
}

// -------------------- Host main --------------------
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <prefix> <input> <output>\n", argv[0]);
        fprintf(stderr, "RANGO hardcodeado en el codigo: [%d .. %d]\n", RANGE_FROM, RANGE_TO);
        return 1;
    }

    const char* prefix = argv[1];
    const char* input = argv[2];
    const char* output = argv[3];

    int input_len = (int)strlen(input);
    int prefix_len = (int)strlen(prefix);

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

    // parametros launching
    int threads = 256;
    int blocks = 1024; // aprox 262k hilos

    mine_kernel_with_attempts<<<blocks, threads>>>(
        d_input, input_len,
        d_prefix, prefix_len,
        RANGE_FROM, RANGE_TO,
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

    // salida: "<nonce> <hash> <attempts>"
    FILE* f = fopen(output, "w");
    if (!f) {
        fprintf(stderr, "No se pudo abrir archivo %s\n", output);
    } else {
        if (found) {
            fprintf(f, "%d %s %llu", nonce, hash_host, attempts_host);
        } else {
            fprintf(f, "0 ");
            for (int i = 0; i < 32; ++i) fputc('0', f);
            fprintf(f, " %llu", attempts_host);
        }
        fclose(f);
    }

    if (d_input) cudaFree(d_input);
    if (d_prefix) cudaFree(d_prefix);
    if (d_hash) cudaFree(d_hash);
    if (d_found) cudaFree(d_found);
    if (d_nonce) cudaFree(d_nonce);
    if (d_attempts) cudaFree(d_attempts);

    return 0;
}