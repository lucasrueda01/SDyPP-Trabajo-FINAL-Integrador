%%writefile md5_gpu.cu

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

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

// rotación izquierda (device)
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

// kernel que procesa todo el mensaje padded (un hilo)
__global__ void md5_kernel(const uint8_t* d_msg, size_t d_len, uint32_t* d_out_h) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // un solo hilo

    uint32_t h[4];
    h[0] = 0x67452301;
    h[1] = 0xefcdab89;
    h[2] = 0x98badcfe;
    h[3] = 0x10325476;

    size_t nblocks = d_len / 64;
    for (size_t i = 0; i < nblocks; ++i) {
        md5_transform_device(d_msg + i*64, h);
    }

    // escribir resultado en device memory
    d_out_h[0] = h[0];
    d_out_h[1] = h[1];
    d_out_h[2] = h[2];
    d_out_h[3] = h[3];
}

// ---------------- Host helpers (padding) ----------------
unsigned char* md5_pad_message_host(const unsigned char* initial_msg, size_t initial_len, size_t* out_len) {
    size_t new_len = initial_len + 1;
    while (new_len % 64 != 56) new_len++;
    new_len += 8;
    unsigned char* msg = (unsigned char*)malloc(new_len);
    if (!msg) return NULL;
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80;
    memset(msg + initial_len + 1, 0, new_len - initial_len - 1);
    uint64_t bits_len = (uint64_t)initial_len * 8;
    for (int i = 0; i < 8; ++i) msg[new_len - 8 + i] = (unsigned char)((bits_len >> (8 * i)) & 0xFF);
    *out_len = new_len;
    return msg;
}

// ---------------- Main (host) ----------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s \"texto a hashear\"\n", argv[0]);
        return 1;
    }
    const char* input = argv[1];
    size_t input_len = strlen(input);

    // padding en host
    size_t padded_len;
    unsigned char* padded = md5_pad_message_host((const unsigned char*)input, input_len, &padded_len);
    if (!padded) {
        fprintf(stderr, "Fallo malloc padding\n");
        return 1;
    }

    // reservar memoria device
    uint8_t* d_msg = nullptr;
    uint32_t* d_out = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_msg, padded_len);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc msg: %s\n", cudaGetErrorString(err)); free(padded); return 1; }
    err = cudaMemcpy(d_msg, padded, padded_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy msg: %s\n", cudaGetErrorString(err)); cudaFree(d_msg); free(padded); return 1; }

    err = cudaMalloc((void**)&d_out, 4 * sizeof(uint32_t));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc out: %s\n", cudaGetErrorString(err)); cudaFree(d_msg); free(padded); return 1; }

    // lanzar kernel (1 bloque, 1 hilo)
    md5_kernel<<<1,1>>>(d_msg, padded_len, d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Launch error: %s\n", cudaGetErrorString(err)); cudaFree(d_msg); cudaFree(d_out); free(padded); return 1; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "Sync error: %s\n", cudaGetErrorString(err)); cudaFree(d_msg); cudaFree(d_out); free(padded); return 1; }

    uint32_t h_out[4];
    err = cudaMemcpy(h_out, d_out, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "Memcpy out: %s\n", cudaGetErrorString(err)); cudaFree(d_msg); cudaFree(d_out); free(padded); return 1; }

    // imprimir en little-endian como MD5
    for (int i = 0; i < 4; ++i) {
        uint32_t v = h_out[i];
        printf("%02x%02x%02x%02x", v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF);
    }
    printf("\n");

    cudaFree(d_msg);
    cudaFree(d_out);
    free(padded);
    return 0;
}