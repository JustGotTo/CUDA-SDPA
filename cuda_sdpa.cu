#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cmath>

__global__ void attention_scores(
    const float* Q,
    const float* K,
    float* S,
    int N,
    int D,
    float scale
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N) {

        int row = idx / N;
        int col = idx % N;

        float sum = 0.0f;

        for (int k = 0; k < D; k++) {
            sum += Q[row * D + k] *
                   K[col * D + k];
        }

        S[index] = sum * scale;
    }
}

__global__ void attention_softmax(
    const float* S,
    float* A,
    int N
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {

        float max_val = -INFINITY;

        // Find maximum score in this row
        for (int j = 0; j < N; j++) {

            float value = S[index * N + j];

            if (value > max_val) {
                max_val = value;
            }
        }

        float sum = 0.0f;

        for (int j = 0; j < N; j++) {

            float value = expf(
                S[index * N + j] - max_val
            );

            A[index * N + j] = value;
            sum += value;
        }

        // Normalise
        for (int j = 0; j < N; j++) {

            A[index * N + j] /= sum;
        }
    }
}

__global__ void attention_output(
    const float* A,
    const float* V,
    float* O,
    int N,
    int Dv
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N * Dv) {

        int row = idx / Dv;
        int col = idx % Dv;

        float sum = 0.0f;

        // Weighted sum of value vectors
        for (int k = 0; k < N; k++) {

            sum += A[row * N + k] *
                   V[k * Dv + col];

        }

        O[index] = sum;
    }
}

torch::Tensor attention (
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
    )
{

    TORCH_CHECK(
        Q.is_cuda(),
        "Query must be on CUDA"
        );
    TORCH_CHECK(
        K.is_cuda(),
        "Key must be on CUDA"
        );
    TORCH_CHECK(
        V.is_cuda(),
        "Value must be on CUDA"
        );

    TORCH_CHECK(
        Q.device() == K.device() &&
        Q.device() == V.device(),
        "Tensors must be on the same GPU"
    );

    TORCH_CHECK(
        Q.scalar_type() == torch::kFloat32 &&
        K.scalar_type() == torch::kFloat32 &&
        V.scalar_type() == torch::kFloat32,
        "Only FP32 is supported"
    );

    TORCH_CHECK(
        Q.is_contiguous() &&
        K.is_contiguous() &&
        V.is_contiguous(),
        "Inputs must be contiguous"
    );

    TORCH_CHECK(
        Q.dim() == 2 &&
        K.dim() == 2 &&
        V.dim() == 2,
        "Inputs must be two-dimensional"
    );

    int N = Q.size(0);
    int D = Q.size(1);
    int Dv = V.size(1);

    TORCH_CHECK(
        N > 0 && D > 0 && Dv > 0,
        "Dimensions must be positive"
        );

    TORCH_CHECK(
        K.size(0) == N && K.size(1) == D,
        "K must have shape [N, D]"
    );

    TORCH_CHECK(
        V.size(0) == N,
        "V must have shape [N, Dv]"
    );

    auto S = torch::empty(
            {N, N}, Q.options()
        );

    auto A = torch::empty(
        {N, N}, Q.options()
    );

    auto O = torch::empty(
        {N, Dv}, Q.options()
    );

    int threads = 256;
    float scale = 1.0f / sqrtf((float)D);

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream();

    int score_blocks = (N * N + threads - 1) / threads;

    attention_scores<<<score_blocks, threads, 0, stream>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        S.data_ptr<float>(),
        N,
        D,
        scale
    );

    int softmax_blocks = (N + threads - 1) / threads;

    attention_softmax<<<softmax_blocks, threads, 0, stream>>>(
        S.data_ptr<float>(),
        A.data_ptr<float>(),
        N
    );

    int output_blocks = (N * Dv + threads - 1) / threads;

    attention_output<<<output_blocks, threads, 0, stream>>>(
        A.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N,
        Dv
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def(
        "attention",
        &attention_cuda,
        "Custom CUDA Scaled Dot Product Attention"
    );

}

