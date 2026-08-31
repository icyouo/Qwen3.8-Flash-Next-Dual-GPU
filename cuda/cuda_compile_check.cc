#include "q38/cuda_transport.h"
#include "q38/cuda_weights.h"
#include "q38/cuda_kernels.h"
#include "q38/cuda_gdn.h"
#include "q38/cuda_qsa.h"
#include "q38/cuda_hyper.h"
#include "q38/cuda_moe.h"
#include "q38/cuda_ple.h"
#include "q38/cuda_backend.h"

int main() {
    return q38::cuda_transport_compiled() &&
                   q38::cuda_device_weights_compiled() &&
                   q38::cuda_q38_kernels_compiled() &&
                   q38::cuda_q38_gdn_compiled() &&
                   q38::cuda_q38_qsa_compiled() &&
                   q38::cuda_q38_hyper_compiled() &&
                   q38::cuda_q38_moe_compiled() &&
                   q38::cuda_q38_ple_compiled() &&
                   q38::cuda_q38_backend_compiled()
               ? 0
               : 1;
}
