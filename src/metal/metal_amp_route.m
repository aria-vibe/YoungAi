/* metal_amp_route.m — 带路由的低秩放大器(契约 ds4_gpu_v41.h / ds4_gpu_bwd.h, CUDA 实现 src/cuda/cuda_amp_route.inc.cu)在 Metal 上还没写。
 * 10-29 它是段 B 的试验件, 只在 spark(CUDA)上训和判; Metal 上挂到带 ampr_Lnn.bin 的侧车会在这里报错停车(返回 0 → 前向失败),
 * 不会悄悄退回不带路由的算法 —— 那样跑出来的是另一个模型。要在 Mac 上用, 照 CUDA 那份补两个核(metal/v41_bwd.metal)。 */
#import "metal_v41.h"

int ds4_gpu_v41_amp_route_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                       const ds4_gpu_tensor *S, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, ds4_gpu_tensor *T,
                                       uint32_t n_tok, uint32_t D, uint32_t K, uint32_t NU, uint32_t NE) {
    (void)y; (void)x; (void)A; (void)B; (void)S; (void)sel; (void)rw; (void)T; (void)n_tok; (void)D; (void)K; (void)NU; (void)NE;
    fprintf(stderr, "ds4: ★带路由的放大器(ampr_Lnn.bin)在 Metal 上还没实现★\n");
    return 0;
}

int ds4_gpu_bwd_amp_route_tensor(ds4_gpu_tensor *gA, ds4_gpu_tensor *gB, ds4_gpu_tensor *gS, ds4_gpu_tensor *gx, const ds4_gpu_tensor *gy,
                                 const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B, const ds4_gpu_tensor *S,
                                 const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, ds4_gpu_tensor *T, ds4_gpu_tensor *gT,
                                 uint32_t n_tok, uint32_t D, uint32_t K, uint32_t NU, uint32_t NE) {
    (void)gA; (void)gB; (void)gS; (void)gx; (void)gy; (void)x; (void)A; (void)B; (void)S; (void)sel; (void)rw; (void)T; (void)gT;
    (void)n_tok; (void)D; (void)K; (void)NU; (void)NE;
    fprintf(stderr, "ds4: ★带路由的放大器反向在 Metal 上还没实现★\n");
    return 0;
}
