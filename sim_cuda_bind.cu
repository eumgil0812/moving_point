// 빌드 예시(리눅스):
//   EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
//   nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
//        $(python -m pybind11 --includes) \
//        -I/usr/local/cuda/include sim_cuda_bind.cu -lcudart \
//        -o sim_cuda${EXT}

/*
EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
     $(python -m pybind11 --includes) \
     -I/usr/local/cuda/include sim_cuda_bind.cu -lcudart \
     -o sim_cuda${EXT}

python - <<'PY'
import numpy as np, sim_cuda as sim
N,K=100000,64
pos=np.random.rand(N,2)*100
vel=np.random.randn(1,1,2) # dummy for signature
pos_batched=pos[None,:,:]  # [1,N,2]
vel_batched=np.zeros_like(pos_batched)
sim.update_boxes(pos_batched, vel_batched, 100.0, 100.0, 0.01)
assign, counts = sim.nearest_assign(pos, np.random.rand(K,2)*100, 0.0)
print(assign.shape, counts.sum()==N)
PY

*/
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

// ---- 유틸 ----
#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(_e)); \
} while(0)

template<typename T>
T* dmalloc(size_t n) {
    T* p=nullptr;
    CUDA_CHECK(cudaMalloc(&p, n*sizeof(T)));
    return p;
}

inline dim3 grid1d(int n, int block=256) {
    int g = (n + block - 1) / block;
    return dim3(g,1,1);
}

// ---- 1) 위치 업데이트 + 박스 반사 경계 ----
__global__ void k_update_boxes(double* __restrict__ pos, // [N,2]
                               double* __restrict__ vel, // [N,2]
                               int N, double W, double H, double dt)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double x = pos[2*n+0] + vel[2*n+0]*dt;
    double y = pos[2*n+1] + vel[2*n+1]*dt;

    // X
    if (x < 0.0) { x = -x; vel[2*n+0] = -vel[2*n+0]; }
    else if (x > W) { x = 2.0*W - x; vel[2*n+0] = -vel[2*n+0]; }
    // Y
    if (y < 0.0) { y = -y; vel[2*n+1] = -vel[2*n+1]; }
    else if (y > H) { y = 2.0*H - y; vel[2*n+1] = -vel[2*n+1]; }

    pos[2*n+0] = x;
    pos[2*n+1] = y;
}

void update_boxes_cuda(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                       py::array_t<double, py::array::c_style | py::array::forcecast> vel,
                       double box_width, double box_height, double dt)
{
    // 파이썬 쪽은 [1,N,2]로 오지만, 여기선 [N,2]로 평탄화해서 처리
    if (pos.ndim()!=3 || pos.shape(2)!=2) throw std::runtime_error("pos must be [B,N,2]");
    if (vel.ndim()!=3 || vel.shape(2)!=2) throw std::runtime_error("vel must be [B,N,2]");
    if (pos.shape(0)!=1 || vel.shape(0)!=1) throw std::runtime_error("Only B=1 supported");

    const int N = static_cast<int>(pos.shape(1));
    if (N<=0) return;

    auto P = pos.mutable_unchecked<3>();
    auto V = vel.mutable_unchecked<3>();

    // 호스트→연속 버퍼로 복사
    std::vector<double> h_pos(2*N), h_vel(2*N);
    for (int n=0; n<N; ++n) {
        h_pos[2*n+0] = P(0,n,0);
        h_pos[2*n+1] = P(0,n,1);
        h_vel[2*n+0] = V(0,n,0);
        h_vel[2*n+1] = V(0,n,1);
    }

    // 디바이스 메모리
    double *d_pos = dmalloc<double>(2*N);
    double *d_vel = dmalloc<double>(2*N);
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), sizeof(double)*2*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), sizeof(double)*2*N, cudaMemcpyHostToDevice));

    // 커널
    int block=256;
    k_update_boxes<<<grid1d(N,block), block>>>(d_pos, d_vel, N, box_width, box_height, dt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos, sizeof(double)*2*N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel, sizeof(double)*2*N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));

    // 결과 반영
    for (int n=0; n<N; ++n) {
        P(0,n,0) = h_pos[2*n+0];
        P(0,n,1) = h_pos[2*n+1];
        V(0,n,0) = h_vel[2*n+0];
        V(0,n,1) = h_vel[2*n+1];
    }
}

// ---- 2) 최근접 중심 배정 assign[N], counts[K] ----
__global__ void k_nearest_assign(const double* __restrict__ pos,     // [N,2]
                                 const double* __restrict__ centers, // [K,2]
                                 int * __restrict__ assign,          // [N]
                                 int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double px = pos[2*n+0];
    double py = pos[2*n+1];

    double best = 1e300;
    int bestk = 0;
    for (int k=0;k<K;++k) {
        double dx = px - centers[2*k+0];
        double dy = py - centers[2*k+1];
        double d2 = dx*dx + dy*dy;
        if (d2 < best) { best=d2; bestk=k; }
    }
    assign[n] = bestk;
}

__global__ void k_count_from_assign(const int* __restrict__ assign, int* __restrict__ counts, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    atomicAdd(&counts[assign[n]], 1);
}

py::tuple nearest_assign_cuda(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                              py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                              double /*box_size_unused*/)
{
    if (pos.ndim()!=2 || pos.shape(1)!=2)  throw std::runtime_error("pos must be [N,2]");
    if (centers.ndim()!=2 || centers.shape(1)!=2) throw std::runtime_error("centers must be [K,2]");
    const int N = static_cast<int>(pos.shape(0));
    const int K = static_cast<int>(centers.shape(0));
    if (N<=0 || K<=0) throw std::runtime_error("N,K>0");

    auto P = pos.unchecked<2>();
    auto C = centers.unchecked<2>();

    // 호스트 연속화
    std::vector<double> h_pos(2*N), h_ctr(2*K);
    for (int n=0;n<N;++n){ h_pos[2*n+0]=P(n,0); h_pos[2*n+1]=P(n,1); }
    for (int k=0;k<K;++k){ h_ctr[2*k+0]=C(k,0); h_ctr[2*k+1]=C(k,1); }

    // 디바이스
    double *d_pos = dmalloc<double>(2*N);
    double *d_ctr = dmalloc<double>(2*K);
    int    *d_assign = dmalloc<int>(N);
    int    *d_counts = dmalloc<int>(K);
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), sizeof(double)*2*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctr, h_ctr.data(), sizeof(double)*2*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(int)*K));

    // 커널
    int block=256;
    k_nearest_assign<<<grid1d(N,block), block>>>(d_pos, d_ctr, d_assign, N, K);
    CUDA_CHECK(cudaGetLastError());
    k_count_from_assign<<<grid1d(N,block), block>>>(d_assign, d_counts, N);
    CUDA_CHECK(cudaGetLastError());

    // 결과 회수
    py::array_t<int32_t> assign({N});
    py::array_t<int32_t> counts({K});
    std::vector<int> h_assign(N), h_counts(K);
    CUDA_CHECK(cudaMemcpy(h_assign.data(), d_assign, sizeof(int)*N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, sizeof(int)*K, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_ctr));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_counts));

    // 파이썬 배열에 복사
    auto A = assign.mutable_unchecked<1>();
    for (int n=0;n<N;++n) A(n)=h_assign[n];
    auto CNT = counts.mutable_unchecked<1>();
    for (int k=0;k<K;++k) CNT(k)=h_counts[k];

    return py::make_tuple(assign, counts);
}

// ---- 3) 반경 r 이내 카운트(겹침 허용) counts[K] ----
// 스레드 (k,n) 2D 매핑 → 조건 만족 시 counts[k]에 atomicAdd
__global__ void k_count_within_radius_multi(const double* __restrict__ pos,     // [N,2]
                                            const double* __restrict__ centers, // [K,2]
                                            int* __restrict__ counts,
                                            int N, int K, double r2)
{
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k>=K || n>=N) return;

    double dx = pos[2*n+0] - centers[2*k+0];
    double dy = pos[2*n+1] - centers[2*k+1];
    if (dx*dx + dy*dy <= r2) atomicAdd(&counts[k], 1);
}

py::array_t<int32_t> count_within_radius_multi_cuda(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                                                    py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                                                    double radius)
{
    if (pos.ndim()!=2 || pos.shape(1)!=2)  throw std::runtime_error("pos must be [N,2]");
    if (centers.ndim()!=2 || centers.shape(1)!=2) throw std::runtime_error("centers must be [K,2]");
    const int N = static_cast<int>(pos.shape(0));
    const int K = static_cast<int>(centers.shape(0));
    auto P = pos.unchecked<2>();
    auto C = centers.unchecked<2>();

    std::vector<double> h_pos(2*N), h_ctr(2*K);
    for (int n=0;n<N;++n){ h_pos[2*n+0]=P(n,0); h_pos[2*n+1]=P(n,1); }
    for (int k=0;k<K;++k){ h_ctr[2*k+0]=C(k,0); h_ctr[2*k+1]=C(k,1); }

    double r2 = radius*radius;

    double *d_pos = dmalloc<double>(2*N);
    double *d_ctr = dmalloc<double>(2*K);
    int *d_cnt = dmalloc<int>(K);
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), sizeof(double)*2*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ctr, h_ctr.data(), sizeof(double)*2*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(int)*K));

    dim3 block(16,16);
    dim3 grid((N + block.x -1)/block.x, (K + block.y -1)/block.y);
    k_count_within_radius_multi<<<grid, block>>>(d_pos, d_ctr, d_cnt, N, K, r2);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> h_cnt(K);
    CUDA_CHECK(cudaMemcpy(h_cnt.data(), d_cnt, sizeof(int)*K, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pos)); CUDA_CHECK(cudaFree(d_ctr)); CUDA_CHECK(cudaFree(d_cnt));

    py::array_t<int32_t> counts({K});
    auto CNT = counts.mutable_unchecked<1>();
    for (int k=0;k<K;++k) CNT(k)=h_cnt[k];
    return counts;
}

// ---- 바인딩 ----
PYBIND11_MODULE(sim_cuda, m) {
    m.doc() = "CUDA version of HPC core for moving points";
    m.def("update_boxes", &update_boxes_cuda,
          "CUDA: Updates positions with elastic box collisions",
          py::arg("pos"), py::arg("vel"), py::arg("box_width"), py::arg("box_height"), py::arg("dt"));
    m.def("nearest_assign", &nearest_assign_cuda,
          "CUDA: nearest center assignment",
          py::arg("pos"), py::arg("centers"), py::arg("box_size"));
    m.def("count_within_radius_multi", &count_within_radius_multi_cuda,
          "CUDA: counts of points within radius per center (overlap allowed)",
          py::arg("pos"), py::arg("centers"), py::arg("radius"));
}
