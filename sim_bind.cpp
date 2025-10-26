// sim_bind.cpp
// 빌드: (기본) -O3, (병렬) -DUSE_OPENMP -fopenmp

/*

    source ~/study_env/bin/activate
    # 확장자 얻기
    EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

    # (빠름) 기본
    c++ -O3 -std=c++20 -shared -fPIC \
      $(python -m pybind11 --includes) sim_bind.cpp \
      -o sim${EXT}

    # (선택) OpenMP 병렬화 켜기
    c++ -O3 -std=c++20 -shared -fPIC -DUSE_OPENMP -fopenmp \
      $(python -m pybind11 --includes) sim_bind.cpp \
      -o sim${EXT}
      python viz_centers.py
 */
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm> // for std::sort
#include <stdexcept>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#ifdef USE_OPENMP
  #include <omp.h>
  #define OMP_FOR _Pragma("omp parallel for")
  #define OMP_FOR_C2 _Pragma("omp parallel for collapse(2)")
#else
  #define OMP_FOR
  #define OMP_FOR_C2
#endif

// [수정] 위치 업데이트 및 탄성 경계 처리 (box_width, box_height 사용)
void update_boxes(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                  py::array_t<double, py::array::c_style | py::array::forcecast> vel,
                  double box_width, double box_height, double dt)
{
    if (pos.ndim()!=3 || pos.shape(2)!=2) throw std::runtime_error("pos must be [B,N,2]");
    if (vel.ndim()!=3 || vel.shape(2)!=2) throw std::runtime_error("vel must be [B,N,2]");
    if (pos.shape(0)!=1 || vel.shape(0)!=1) throw std::runtime_error("Only B=1 is supported for this simple simulation.");

    const int N = static_cast<int>(pos.shape(1));
    if (N<=0) return;

    auto P = pos.mutable_unchecked<3>(); // [1, N, 2]
    auto V = vel.mutable_unchecked<3>(); // [1, N, 2]

    OMP_FOR
    for (int n = 0; n < N; ++n) {
        // 위치 업데이트
        P(0, n, 0) += V(0, n, 0) * dt;
        P(0, n, 1) += V(0, n, 1) * dt;

        // 경계 조건 (탄성 충돌: 0.0과 box_width/box_height 사이)
        // X축 처리 (box_width 사용)
        if (P(0, n, 0) < 0.0) {
            P(0, n, 0) = -P(0, n, 0);        // 위치를 경계 안으로 보정
            V(0, n, 0) = -V(0, n, 0);        // 속도 반전
        } else if (P(0, n, 0) > box_width) {
            P(0, n, 0) = 2.0 * box_width - P(0, n, 0); // 위치를 경계 안으로 보정
            V(0, n, 0) = -V(0, n, 0);        // 속도 반전
        }

        // Y축 처리 (box_height 사용)
        if (P(0, n, 1) < 0.0) {
            P(0, n, 1) = -P(0, n, 1);
            V(0, n, 1) = -V(0, n, 1);
        } else if (P(0, n, 1) > box_height) {
            P(0, n, 1) = 2.0 * box_height - P(0, n, 1);
            V(0, n, 1) = -V(0, n, 1);
        }
    }
}

// [기존] 최근접 중심 배정: pos[N,2], centers[K,2]  → (assign[N], counts[K])
py::tuple nearest_assign(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                         py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                         double /*box_size_unused*/)
{
    if (pos.ndim()!=2 || pos.shape(1)!=2)  throw std::runtime_error("pos must be [N,2]");
    if (centers.ndim()!=2 || centers.shape(1)!=2) throw std::runtime_error("centers must be [K,2]");

    const int N = static_cast<int>(pos.shape(0));
    const int K = static_cast<int>(centers.shape(0));
    if (N<=0 || K<=0) throw std::runtime_error("N,K must be >0");

    auto P  = pos.unchecked<2>();
    auto C  = centers.unchecked<2>();

    py::array_t<int32_t> assign({N});
    auto A = assign.mutable_unchecked<1>();
    // 1) 최근접 중심 인덱스 계산
    OMP_FOR
    for (int n=0; n<N; ++n) {
        double x = P(n,0), y = P(n,1);
        double best = std::numeric_limits<double>::infinity();
        int bestk = 0;
        for (int k=0; k<K; ++k) {
            double dx = x - C(k,0);
            double dy = y - C(k,1);
            double d2 = dx*dx + dy*dy;
            if (d2 < best) { best = d2; bestk = k; }
        }
        A(n) = bestk;
    }
    // 2) 카운트
    py::array_t<int32_t> counts({K});
    auto CNT = counts.mutable_unchecked<1>();
    for (int k=0;k<K;++k) CNT(k)=0;
    for (int n=0;n<N;++n) ++CNT(A(n));

    return py::make_tuple(assign, counts);
}

// [기존] K-Nearest 중심점 인덱스 계산: pos[N,2], centers[K,2] → assign[N, K_VAL]
py::array_t<int32_t> k_nearest_assign(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                                      int K_VAL = 3)
{
    if (pos.ndim()!=2 || pos.shape(1)!=2)  throw std::runtime_error("pos must be [N,2]");
    if (centers.ndim()!=2 || centers.shape(1)!=2) throw std::runtime_error("centers must be [K,2]");

    const int N = static_cast<int>(pos.shape(0));
    const int K = static_cast<int>(centers.shape(0));
    
    const int K_ASSIGN = (K_VAL > K) ? K : K_VAL; 
    
    if (N <= 0 || K <= 0 || K_ASSIGN <= 0) 
        throw std::runtime_error("N, K, and K_ASSIGN must be > 0");

    auto P  = pos.unchecked<2>();
    auto C  = centers.unchecked<2>();

    py::array_t<int32_t> assign({N, K_ASSIGN});
    auto A = assign.mutable_unchecked<2>();

    struct DistIndex {
        double dist2;
        int index;
    };
    
    OMP_FOR
    for (int n = 0; n < N; ++n) {
        double px = P(n, 0), py = P(n, 1);
        
        std::vector<DistIndex> distances;
        distances.reserve(K);

        for (int k = 0; k < K; ++k) {
            double dx = px - C(k, 0);
            double dy = py - C(k, 1);
            distances.push_back({dx * dx + dy * dy, k});
        }

        std::sort(distances.begin(), distances.end(), 
            [](const DistIndex& a, const DistIndex& b) {
                return a.dist2 < b.dist2;
            });
            
        for (int i = 0; i < K_ASSIGN; ++i) {
            A(n, i) = distances[i].index;
        }
    }

    return assign;
}


// [기존] 반경 r 이내 카운트(겹침 허용): 반환 counts[K]
py::array_t<int32_t> count_within_radius_multi(py::array_t<double, py::array::c_style | py::array::forcecast> pos,
                                               py::array_t<double, py::array::c_style | py::array::forcecast> centers,
                                               double radius)
{
    if (pos.ndim()!=2 || pos.shape(1)!=2)  throw std::runtime_error("pos must be [N,2]");
    if (centers.ndim()!=2 || centers.shape(1)!=2) throw std::runtime_error("centers must be [K,2]");
    const int N = static_cast<int>(pos.shape(0));
    const int K = static_cast<int>(centers.shape(0));
    auto P  = pos.unchecked<2>();
    auto C  = centers.unchecked<2>();
    py::array_t<int32_t> counts({K});
    auto CNT = counts.mutable_unchecked<1>();
    for (int k=0;k<K;++k) CNT(k)=0;

    const double r2 = radius*radius;

    OMP_FOR
    for (int k=0; k<K; ++k) {
        int local = 0;
        double cx=C(k,0), cy=C(k,1);
        for (int n=0; n<N; ++n) {
            double dx=P(n,0)-cx, dy=P(n,1)-cy;
            if (dx*dx + dy*dy <= r2) ++local;
        }
        CNT(k) = local;
    }
    return counts;
}

PYBIND11_MODULE(sim, m) {
    m.doc() = "C++ HPC core for moving points (pybind11)";
    // 5 C++ 인자: pos, vel, box_width, box_height, dt -> 5 py::arg()
    m.def("update_boxes", &update_boxes,
          "Updates positions and handles elastic boundary collisions (Rectangular Box).",
          py::arg("pos"), py::arg("vel"), py::arg("box_width"), py::arg("box_height"), py::arg("dt")); 

    m.def("nearest_assign", &nearest_assign,
          "Assign each point to the nearest center (Voronoi)",
          py::arg("pos"), py::arg("centers"), py::arg("box_size"));

    m.def("k_nearest_assign", &k_nearest_assign,
          "Finds the indices of the K nearest centers for each point. (K-NN Assignment)",
          py::arg("pos"), py::arg("centers"), py::arg("k_val") = 3);

    m.def("count_within_radius_multi", &count_within_radius_multi,
          "Counts of points within radius per center (overlap allowed)",
          py::arg("pos"), py::arg("centers"), py::arg("radius"));
}
