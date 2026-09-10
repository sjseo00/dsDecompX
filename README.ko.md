# dsDecompX 0.3

[English](README.md) | 한국어

dsDecompX는 여러 CPU 또는 GPU가 시뮬레이션 공간을 나누어 처리할 때
사용하는 domain decomposition 및 particle communication 라이브러리입니다.
MD 편의 field는 유지하면서 DEM, SPH, PIC, Lagrangian particle tracking에
필요한 사용자 정의 field도 지원합니다. 일반 사용자는 `<dsdecompx.hpp>`
하나만 include하면 됩니다.

핵심 기능은 다음과 같습니다.

- 새 owner로 직접 보내는 particle migration
- 26-neighbor halo exchange
- 사용자 정의 scalar 및 고정 길이 vector field
- 범용 reverse ghost-contribution reduction

## 빠른 시작

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /desired/prefix
```

```cmake
find_package(dsDecompX 0.3 REQUIRED CONFIG)
target_link_libraries(my_particles PRIVATE dsdecompx::dsdecompx)
```

```cpp
dsdecompx::Config config;
config.domain.upper = {100.0, 100.0, 100.0};
config.halo.width = cutoff + skin;
config.execution = dsdecompx::ExecutionMode::Host; // 또는 Cuda

dsdecompx::Decomp decomp(MPI_COMM_WORLD, config);
dsdecompx::ParticleInfo particles(1000);

decomp.migrate(particles);
decomp.exchange_halo(particles);

// 여기서 particle interaction을 계산합니다. 등록한 field는 owned와
// ghost particle 모두에서 사용할 수 있습니다.

decomp.reverse_halo(particles);
```

- `migrate()`는 자기 구역을 벗어난 입자를 새 담당 rank/GPU로 보냅니다.
- `exchange_halo()`는 구역 경계에서 힘 계산에 필요한 이웃 Particle 정보를 주고받습니다.
- `reverse_halo()`는 이웃 입자를 이용해 계산한 힘을 원래 입자에 돌려줍니다.

CUDA 빌드는 CUDA stream 인자 하나만 추가되고 함수 이름은 동일합니다.
CPU와 GPU는 같은 옵션을 지원합니다.

## 자세한 문서

- [설치 및 빌드 옵션](docs/user-guide/installation.md)
- [아키텍처](docs/architecture.md)
- [API 레퍼런스](docs/api-reference.md) — `ParticleInfo`/`Decomp`/`Config` 전체 함수, 코드 예제, 현재 제한사항
- [사용자 정의 field 안내](docs/user-guide/custom-fields.md)
- [Backend 선택](docs/user-guide/backend-selection.md), [MPI](docs/user-guide/mpi.md), [CUDA](docs/user-guide/cuda.md), [NCCL](docs/user-guide/nccl.md), [NVSHMEM](docs/user-guide/nvshmem.md)
- [성능 측정](docs/user-guide/performance.md)
- [문제 해결](docs/user-guide/troubleshooting.md)
- [Benchmark](examples/benchmark/README.md)

GPL-3.0-only; [LICENSE](LICENSE)를 참고하십시오.
