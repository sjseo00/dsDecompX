# dsDecompX 1.0

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
find_package(dsDecompX 1.0 REQUIRED CONFIG)
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

- [시작하기](docs/00_Getting_Started.md) — 초보자용 단계별 안내서: 빌드하기, 뼈대 코드 작성, 옵션 채우기, 메인 루프 작성, 실행까지 순서대로 설명
- [개요 및 아키텍처](docs/01_Overview_and_Architecture.md) — dsDecompX가 무엇을 하는지, 공간을 어떻게 나누는지, dsDecompX와 응용 프로그램의 역할 분담, 현재 제한사항
- [설치 및 backend 선택](docs/02_Installation_and_Backend_Selection.md) — 빌드 옵션, Auto backend 선택 동작 방식, 문제 해결
- [통신 backend](docs/03_Communication_Backends.md) — MPI/NCCL/NVSHMEM이 각각 입자 데이터를 옮기는 방식, CUDA stream/event 모델, 동기화 규칙, backend 추가 방법
- [API 레퍼런스](docs/04_API_Reference.md) — `ParticleInfo`/`Decomp`/`Config` 전체 함수, 사용자 정의 field, 코드 예제, 현재 제한사항
- [부하 분산 및 성능](docs/05_Load_Balancing_and_Performance.md) — 동적/수동 재분배와 성능 측정
- [MD 통합 안내](docs/06_MD_Integration_Guide.md) — 통합 패턴과 DeepScaleMD 실제 적용 사례

GPL-3.0-only; [LICENSE](LICENSE)를 참고하십시오.
