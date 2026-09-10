# dsDecomp 0.3

[English](README.md) | 한국어

dsDecomp는 여러 CPU 또는 GPU가 시뮬레이션 공간을 나누어 처리할 때
사용하는 domain decomposition 및 particle communication 라이브러리입니다.
MD 편의 field는 유지하면서 DEM, SPH, PIC, Lagrangian particle tracking에
필요한 사용자 정의 field도 지원합니다. 일반 사용자는 `<runtime.hpp>`
하나만 include하면 됩니다.

핵심 기능은 다음과 같습니다.

- 새 owner로 직접 보내는 particle migration
- 26-neighbor halo exchange
- 사용자 정의 scalar 및 고정 길이 vector field
- 범용 reverse ghost-contribution reduction

## 빌드

CPU 빌드:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix /desired/prefix
```

CUDA 빌드:

```sh
cmake -S . -B build -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA 빌드에서는 설치된 NCCL과 NVSHMEM을 자동으로 찾아 사용할 수 있게 만듭니다.

dsDecomp만 사용해 CPU/GPU 및 strong/weak scaling을 측정하는 코드는 [examples/benchmark/README.md](examples/benchmark/README.md)를 참고하십시오.

```cmake
find_package(dsDecomp 0.3 REQUIRED CONFIG)
target_link_libraries(my_particles PRIVATE dsdecomp::dsdecomp)
```

## 가장 기본적인 사용 순서

```cpp
dsdecomp::Config config;
config.domain.upper = {100.0, 100.0, 100.0};
config.halo.width = cutoff + skin;
config.execution = dsdecomp::ExecutionMode::Host; // 또는 Cuda

dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);
constexpr std::size_t capacity = 2000;
dsdecomp::ParticleInfo particles(1000);

std::vector<double> temperature(capacity);
std::vector<int> species(capacity);
particles
    .add_field<double>("temperature", temperature.data(), capacity)
    .add_field<int>("species", species.data(), capacity);

decomp.migrate(particles);
decomp.exchange_halo(particles);

// 여기서 particle interaction을 계산합니다. 등록한 field는 owned와
// ghost particle 모두에서 사용할 수 있습니다.

decomp.reverse_halo(particles);
// 또는 사용자 정의 ghost contribution을 owner에 reduction합니다.
decomp.reverse_exchange(
    particles, {{"temperature", dsdecomp::ReduceOp::Sum}});
```

- `migrate()`는 자기 구역을 벗어난 입자를 새 담당 rank/GPU로 보냅니다.
- `exchange_halo()`는 구역 경계에서 힘 계산에 필요한 이웃 Particle 정보를 주고받습니다.
- `reverse_halo()`는 이웃 입자를 이용해 계산한 힘을 원래 입자에 돌려줍니다.
- `reverse_exchange()`는 선택한 ghost field를 원래 owner field에 reduction합니다.

## `ParticleInfo`: Particle 정보를 넣는 객체

`ParticleInfo`는 라이브러리 또는 application이 소유한 particle array를
기술합니다. Host decomposition에서 본질적으로 필요한 값은 위치와 global
ID입니다. 속도, 힘, 질량, 전하, type은 MD 편의 field로 유지되며 그 밖의
application 데이터는 `add_field()`로 등록합니다.

### Particle 정보(CPU 메모리)를 새로 만들기

```cpp
dsdecomp::ParticleInfo particles(1000);

particles.x[0] = 1.0;
particles.y[0] = 2.0;
particles.z[0] = 3.0;
particles.mass[0] = 1.0;
particles.global_id[0] = 0;
```

`ParticleInfo(1000)`은 입자 1000개의 Particle 정보(CPU 메모리)를 직접 만듭니다.

### Particle 정보(CUDA 메모리)를 새로 만들기

```cpp
auto particles = dsdecomp::ParticleInfo::allocate(
    dsdecomp::MemorySpace::CudaDevice, 1000);

double* d_x = particles.x_data();
double* d_y = particles.y_data();
double* d_z = particles.z_data();
```

`ParticleInfo::allocate()`는 지정한 메모리에 Particle 정보를 만들고 객체가 없어질 때 자동으로 해제합니다.

### 프로그램에 이미 있는 Particle 정보(CPU/CUDA 메모리) 사용하기

```cpp
auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::CudaDevice,
    capacity, owned_count, ghost_count);

particles
    .positions(d_x, d_y, d_z)
    .velocities(d_vx, d_vy, d_vz)
    .forces(d_fx, d_fy, d_fz)
    .properties(d_mass, d_charge, d_type, d_global_id);
```

이 방법은 프로그램이 이미 만든 Particle 정보(CPU/CUDA 메모리)의 주소를 `ParticleInfo`에 알려줍니다. 해당 메모리는 만든 프로그램이 직접 해제해야 합니다.

### 사용자 정의 field

Host application은 scalar 또는 고정 길이 vector field를 등록해 halo
exchange와 migration에서 입자와 함께 이동시킬 수 있습니다.

```cpp
particles
    .add_field<double>("temperature", temperature, capacity)
    .add_field<int>("species", species, capacity)
    .add_field<double>("angular_velocity", omega, capacity, 3);

decomp.reverse_exchange(
    particles, {{"temperature", dsdecomp::ReduceOp::Sum}});
```

기존 MD 편의 field는 그대로 유지하면서 DEM, SPH, PIC, Lagrangian particle
tracking 데이터를 추가할 수 있습니다. 저장 공간, rank 간 schema,
reduction 및 현재 CUDA 제한은
[사용자 정의 field 안내](docs/user-guide/custom-fields.md)를 참고하십시오.

### `ParticleInfo` 함수

| 함수 | 입력값 | 반환값 | 호출하면 하는 일 |
|---|---|---|---|
| `ParticleInfo()` | 없음 | 새 `ParticleInfo` 객체 | 입자가 없는 Particle 정보(CPU 메모리)를 만듭니다. |
| `ParticleInfo(count)` | `count`: 담당 입자 수 | 새 `ParticleInfo` 객체 | 입자 `count`개의 Particle 정보(CPU 메모리)를 만듭니다. |
| `allocate(memory, count)` | `memory`: 메모리 위치, `count`: 담당 입자 수 | 새 `ParticleInfo` 객체 | 지정한 CPU 또는 CUDA 메모리에 Particle 정보를 직접 만듭니다. |
| `external(memory, capacity, owned, ghosts)` | `memory`: 메모리 위치, `capacity`: 저장 가능 수, `owned`: 담당 입자 수, `ghosts`: 받은 입자 수 | 새 `ParticleInfo` 객체 | 프로그램이 이미 가진 Particle 정보(CPU/CUDA 메모리)를 등록할 준비를 합니다. |
| `positions(x, y, z)` | x/y/z 위치가 저장된 메모리 주소 | 현재 `ParticleInfo&` | 사용할 위치 정보를 등록합니다. 연속해서 다른 등록 함수를 호출할 수 있습니다. |
| `velocities(vx, vy, vz)` | vx/vy/vz 속도가 저장된 메모리 주소 | 현재 `ParticleInfo&` | 사용할 속도 정보를 등록합니다. |
| `forces(fx, fy, fz)` | fx/fy/fz 힘이 저장된 메모리 주소 | 현재 `ParticleInfo&` | 사용할 힘 정보를 등록합니다. |
| `properties(mass, charge, type, global_id)` | 질량/전하/타입/ID가 저장된 메모리 주소 | 현재 `ParticleInfo&` | 사용할 입자 속성 정보를 등록합니다. |
| `add_field<T>(name, data, capacity, components)` | field 이름, 저장 공간, 입자 capacity, 선택적 component 수 | 현재 `ParticleInfo&` | halo exchange와 migration에 포함할 사용자 정의 field를 등록합니다. |
| `resize(count)` | `count`: 새 전체 입자 수 | 없음 | 직접 만든 Particle 정보(CPU 메모리)의 크기를 바꿉니다. |
| `resize_owned(count)` | `count`: 새 담당 입자 수 | 없음 | 담당 입자 수를 바꾸고 필요한 경우 Particle 정보 메모리를 늘립니다. |
| `reset_counts(owned, ghosts)` | `owned`: 담당 입자 수, `ghosts`: 받은 입자 수 | 없음 | 외부 Particle 정보에 현재 몇 개의 입자가 들어 있는지 갱신합니다. |
| `owned_count()` | 없음 | 담당 입자 수 | 현재 rank가 담당하는 입자 수를 반환합니다. |
| `ghost_count()` | 없음 | ghost 입자 수 | 이웃 rank에서 받은 입자 수를 반환합니다. |
| `size()` | 없음 | 전체 사용 중인 입자 수 | 담당 입자 수와 ghost 입자 수의 합을 반환합니다. |
| `capacity()` | 없음 | 저장 가능한 최대 입자 수 | 현재 메모리에 최대 몇 개의 입자를 담을 수 있는지 반환합니다. |
| `memory_space()` | 없음 | `MemorySpace` | Particle 정보가 CPU, CUDA device, CUDA managed 중 어디에 있는지 반환합니다. |
| `uses_external_memory()` | 없음 | `bool` | 프로그램이 따로 만든 메모리를 사용 중이면 `true`를 반환합니다. |
| `x_data()` ~ `global_id_data()` | 없음 | 해당 정보의 메모리 주소 | CUDA kernel이나 외부 함수에 넘길 Particle 정보의 주소를 반환합니다. |

직접 만든 Particle 정보(CPU 메모리)는 `particles.x`, `particles.vx`, `particles.fx`, `particles.mass`, `particles.type`, `particles.global_id`처럼 접근합니다. `y/z`, `vy/vz`, `fy/fz`, `charge`도 같은 방식입니다.

## `Decomp`: 공간 분할과 입자 통신

### 생성과 상태 확인

| 함수 | 입력값 | 반환값 | 호출하면 하는 일 |
|---|---|---|---|
| `Decomp(communicator, config)` | `communicator`: 사용할 MPI communicator, `config`: 공간과 실행 설정 | 새 `Decomp` 객체 | MPI rank에 공간을 나누어 배정하고 입자 통신을 준비합니다. |
| `topology()` | 없음 | `const Topology&` | 내 rank 번호, 내 구역, 이웃 rank 정보를 반환합니다. |
| `backend_kind()` | 없음 | `BackendKind` | 실제 선택된 통신 방식이 MPI, NCCL, NVSHMEM 중 무엇인지 반환합니다. |
| `owner_rank(position)` | `position`: 입자의 `{x,y,z}` 위치 | 담당 rank 번호 | 해당 위치를 담당하는 rank를 찾습니다. 처음 입자를 rank별로 나눌 때 사용합니다. |
| `any_rank_true(local_flag)` | `local_flag`: 현재 rank의 `bool` 값 | 모든 rank에 동일한 `bool` 값 | rank 중 하나라도 `true`를 전달했는지 확인합니다. 모든 rank가 함께 호출해야 합니다. |
| `has_out_of_domain_particles(particles)` | `particles`: 확인할 Particle 정보 | 모든 rank에 동일한 `bool` 값 | 어느 rank에서든 자기 구역을 벗어난 담당 입자가 있는지 확인합니다. |

```cpp
dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);
int my_rank = decomp.topology().rank();
int particle_rank = decomp.owner_rank({x, y, z});
bool need_migration = decomp.has_out_of_domain_particles(particles);
```

### 구역과 경계 거리 변경

| 함수 | 입력값 | 반환값 | 호출하면 하는 일 |
|---|---|---|---|
| `update_domain(domain)` | `domain`: 바뀐 전체 공간의 범위와 주기 경계 정보 | 없음 | 전체 공간과 각 rank의 담당 구역을 다시 계산합니다. |
| `set_halo_width(width)` | `width`: 경계에서 확인할 거리 | 없음 | 이 거리 안에 있는 Particle 정보를 `exchange_halo()`가 이웃과 주고받도록 설정합니다. |
| `halo_width()` | 없음 | 현재 경계 거리 | 현재 설정값을 반환합니다. |
| `invalidate_halo()` | 없음 | 없음 | 다음 `exchange_halo()`가 경계 입자를 처음부터 다시 찾게 합니다. |

```cpp
domain.upper = {new_lx, new_ly, new_lz};
decomp.update_domain(domain);
decomp.set_halo_width(cutoff + skin);
```

### `migrate()`

```cpp
auto result = decomp.migrate(particles, stream);
```

| 입력값 | 반환값 |
|---|---|
| `particles`: 이동시킬 Particle 정보(CPU/CUDA 메모리), `stream`: CUDA stream(CPU는 `nullptr`) | `MigrationResult`: 몇 개의 입자를 보내고 받았는지 나타내는 결과 |

호출하면 다음 행동을 합니다.

1. 내 구역을 벗어난 입자를 찾습니다.
2. 각 입자의 새 담당 rank/GPU를 찾습니다.
3. 벗어난 입자를 새 담당 rank/GPU로 보냅니다.
4. 다른 rank/GPU에서 내 구역으로 들어온 입자를 받습니다.
5. 주기 경계를 넘어간 입자의 위치를 반대편에 맞게 조정합니다.
6. 이전에 받은 ghost 입자를 제거합니다.

영역을 벗어난 입자가 없으면 이동하는 입자는 없고 `ownership_changed`는 `false`입니다.

| 결과값 | 의미 |
|---|---|
| `ownership_changed` | 담당 rank가 바뀐 입자가 있었는지 알려줍니다. |
| `previous_owned_count` | 호출 전 담당 입자 수입니다. |
| `retained_count` | 이동하지 않고 내 rank에 남은 입자 수입니다. |
| `sent_count` | 다른 rank로 보낸 입자 수입니다. |
| `received_count` | 다른 rank에서 받은 입자 수입니다. |
| `new_owned_count` | 호출 후 내 rank가 담당하는 입자 수입니다. |

CUDA의 기본 `Auto`는 NVSHMEM, NCCL, CUDA-aware MPI, 일반 MPI 순으로 가능한 방법을 선택합니다. NVSHMEM, NCCL, CUDA-aware MPI는 Particle 정보를 GPU 메모리에서 직접 전송합니다. 일반 MPI는 허용된 경우 CPU pinned memory를 거쳐 전송합니다.

### `exchange_halo()`

```cpp
auto result = decomp.exchange_halo(
    particles,
    dsdecomp::HaloExchangeOptions{rebuild, stream});
```

| 입력값 | 반환값 |
|---|---|
| `particles`: 경계 정보를 주고받을 Particle 정보(CPU/CUDA 메모리), `options`: 다시 찾을지와 CUDA stream | `HaloResult`: 받은 입자 수와 메모리 변경 여부 |

호출하면 다음 행동을 합니다.

1. 내 구역의 경계 가까이에 있는 입자를 찾습니다.
2. 그 입자의 built-in 데이터와 등록된 사용자 field를 이웃 rank/GPU로 보냅니다.
3. 이웃 구역의 경계 Particle 정보를 받습니다.
4. 받은 Particle 정보를 `particles`의 ghost 영역에 추가합니다.
5. 내 입자와 이웃 구역 입자 사이의 힘을 계산할 수 있는 상태로 만듭니다.

`migrate()`는 입자의 담당 rank를 바꾸지만 `exchange_halo()`는 담당 rank를 바꾸지 않습니다.

| 옵션 | 의미 |
|---|---|
| `rebuild` | `true`이면 경계 가까이에 있는 입자를 처음부터 다시 찾습니다. |
| `stream` | CUDA 작업을 실행할 stream입니다. CPU에서는 `nullptr`입니다. |

| 결과값 | 의미 |
|---|---|
| `ghost_count` | 받은 ghost 입자 수입니다. |
| `layout_changed` | ghost 입자 수나 순서가 이전 호출과 달라졌는지 알려줍니다. |
| `storage_reallocated` | Particle 정보를 담을 메모리가 부족해 더 큰 메모리를 만들었는지 알려줍니다. |

#### 구현 위치(함수명)

`exchange_halo()` 호출은 아래 함수 체인으로 처리됩니다.

- 사용자 API 진입: `dsdecomp::Decomp::exchange_halo` (`src/runtime.cpp`)
- 코어 시작/완료: `dsdecomp::Decomposition::begin_halo_exchange`, `dsdecomp::Decomposition::end_halo_exchange` (`src/halo.cpp`)
- Host 경계 선별/패킹: `dsdecomp::Decomposition::rebuild_halo_plan`, `dsdecomp::Decomposition::refresh_halo_buffers` (`src/halo.cpp`)
- Host 이웃 레코드 교환: `dsdecomp::detail::MpiBackend::begin_exchange_records`, `dsdecomp::detail::MpiBackend::end_exchange_records` (`src/backends/mpi_backend.cpp`)
- Device 경로 시작/완료: `dsdecomp::detail::Backend::begin_device_exchange`, `dsdecomp::detail::Backend::end_device_exchange` 구현체 (`src/backends/*`)

backend별 `exchange_halo` 데이터 경로 함수는 다음과 같습니다.

- MPI(Host): `MpiBackend::begin_exchange_records` / `MpiBackend::end_exchange_records`
- CUDA-aware MPI 또는 staging MPI(Device): `MpiBackend::begin_device_exchange` / `MpiBackend::end_device_exchange`
- NCCL(Device): `NcclBackend::begin_device_exchange` / `NcclBackend::end_device_exchange` (`src/backends/nccl/nccl_backend.cpp`)
- NVSHMEM(Device): `NvshmemBackend::begin_device_exchange` / `NvshmemBackend::end_device_exchange` (`src/backends/nvshmem/nvshmem_backend.cpp`)

`migrate()`와 `reverse_halo()`의 핵심 구현 함수명은 아래와 같습니다.

- `migrate()`: `dsdecomp::Decomp::migrate` (`src/runtime.cpp`), `dsdecomp::Decomposition::migrate_particles` (`src/migration.cpp`)
- `reverse_halo()`: `dsdecomp::Decomp::reverse_halo` (`src/runtime.cpp`), `dsdecomp::Decomposition::reverse_halo` (`src/halo.cpp`)
- Host force 환원 경로: `dsdecomp::Decomposition::reverse_exchange` (`src/halo.cpp`)

### `reverse_halo()`

```cpp
decomp.reverse_halo(particles, stream);
```

| 입력값 | 반환값 |
|---|---|
| `particles`: 계산된 힘이 들어 있는 Particle 정보(CPU/CUDA 메모리), `stream`: CUDA stream(CPU는 `nullptr`) | 없음 |

호출하면 다음 행동을 합니다.

1. ghost 입자를 이용해 계산한 `fx`, `fy`, `fz`를 모읍니다.
2. 그 힘을 원래 입자를 담당하는 rank/GPU로 돌려보냅니다.
3. 받은 힘을 원래 입자의 `fx`, `fy`, `fz`에 더합니다.

CUDA에서는 NCCL 또는 CUDA-aware MPI를 사용하면 힘 데이터를 GPU 메모리에서 직접 보냅니다. 필요한 경우 일반 MPI와 CPU pinned-memory staging을 사용할 수 있습니다.

현재 NVSHMEM backend에서는 `reverse_halo()`를 지원하지 않습니다. `Cuda + Auto`에서 NVSHMEM이 선택된 상태로 이 함수를 호출하면 실패합니다.

### `reverse_exchange()`

```cpp
decomp.reverse_exchange(
    particles,
    {{"density_contribution", dsdecomp::ReduceOp::Sum},
     {"contact_temperature", dsdecomp::ReduceOp::Max}});
```

`reverse_exchange()`는 ghost particle에서 계산한 값을 원래 particle의
owner rank로 돌려보낸 뒤 지정된 reduction을 적용합니다.

| Reduction | Owner 쪽 결과 |
|---|---|
| `ReduceOp::Sum` | 모든 ghost contribution을 더합니다. |
| `ReduceOp::Min` | owner와 ghost 값 중 최솟값을 유지합니다. |
| `ReduceOp::Max` | owner와 ghost 값 중 최댓값을 유지합니다. |
| `ReduceOp::Replace` | 결정적인 receive 순서에 따라 owner 값을 교체합니다. |

대상 field는 먼저 `add_field()`로 등록해야 합니다. 모든 MPI rank에서 field
이름, scalar type, component 수와 등록 순서가 같아야 하며 dsDecomp가 이를
collective하게 검증합니다. 등록된 메모리는 사용하는 동안 유효해야 하고
`owned_count + ghost_count`만큼의 capacity를 가져야 합니다.

현재 사용자 정의 field 통신과 범용 reverse reduction은 host memory 및 MPI
host data plane을 사용합니다. Built-in particle field의 CUDA-aware MPI,
NCCL, NVSHMEM 경로는 그대로 유지됩니다. CUDA custom field는 조용히
누락되지 않고 명시적인 오류를 발생시킵니다.

## CUDA에서 호출하는 방법

CPU와 함수 이름은 같고 CUDA stream만 추가합니다.

```cpp
config.execution = dsdecomp::ExecutionMode::Cuda;
dsdecomp::Decomp decomp(MPI_COMM_WORLD, config);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {true, stream});
decomp.reverse_halo(particles, stream);
```

CUDA 빌드의 기본 통신 설정은 `Auto`입니다. 일반 사용자는 NCCL, NVSHMEM, MPI를 직접 선택할 필요가 없습니다. 실제 선택 결과는 `decomp.backend_kind()`로 확인합니다.

## `Config`: 실행 설정

| 설정 | 값을 넣으면 바뀌는 동작 |
|---|---|
| `domain.lower`, `domain.upper` | 전체 시뮬레이션 공간의 시작과 끝을 정합니다. |
| `domain.periodic` | x, y, z 방향별로 주기 경계를 사용할지 정합니다. |
| `process_grid.dimensions` | rank를 x, y, z 방향으로 몇 개씩 배치할지 정합니다. `{0,0,0}`이면 자동으로 정합니다. |
| `process_grid.allow_rank_reorder` | MPI가 rank 배치 순서를 바꿀 수 있게 할지 정합니다. |
| `halo.width` | 경계에서 이 거리 안에 있는 Particle 정보를 이웃과 주고받습니다. |
| `halo.include_edges_and_corners` | `true`이면 면뿐 아니라 모서리와 꼭짓점 방향의 이웃과도 주고받습니다. |
| `grid_policy` | rank 배치를 직접 정할지, 공간 크기에 맞출지, 노드 연결을 고려할지 정합니다. |
| `topology_info` | `TopologyAware` 배치에 사용할 노드 정보를 넣습니다. |
| `execution` | Particle 정보(CPU 메모리)를 사용하면 `Host`, Particle 정보(CUDA 메모리)를 사용하면 `Cuda`로 설정합니다. |

CUDA를 포함해 빌드한 라이브러리는 `Cuda`, CPU 전용 라이브러리는 `Host`가 기본값입니다.

### 전문가용 backend 설정

일반 사용자는 아래 값을 바꾸지 않아도 됩니다.

| 설정 | 값을 넣으면 바뀌는 동작 |
|---|---|
| `advanced.selection` | `Auto` 대신 MPI, NCCL 또는 NVSHMEM을 강제로 사용합니다. 초기화할 수 없으면 다른 방식으로 바꾸지 않고 오류를 냅니다. |
| `advanced.read_environment` | `BACKEND` 환경변수를 읽을지 정합니다. |
| `advanced.allow_host_staging` | CUDA-aware MPI를 쓸 수 없을 때 GPU 데이터를 CPU pinned memory로 옮겨 통신할 수 있게 합니다. |

## `Topology`: 현재 rank의 배치 정보

```cpp
const auto& topology = decomp.topology();
```

| 함수 | 입력값 | 반환값 | 의미 |
|---|---|---|---|
| `rank()` | 없음 | `int` | 현재 rank 번호입니다. |
| `size()` | 없음 | `int` | 전체 rank 수입니다. |
| `dimensions()` | 없음 | `{x방향 수, y방향 수, z방향 수}` | rank가 각 방향으로 몇 개씩 배치됐는지 알려줍니다. |
| `coordinates()` | 없음 | `{x위치, y위치, z위치}` | 현재 rank가 rank 배치에서 어느 위치인지 알려줍니다. |
| `local_domain()` | 없음 | 현재 rank 구역의 시작과 끝 | 현재 rank가 담당하는 공간을 알려줍니다. |
| `neighbors()` | 없음 | 이웃 `Neighbor` 목록 | 현재 rank와 경계를 맞대는 이웃 rank를 알려줍니다. |

```cpp
auto local_box = topology.local_domain();
for (const auto& neighbor : topology.neighbors()) {
    int neighbor_rank = neighbor.rank;
}
```

## `choose_process_grid()`

```cpp
auto grid = dsdecomp::choose_process_grid(
    mpi_size, domain_shape, topology_info,
    dsdecomp::GridPolicy::Balanced);
```

`choose_process_grid()`는 MPI rank를 x, y, z 방향으로 몇 개씩 배치할지 계산해서 `GridShape`으로 반환합니다. `Decomp`를 만들기 전에 예상 배치를 확인할 때 사용합니다.

| 입력값 | 반환값 |
|---|---|
| `nranks`: 전체 rank 수, `domain`: 공간 크기, `topology`: 노드 배치 정보, `policy`: 계산 방법, `manual`: 직접 지정한 배치 | `GridShape`: x/y/z 방향의 rank 수인 `px`, `py`, `pz` |

- `Manual`: 사용자가 지정한 `px`, `py`, `pz`를 사용합니다.
- `Balanced`: 공간의 x, y, z 길이에 맞춰 구역 크기가 비슷해지도록 정합니다.
- `TopologyAware`: 노드 안 통신과 노드 사이 통신 비용까지 고려합니다.

## 성능 측정 함수

```cpp
decomp.set_performance_counters_enabled(true, true);
auto local = decomp.get_stats();
auto global = decomp.get_global_stats();
decomp.reset_stats();
```

| 함수 | 입력값 | 반환값 | 호출하면 하는 일 |
|---|---|---|---|
| `set_performance_counters_enabled(enabled, neighbor_detail)` | `enabled`: 기록 여부, `neighbor_detail`: 이웃별 기록 여부 | 없음 | 시간과 송수신량 기록을 켜거나 끕니다. |
| `get_stats()` | 없음 | 현재 rank의 `PerfStats` | 현재 rank의 호출 횟수, 시간, 송수신 byte를 반환합니다. |
| `get_global_stats()` | 없음 | 모든 rank를 합친 `PerfStats` | 모든 rank의 기록을 합칩니다. 모든 rank가 함께 호출해야 합니다. |
| `reset_stats()` | 없음 | 없음 | 지금까지 기록한 성능 값을 0으로 초기화합니다. |

`PerfStats`에는 halo/migration 호출 횟수, 송수신 byte, Particle 정보 메모리 확장 횟수, 보내기 전 데이터 준비 시간, 통신 시간, 대기 시간, 받은 데이터를 Particle 정보에 넣는 시간이 들어 있습니다.

## 기타 공개 함수

| 함수/API | 입력값 | 반환값 | 호출하면 하는 일 |
|---|---|---|---|
| `parse_backend_selection(text)` | `text`: `"auto"`, `"mpi"`, `"nccl"`, `"nvshmem"` 중 하나 | `BackendSelection` | 문자열을 backend 선택값으로 바꿉니다. |
| `backend_selection_name(value)` | `value`: `BackendSelection` | backend 이름 문자열 | backend 선택값을 문자열로 바꿉니다. |
| `backend_compiled(value)` | `value`: 확인할 `BackendSelection` | `bool` | 해당 backend가 현재 dsDecomp 빌드에 포함돼 있는지 확인합니다. |
| `GridShape::as_array()` | 없음 | `{px, py, pz}` | x/y/z 방향 rank 수를 한 묶음으로 반환합니다. |
| `SampleAggregate::add(value)` | `value`: 추가할 측정값 | 없음 | 측정 횟수, 합계, 최솟값, 최댓값을 갱신합니다. |
| `version_string` | 없음 | 버전 문자열 | 현재 dsDecomp 버전을 제공합니다. |

이 함수들은 설정 파일 처리, 진단, 성능 비교 도구를 만드는 경우에 사용합니다.

## 연동 개발자용 저수준 API

일반 사용자는 이 부분을 사용할 필요가 없습니다. 외부 라이브러리가 실행 중에 Particle 정보(CUDA 메모리)를 자동으로 늘리고 메모리 주소를 바꾸는 경우에만 사용합니다.

| API | 입력값 | 반환값 | 하는 일 |
|---|---|---|---|
| `ParticleInfo(owned, ghosts, view)` | `owned`: 담당 입자 수, `ghosts`: 받은 입자 수, `view`: Particle 정보의 메모리 주소와 크기 | 새 `ParticleInfo` 객체 | 저수준 메모리 정보를 한 번에 등록합니다. |
| `ParticleInfo(owned, ghosts, provider)` | 담당/ghost 입자 수와 메모리 관리 객체 | 새 `ParticleInfo` 객체 | Particle 정보 메모리를 자동으로 늘릴 외부 객체를 연결합니다. |
| `reset_view(view)` | `view`: 바뀐 메모리 주소와 크기 | 없음 | 외부 Particle 정보의 메모리 주소를 갱신합니다. |
| `Provider::particle_info_view(minimum)` | `minimum`: 필요한 최소 입자 수 | `ParticleView` | 현재 Particle 정보의 메모리 주소와 크기를 dsDecomp에 전달합니다. |
| `Provider::resize_particle_info(required, preserve)` | `required`: 필요한 크기, `preserve`: 유지할 입자 수 | `StorageResizeResult` | 메모리를 늘리고 새 주소와 실제 확장 여부를 전달합니다. |

## 코드 예제

아래 예제는 MPI 초기화와 `Decomp decomp(...)` 생성이 끝난 상태를 기준으로 합니다.

### CPU 예제 1: Particle 정보(CPU 메모리)를 직접 만들기

```cpp
dsdecomp::ParticleInfo particles(1000);

for (std::size_t i = 0; i < particles.owned_count(); ++i) {
    particles.x[i] = initial_x(i);
    particles.y[i] = initial_y(i);
    particles.z[i] = initial_z(i);
    particles.mass[i] = 1.0;
    particles.global_id[i] = static_cast<int>(i);
}

decomp.migrate(particles);
decomp.exchange_halo(particles);
decomp.reverse_halo(particles);
```

### CPU 예제 2: 기존 Particle 정보(CPU 메모리) 주소 전달하기

```cpp
const std::size_t owned = 1000;
const std::size_t capacity = 2000;
std::array<std::vector<double>, 11> values;
std::array<std::vector<int>, 2> integers;
for (auto& field : values) field.resize(capacity);
for (auto& field : integers) field.resize(capacity);

auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::Host, capacity, owned);
particles
    .positions(values[0].data(), values[1].data(), values[2].data())
    .velocities(values[3].data(), values[4].data(), values[5].data())
    .forces(values[6].data(), values[7].data(), values[8].data())
    .properties(values[9].data(), values[10].data(),
                integers[0].data(), integers[1].data());

decomp.migrate(particles);
decomp.exchange_halo(particles);
```

### CUDA 예제 1: Particle 정보(CUDA 메모리)를 직접 만들기

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

auto particles = dsdecomp::ParticleInfo::allocate(
    dsdecomp::MemorySpace::CudaDevice, 1000);

initialize_particles_on_gpu(particles, stream);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {false, stream});
decomp.reverse_halo(particles, stream);

cudaStreamDestroy(stream);
```

### CUDA 예제 2: 기존 Particle 정보(CUDA 메모리) 주소 전달하기

```cpp
const std::size_t owned = 1000;
const std::size_t capacity = 2000;
double* values[11]{};
int* integers[2]{};

for (auto& field : values)
    cudaMalloc(&field, capacity * sizeof(double));
for (auto& field : integers)
    cudaMalloc(&field, capacity * sizeof(int));

auto particles = dsdecomp::ParticleInfo::external(
    dsdecomp::MemorySpace::CudaDevice, capacity, owned);
particles
    .positions(values[0], values[1], values[2])
    .velocities(values[3], values[4], values[5])
    .forces(values[6], values[7], values[8])
    .properties(values[9], values[10], integers[0], integers[1]);

decomp.migrate(particles, stream);
decomp.exchange_halo(particles, {false, stream});

for (auto* field : integers) cudaFree(field);
for (auto* field : values) cudaFree(field);
```

외부 Particle 정보의 메모리 주소를 전달한 경우에는 `ParticleInfo`보다 해당 메모리를 먼저 해제하면 안 됩니다.

## 현재 제한

- 통신하는 입자 필드는 위치, 속도, 힘, 질량, 전하, 타입, ID로 고정되어 있습니다.
- 3차원 직육면체 공간을 x, y, z 방향으로 나누는 방식만 지원합니다.
- NCCL과 NVSHMEM도 처음 rank를 연결할 때 MPI가 필요합니다.
- NVSHMEM에서는 `reverse_halo()`를 지원하지 않습니다.
- 하나의 `Decomp` 객체에서 두 개의 입자 통신을 동시에 실행할 수 없습니다.

GPL-3.0-only; [LICENSE](LICENSE)를 참고하십시오.
