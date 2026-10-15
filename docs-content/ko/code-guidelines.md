# 코드 수정 가이드라인

## 파일 크기와 모듈 구조

파일을 하나의 단위로 유지해야 하는 경우, helpers 파일을 분리해야 하는 경우, 디렉터리 모듈로 나눠야 하는 경우에 대한 지침입니다. 이 섹션은 수치 기준의 공식 출처입니다. `.claude/skills/` 아래의 에이전트용 도구를 포함해 이 수치를 인용하는 다른 모든 곳은 수치를 다시 적지 말고 이곳을 링크해야 하며, 그래야 수치가 서로 어긋나지 않습니다.

독립적이고 작은 기능은 단일 파일로 두어도 좋습니다. 예를 들어 `src/models/dbrx.rs`(약 600줄로 완결된 모델)나 `src/distributed/heartbeat.rs`(약 310줄로 완결된 기능)가 있습니다. 그보다 큰 경우에는 다음 기준을 적용합니다.

| 크기 | 조치 |
|------|--------|
| 800줄 미만 | 단일 파일로 충분함 |
| 800~1,200줄 | 복잡한 모델에는 허용됨. helpers 파일 분리를 고려 |
| 1,200줄 이상 | helpers를 `<name>_helpers.rs`로 분리 |
| 1,500줄 이상 | 디렉터리 모듈로 분할하는 것을 적극 고려 |
| 2,000줄 이상이면서 helpers 파일 없음 | 안티패턴. `<name>_helpers.rs`로 분리 |

이는 지침이지 관문(gate)이 아닙니다. 숫자보다 파일이 기준을 넘어도 되는 이유가 더 중요합니다.

- `src/models/nemotron_h.rs`(약 2,900줄, helpers 파일 없음)는 문서화된 예외입니다. 하이브리드 Mamba 및 Transformer 모델이며, 이를 분할하면 하나의 모델로 성립시키는 레이어 인터리빙 로직이 깨집니다.
- `src/models/llama4.rs`(약 1,860줄)는 1,200줄 이상 행을 적용한 사례입니다. MoE, iGQA, ChunkedKV가 긴밀하게 결합되어 있어 모델 자체는 하나의 파일로 유지했고, 분리 가능했던 마스크 생성과 가중치 로딩은 `src/models/llama4_helpers.rs`로 옮겼습니다. `src/models/` 아래에서 이와 같이 분리된 다른 사례는 `gemma3n_helpers.rs`와 `qwen3_next_helpers.rs`뿐입니다.

여러 파일이 정당한 이유가 기록되지 않은 채 이 기준을 크게 넘고 있습니다. `src/models/gemma4.rs`(약 6,700줄)와 `src/models/qwen3_5.rs`(약 3,500줄)에는 helpers 파일이 없고, `src/models/gemma3n.rs`는 약 5,600줄입니다. 이들은 선례가 아니라 부채로 받아들여야 합니다. 트리에서 가장 큰 파일이 프로젝트가 허용하는 범위를 보여 주는 증거는 아니며, 리뷰에서 인용할 수 있는 수치도 아닙니다.

모델을 수정하면서 줄 수는 계속 변하므로, 위의 모든 수치는 근사치로 취급하고 어디에 인용된 숫자(이 섹션 포함)도 그대로 믿지 말고 파일을 직접 확인하십시오.

### 인라인 테스트와 형제 `_tests.rs` 파일

테스트가 짧고 모듈과 긴밀히 결합되어 있는 동안(대략 100줄 미만)에는 `#[cfg(test)] mod tests { ... }` 안에 인라인으로 유지합니다. 모듈의 테스트가 그 이상으로 커지면, 늦어도 200줄에 이르기 전에 형제 파일로 옮깁니다(`config.rs`에는 `config_tests.rs`). 연결은 `#[cfg(test)] mod config_tests;` 또는 `#[path = "config_tests.rs"] mod config_tests;` 속성으로 합니다.

### 이름 규칙

| 패턴 | 예시 | 사용 시점 |
|---------|---------|------|
| `<name>_helpers.rs` | `gemma3n_helpers.rs` | 대형 모델에서 분리한 helpers |
| `<name>_tests.rs` | `config_tests.rs` | 인라인 테스트가 모듈보다 커졌을 때의 형제 테스트 파일 |

## 공유 함수 주석

여러 모델이 의존하는 공유/공통 함수(예: attention 구현, 정규화, KV cache, 활성화 함수)를 수정할 때는 **항상** 해당 함수를 사용하는 모델을 나타내는 주석을 추가하거나 갱신해야 합니다.

**형식:**
```rust
// Used by: Llama, Qwen, Gemma2, Gemma3
fn repeat_kv(keys: &MlxArray, values: &MlxArray, n_rep: i32) -> ... {
    ...
}
```

**이것이 중요한 이유:**
- 한 모델을 고치다가 다른 모델이 깨지는 회귀를 방지합니다
- 변경 후 어떤 모델을 다시 테스트해야 하는지 분명해집니다
- 이후 개발자가 수정의 영향 범위를 이해하는 데 도움이 됩니다

**갱신 시점:**
1. 기존 공유 함수를 사용하는 새 모델을 추가할 때
2. 공유 함수의 동작을 수정할 때
3. 목록에 없는 모델이 해당 함수를 사용하고 있음을 발견했을 때

**호출자 목록이 너무 길어 모두 나열할 수 없을 때:**

호출자가 대략 열두 개를 넘으면 손으로 적은 목록은 다음 릴리스에서 틀리게 되고 아무도 고치지 않습니다. 이 경우에는 나열하지 마십시오. 호출자가 목록에 *왜* 포함되는지를 설명하는 규칙을 적고, 그룹별로 대표 몇 개를 명시하며, 의도적으로 제외된 계열을 명시한 다음, 정확한 집합을 다시 생성하는 `grep` 한 줄 명령으로 마무리합니다.

```rust
/// Used by: decoders that materialize an explicit prefill mask instead of
/// leaving `mask: None` for fused SDPA to apply causality itself. At this
/// commit that is N non-test files under `src/models`, in four groups.
/// (groups and representatives ...)
///
/// Not used by the mainstream dense decoders (Llama3, Mixtral, Gemma2 and
/// similar): they pass `mask: None` for `seq_len > 1`, so they are unaffected
/// by changes here.
///
/// The caller set is too large to enumerate by name without going stale, so
/// the groups above are a summary. Regenerate the exact list with
/// `grep -rln '\bcreate_causal_mask(' src/models --include='*.rs' | grep -v 'tests\.rs$'`.
pub fn create_causal_mask(size: i32, offset: i32) -> UniquePtr<MlxArray> {
```

`utils.rs`의 `create_causal_mask`가 실제 사례입니다. "사용하지 않음" 부분은 목록만큼 중요합니다. 몇 번의 릴리스 전에 다른 방식으로 옮겨 간 계열을 기여자가 여전히 공유 helper가 처리해 준다고 가정하지 않도록 막아 주기 때문입니다. 모든 목록은 기억에 의존하지 말고 현재 커밋에서 grep을 실행하여 도출하며, 공개 항목에는 `//`보다 `///` 문서 주석을 사용하여 주석이 rustdoc에도 남도록 하십시오.

**추적해야 할 주요 공유 컴포넌트:**
- `src/lib/mlxcel-core/src/layers.rs` - KVCache, Attention, Normalization
- `src/lib/mlxcel-core/src/utils.rs` - create_causal_mask, softcap, repeat_kv
- `src/models/*.rs`의 모델별 attention 변형

## 브리지 helper는 입력 dtype을 반환합니다

`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`의 helper는 전달받은 dtype을 그대로 반환해야 합니다. 내부 계산은 더 넓은 dtype으로 해도 되고 실제로 그렇게 하는 helper도 여럿 있지만, 반환하기 전에 원래 dtype으로 다시 캐스팅해야 합니다.

```cpp
auto result = mlx::core::multiply(a.inner, scale);          // f32, because the constants are
return std::make_unique<MlxArray>(
    mlx::core::astype(result, a.inner.dtype()));            // hand back what came in
```

**이것이 중요한 이유:**

`mlx::core::array(0.5f)`는 f32 스칼라이므로, f16 또는 bf16 입력과의 이항 연산은 모두 승격되어 helper가 f32를 반환하게 됩니다. 결과 자체에는 잘못된 점이 없습니다. 같은 값이 더 높은 정밀도로 나올 뿐이므로 parity 검사, 출력 diff, 단위 테스트가 모두 통과합니다. 깨지는 것은 처리량이며, 그것도 다른 파일에서 나타납니다. 넓어진 activation이 첫 번째 MLP에서 residual stream에 합류하고, 이후 모든 레이어에서 hidden state가 f32로 유지되며, 각 matmul은 피연산자에 맞추기 위해 자신의 half precision 가중치를 f32로 승격합니다. Decode는 대역폭에 묶여 있으므로 모델은 대략 두 배의 가중치 바이트와 캐스트 트래픽을 읽게 됩니다.

2026-09-08에 이 문제로 인해 M1 Ultra에서 `gpt_bigcode-santacoder`는 mlx-lm 대비 28%, `pythia-1b`는 31%였고, M5 Max에서 `recurrent_gemma`는 5분의 1 수준이었습니다. 반면 `starcoder2-3b-4bit`는 GELU가 항상 다시 캐스팅하던 `compiled_gelu`를 사용했다는 단 하나의 이유로 100%였습니다.

이 규칙은 호출 지점들을 맞춰 유지하는 것이 아니라 함수 단위로 적용됩니다. helper *내부*의 승격은 비용이 들지 않습니다. `mx.compile`이 이를 융합하여 넓어진 값이 레지스터 밖으로 나가지 않기 때문이며, `compiled_gelu_topk`의 내부 스칼라를 다시 작성해도 `gemma3n-e4b-bf16`이 0.7%만 움직여 노이즈 범위 안인 이유가 이것입니다. helper 밖으로 새어 나가는 승격만 비용을 발생시킵니다. 반환값이 곧 전체 결과인 helper는 `gelu_tanh_approx`와 `relu`처럼 처음부터 입력 dtype으로 스칼라를 만들어도 됩니다. 중간 값을 하나 아낄 수 있고 보장하는 내용은 같습니다.

이를 grep으로 점검하지 마십시오. 복원은 callee 안에 있을 수 있습니다. `compiled_softcap_sdpa_gqa`는 위반처럼 보이지만 그렇지 않은데, fallback 분기가 `softcap_sdpa_graph` 안에서 복원하기 때문입니다.

**강제 방법:**

[`src/lib/mlxcel-core/src/ffi_tests.rs`](../src/lib/mlxcel-core/src/ffi_tests.rs)의 `activation_helpers_return_the_input_dtype`는 export된 각 활성화 함수를 f16, bf16, f32로 호출하여 dtype이 유지되는지 검증합니다. 새 활성화 helper는 여기에 추가하십시오.

## 하나의 체크포인트, 하나의 dtype

로드 시점의 dtype 정책은 체크포인트가 하나의 부동소수점 dtype만 갖도록 만들어야 합니다. 모델의 일부는 한 dtype으로 유지하고 나머지를 변환하는 것은 위 섹션과 같은 승격 결함이 한 계층 위에서 발생하는 것이며, 둘 중 찾기가 더 어렵습니다.

**이것이 중요한 이유:**

MLX는 f16과 bf16이 만나면 f32로 승격합니다. 따라서 일부 텐서는 bf16으로 유지하고 그 이웃은 f16으로 변환하는 정책은 두 dtype의 모든 경계에서 승격을 일으키며, 그 경계는 유지 집합이 끝나는 지점 어디든 됩니다. 어떤 helper도 잘못되지 않았고, 모든 activation은 받은 dtype을 반환하며, 위의 강제 테스트도 통과합니다.

Gemma3n이 이 상태였습니다. 정책은 타당한 이유로 언어 MLP를 bf16으로 유지했고(f16으로 강제하면 3배 나빠지는 것으로 측정되었습니다), 나머지는 모두 변환했습니다. M5 Max에서 `gemma3n-e4b-bf16`으로, 이미지가 포함된 273 token 프롬프트와 128 token 생성 조건에서 체크포인트를 일괄 bf16으로 로드했더니 decode가 38.02에서 47.48 tok/s로, prefill이 2143에서 2371로 올랐고, decode는 mlx-vlm 대비 78%에서 97%가 되었습니다.

**찾는 방법:**

결함 자체는 열거할 수 없더라도 예외가 생기는 지점은 작고 열거할 수 있으므로 거기서 시작합니다. 모든 분할은 `convert_bf16_weights_with_keep`에 전달되는 keep predicate에서 만들어지므로, 이 호출 지점들이 후보 목록 전체입니다. 안전한 predicate와 분할을 일으키는 predicate를 가르는 것은 상수인지 여부가 아니라 *무엇을* 유지하는가입니다.

- **양자화 sidecar는 안전합니다.** `src/loading/` 아래의 호출 지점 아홉 곳은 정확히 `key.ends_with(".scales") || key.ends_with(".biases")`만 유지합니다. 이들은 `quantized_matmul`에 직접 입력되며 일반 matmul 경로에서 dense 가중치와 만나지 않습니다. 4-bit Gemma3n 변형이 135%와 138%로 건강했던 반면 bf16 변형은 98%에 머물렀던 이유가 이것입니다.
- **그 외에는 dense 가중치를 분할합니다.** 그렇게 한 predicate는 `gemma3n_bf16_key`뿐이었으며, 이웃이 변환되는 동안 언어 MLP를 유지했습니다. 지금은 상수 `true`입니다.

따라서 확인할 것은 이 predicate가 일반 matmul에서 변환된 텐서와 곱해질 텐서를 유지하는가입니다. 그렇다면 논증이 아니라 측정이 필요합니다.

이 목록이 깨끗해지고 나면 결함에는 호출 지점이 없으므로 정적 감사는 아무것도 찾지 못합니다. 남는 증상은 위치로 드러납니다. 같은 머신, 같은 계열 안에서 비교하고, 형제들보다 훨씬 낮은 단일 체크포인트를 찾으십시오. M1 Ultra에서 세 Gemma3n 행은 기준 대비 138%, 135%, 98%였습니다. 37포인트 하락이 신호의 전부이며, 4-bit 변형은 변환을 아예 건너뛰므로 처음부터 균일했다는 사실을 알고 나서야 읽힙니다. 세 개의 독립적인 숫자로 읽으면 이 표는 "bf16 체크포인트가 더 느리다"라고 말하는데, 이는 특별할 것이 없는 내용이라 실제로 그냥 넘어갔습니다.

**여기에 함수 계층의 수정을 적용하지 마십시오.** 두 계층은 호출 지점에서 똑같아 보이고, 잘못된 수정은 오히려 파괴적입니다. [`src/vision/encoders/gemma3n.rs`](../src/vision/encoders/gemma3n.rs)의 `RMSNormAct2d::forward` 안에서 입력 dtype을 복원하는 것은 위 규칙에 따르면 올바르고 mlx-vlm의 `rms_norm2d`와 한 줄씩 일치하지만, tower가 여전히 f16으로 로드되던 때에는 깊은 블록을 클리핑했습니다. 184개 norm 호출 중 10개가 f16 상한인 65504를 넘는 입력을 받았고 최대 213055까지 올라갔으며, 그 결과 `gemma3n-e4b-bf16`은 파란색, 초록색, 보라색 모두에 "blue"라고 답했습니다. 감사에서 누수로 지적된 f32가 범위를 제공하던 유일한 요소였습니다. 이 복원은 로드 정책을 고친 뒤에야 안전해졌고 prefill에서 1.04배의 이득까지 얻었으며, 그 전에는 아니었습니다. 경계 복원이 모델을 깨뜨릴 때는 복원 대상 dtype을 확인해야 합니다.

**이 이득을 dtype 자체의 이득으로 일반화하지 마십시오.** 균일한 bf16이 도움이 된 것은 혼합을 제거했기 때문이지 bf16이 더 빠르기 때문이 아닙니다. 통제 실험을 해 보십시오. bf16을 강제로 유지하는 부분 집합이 없는 계열에서는 같은 하드웨어에서 두 dtype이 노이즈 범위 안에 놓이며, `qwen2.5-0.5b-bf16`은 401.8 대 397.3 tok/s, `llama-3.1-8b-bf16`은 33.07 대 33.02로 측정되었습니다. 이 통제가 없으면 결과가 모든 곳에서 변환을 줄여야 한다는 주장처럼 읽히지만, 그런 것이 아닙니다. 다른 모든 계열은 계속 f16으로 변환합니다.

위치 기반 진단에는 형제가 필요합니다. 체크포인트를 하나만 배포하는 계열에는 비교 대상이 없어 하락이 보이지 않으므로, 대신 참조 구현과 비교해야 합니다. `MLXCEL_KEEP_BF16`은 모든 체크포인트를 일괄 bf16으로 로드하며, 이 경우의 A/B 비교와 정책을 바꾸기 전에 의심되는 분할을 확인하는 데 쓰입니다. [`docs/environment-variables.md`](environment-variables.md)를 참고하십시오.

**강제 방법:**

위 섹션과 달리 자동화된 관문은 없습니다. 가장 명백한 불변 조건인 체크포인트당 부동소수점 dtype 하나는 그대로 적용하면 틀립니다. 양자화된 체크포인트는 설계상 u32 평면과 함께 bf16 scales와 biases를 유지하므로, 단순한 규칙을 단언하면 건강한 Gemma3n 변형을 포함한 모든 4-bit 모델에서 실패하게 됩니다. 중요한 불변 조건은 두 부동소수점 dtype이 *일반* matmul에서 만나지 않는다는 것이며, 이는 가중치 맵이 아니라 그래프의 속성이고 로드 시점에는 알 수 없습니다.

사용할 수 있는 것은 위의 predicate 규칙이며, 기계적으로 검사할 수 있을 만큼 단순합니다. keep predicate는 `.scales`와 `.biases`만 지정할 수 있습니다. 열 개의 호출 지점 중 아홉 곳은 검사해 보면 이를 만족하고, 나머지 하나는 이제 상수입니다. [`scripts/ci/check_kernel_dtype_keys.py`](../scripts/ci/check_kernel_dtype_keys.py)와 같은 형태의 소스 검사로 이를 강제할 수 있지만 아직 작성되지 않았습니다. 그때까지는 predicate가 바뀔 때 후보 목록을 리뷰하십시오.

## JIT 커널 캐시 키

`cuda_kernel` 실행에 전달되는 모든 `template_args` 목록은 dtype이 달라질 수 있는 각 입력의 dtype을 명시해야 합니다.

```cpp
std::vector<std::pair<std::string, TemplateArg>> template_args = {
    {"Dim", dim},
    {"KVType", k_pool.dtype()},   // keys the cache, not read by the kernel body
};
```

**이것이 중요한 이유:**

MLX는 입력의 런타임 dtype으로부터 커스텀 커널의 버퍼 매개변수 타입을 생성합니다. Metal은 이 dtype을 커널 이름에 반영합니다. CUDA는 커널 이름을 `"custom_kernel_" + name + template_arguments_hash(template_args)`로 짓는데, upstream ml-explore/mlx#4273 이전에는 `cu::get_jit_module`이 컴파일된 모듈을 정확히 그 이름으로 프로세스 전역 맵에 memoize하고 캐시 미스일 때만 소스 빌더를 호출했습니다. 현재 pin(81ba1c6a)에서 CUDA는 모듈을 그 이름과 생성된 소스의 해시로 키잉하며, ROCm overlay도 #2149 이후 같은 방식이므로 오늘날의 백엔드에서는 키가 이미 dtype을 포함합니다.

이름만을 키로 쓰고 template 인수가 정수뿐인 경우, 처음 컴파일된 dtype이 같은 geometry의 이후 모든 dtype에 제공됩니다. 그러면 그 호출은 잘못된 포인터 타입으로 버퍼를 읽어 입력과 무관한 숫자를 반환합니다. 예외는 발생하지 않으며 macOS에서는 보이지도 않습니다. 이슈 #1053과 #1054가 두 가지 증상입니다. sampler는 프로덕션에서 영향을 받았는데, `gumbel_max_sample_accepts`가 하나의 `NumSplits`에서 f32, f16, bf16을 모두 허용하기 때문입니다.

해당 항목은 커널 본문에서 참조되지 않아도 됩니다. 그 역할은 캐시 키이며 defense in depth로 유지됩니다. 다시 vendor한 ROCm fork, pin 변경, 새 백엔드가 이름만을 키로 쓰는 방식을 되돌릴 수 있습니다. 죽은 코드라고 제거하지 마십시오.

**강제 방법:**

`make verify-kernel-dtype-keys`(`make verify`와 `make verify-rocm`의 일부이며 `kernel dtype keys` CI 작업)는 [`scripts/ci/check_kernel_dtype_keys.py`](../scripts/ci/check_kernel_dtype_keys.py)를 실행합니다. ROCm의 `fast::hip_kernel`도 CUDA와 같은 방식으로 커널 이름을 짓기 때문에(`src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/custom_kernel.cpp`) HIP 실행에도 같은 규칙이 적용됩니다. 파일이 검사 범위에 들어가는 조건은 손으로 관리하는 목록이 아니라 주석 밖에서 `cuda_kernel(` 또는 `hip_kernel(`을 호출하는 것이므로, Metal 전용 launcher는 누군가 CUDA나 HIP 포트를 추가하기 전까지는 범위 밖이며, 추가되는 시점부터 검사가 자동으로 적용됩니다.

이 토큰 규칙만으로는 눈에 보이지 않으며, 리팩터링으로 비어 버릴 수 있습니다. 실행이 헤더나 공유 helper로 옮겨 가면 해당 파일의 `template_args`가 범위 밖으로 빠지고, 아무것도 검사하지 않아도 검사는 OK를 출력합니다. 그래서 범위는 fail-closed입니다(#1875). git이 추적하거나 추적하게 될 모든 C, C++, CUDA, HIP, Objective-C++ 소스와 헤더가 두 디렉터리의 `*.cpp`뿐 아니라 전부 스캔됩니다(`.gitignore`된 빌드 출력과, MLX 자체 launcher를 포함하는 로컬 `references/` 체크아웃은 제외됩니다). 출력은 스캔한 파일 수와 함께 범위에 포함된 파일 수를 보고하며, 범위에 포함된 집합은 스크립트의 `EXPECTED_IN_SCOPE`에 고정되어 있습니다. 고정된 파일이 더 이상 실행을 하지 않거나 사라지면 둘 중 어느 쪽인지 알려 주는 메시지와 함께 검사가 실패하고, pin되지 않은 새 launcher 파일이 생기거나 범위가 비어 있어도 실패합니다. 실행을 추가, 이동, 제거할 때는 같은 변경에서 pin을 갱신하십시오. 같은 target과 CI 작업에서 실행되는 [`scripts/ci/check_kernel_dtype_keys_test.sh`](../scripts/ci/check_kernel_dtype_keys_test.sh)는 임시 복사본에서 실행을 범위 밖으로 옮긴 뒤 검사기가 각 경우를 거부하는지 단언합니다. 두 가지 한계가 남아 있습니다. 이름이 붙은 `std::vector<...TemplateArg>` 초기화만 읽으므로 `template_args`를 인라인으로 전달하는 실행은 범위에는 들어가지만 검사할 것이 없습니다. 또한 파일 안에서 직접 호출하지 않고 도달하는 실행(함수 포인터나 다른 곳에 정의된 매크로)은 해당 파일을 범위에 넣지 못합니다. 이 섹션을 믿지 말고 스크립트를 읽어 보십시오.

## HashMap 순회 순서

`HashMap` 또는 `HashSet`의 순회 결과가 순서가 있는 상태가 되거나 순서에 민감한 소비자에게 전달될 때는, 순회하는 지점에서 순서를 명시적으로 확정해야 합니다.

```rust
// Wrong. `entries` comes out of a HashMap, so its order is arbitrary, and
// `sort_by_key` is stable, so any two entries tied on `last_accessed` keep
// that arbitrary order.
let mut entries: Vec<_> = self.allocations.values().collect();
entries.sort_by_key(|a| a.last_accessed);

// Right. The id component makes the key a TOTAL order, so the stable sort is
// never asked to break a tie and hash order cannot reach the result.
let mut entries: Vec<_> = self.allocations.values().collect();
entries.sort_by_key(|a| (a.last_accessed, a.sequence_id));
```

**이것이 중요한 이유:**

`RandomState`는 프로세스가 아니라 각 맵 *인스턴스*에 시드를 부여합니다. "HashMap 순회 순서는 실행마다 무작위로 정해진다"는 흔한 사고 모델은 중요한 방향에서 틀렸습니다. 이 모델은 고정된 머신의 고정된 바이너리에서는 순서가 고정된다고 암시하지만 실제로는 그렇지 않습니다. 같은 함수 안에서 몇 마이크로초 간격으로, 동일한 입력으로 만든 두 맵도 서로 다르게 순회됩니다.

이 저장소의 Linux 머신에서 `rustc -O`(1.97.1)로 측정했습니다. 한 프로세스 안에서 같은 다섯 개 키로 `HashMap<&str, u32>` 인스턴스 열 개를 만들고, 각 맵의 `.keys()` 순서를 기록한 뒤 서로 다른 순서의 수를 셉니다. 200개 프로세스에서 127개는 10개 중 10개가 모두 다른 순서였고, 62개는 9개, 11개는 8개였습니다. 열 개의 맵이 모두 일치한 프로세스는 200개 중 하나도 없었습니다. 실험 자체가 무작위이므로 정확한 분포는 측정할 때마다 달라집니다. 변하지 않는 것은 열 개가 결코 일치하지 않는다는 점입니다.

그 결과는 확률적 결함이므로 테스트가 통과했다는 것은 증거가 되지 않습니다. `cargo check`도 깨끗하고 clippy도 깨끗하며 테스트는 대부분 통과합니다.

**순서에 민감한 소비자.** 순회는 하위의 무언가가 위치를 따지는 순간에만 문제가 됩니다. 다음은 모두 위치를 따집니다.

- 라운드 로빈 인덱스 선택을 포함한 위치 기반 인덱싱.
- `min_by_key`와 `max_by_key`는 동점일 때 입력 순서로 판정하며 방향이 서로 반대입니다. `min_by_key`와 `min_by`는 **첫 번째** 최솟값을 반환하고, `max_by_key`와 `max_by`는 **마지막** 최댓값을 반환합니다. 따라서 수정은 고치려는 호출에 맞는 방향으로 동점 처리 요소를 두어야 하며, 하나의 정책의 두 분기가 서로 반대 방향을 요구할 수도 있습니다.
- 접두 구간: `take(n)`, 이른 `break`, 루프를 빠져나가는 이른 `return`.
- 루프 간에 이어지는 누산기. 예를 들어 요소마다 한 번씩 진행되는 시드.
- 현재 소비자가 그렇게 읽지 않더라도 우선순위 순서로 문서화된 모든 것.

소비자가 순회와 다른 모듈에 있는 경우가 흔하며, 이 때문에 리뷰를 통과합니다. 정렬되지 않은 목록을 반환하는 accessor는 그것만 보면 평범해 보입니다.

### 정렬이 자동으로 수정이 되는 것은 아닙니다

이 규칙에서 가장 눈에 띄지 않는 부분이면서 다시 발견하는 데 가장 큰 비용이 드는 부분입니다. `slice::sort_by_key`와 `slice::sort_by`는 **안정(stable)** 정렬입니다. 같다고 비교되는 요소는 입력 순서를 유지합니다. 입력이 `HashMap`에서 나왔다면 유지되는 순서는 해시 순서입니다. 따라서 정렬 키가 요소에 대한 **전순서(total order)**, 즉 서로 다른 두 요소가 같다고 비교될 수 없는 순서가 아니면 정렬해도 `HashMap`에서 나온 목록이 결정적이 되지 않습니다.

아래 여덟 건 중 세 건은 후보 목록을 정렬했는데도 정확히 이 이유로 비결정적이었습니다. 정렬을 본 독자는 결과가 고정되었다고 판단하기 쉬웠지만 그렇지 않았습니다.

`sort_unstable_by`도 해결책이 아닙니다. 입력 순서를 보존하지 않지만 순서를 정의하지도 않습니다. 다른 임의의 동점 순서로 대체할 뿐이며 결과는 여전히 재현되지 않습니다.

해결책은 각 요소에 고유한 값을 덧붙여 키를 전순서로 만드는 것입니다. 아래 세 형태는 모두 올바르며 선택은 취향의 문제입니다.

```rust
// Tuple key. Cheapest when the unique component is Copy.
entries.sort_by_key(|a| (a.last_accessed, a.sequence_id));

// Comparator with `then_with`. Preferred when the unique component is a
// String, since `sort_by_key` would force a clone into the tuple.
completed.sort_by(|(key_a, t_a), (key_b, t_b)| t_a.cmp(t_b).then_with(|| key_a.cmp(key_b)));

// Sort on the unique component alone, when the policy key is not needed.
nodes.sort_by(|a, b| a.config.id.cmp(&b.config.id));
```

호출 지점에 키를 전순서로 만드는 요소가 무엇인지 밝히는 주석을 추가하십시오. 그렇지 않으면 그 요소가 불필요해 보여 다음 독자가 단순화해 버립니다.

### 수정 테스트하기

이 계열의 모든 수정에는 특정 형태의 테스트가 필요했으며, 다음 세 가지 중 하나라도 놓친 테스트는 수정되지 않은 코드에서도 통과합니다.

1. **매 반복마다 새 맵을 만들고 최소 32회 반복하십시오.** `RandomState`는 인스턴스마다 시드를 부여하므로 하나의 맵을 반복해서 조사하는 것은 아무것도 측정하지 못합니다. 맵을 다시 만들어야 합니다. 트리에 있는 반복 횟수는 `ORDER_RESOLVE_ITERATIONS = 32`(`src/lang_bias.rs:755`), `ITERATIONS = 64`(`src/vision/detection/rt_detr_v2/sanitize.rs:388`), `ORDER_ITERATIONS = 64`(`src/distributed/registry_tests.rs:257`), `SELECTION_ITERATIONS = 32`(`src/distributed/disaggregated/request_router_tests.rs:865`)입니다.
2. **실제 동점을 만드십시오.** 서로 다른 키는 이미 전순서를 이루므로, 서로 다른 키로 만든 테스트는 수정 없이도 통과하며 아무것도 증명하지 못합니다. 테스트 대상은 바로 동점입니다.
3. **같은 타임스탬프를 의도적으로 기록하십시오.** `Instant::now()`는 이 프로젝트가 대상으로 하는 플랫폼에서 나노초 단위로 해석되므로, 별도의 호출 두 번으로 기록한 두 항목은 사실상 충돌하지 않습니다. `Instant::now()`로 항목에 타임스탬프를 찍고 동점이 되기를 기대하는 테스트는 다루려는 경우를 결코 만들어 내지 못합니다.

반복하는 이유는 수정 전의 실패율이 100퍼센트에 한참 못 미치기 때문입니다. PR #1281의 새 테스트는 수정되지 않은 함수에 대해 새로 만든 맵 **64개 중 27개**에서 실패했습니다. 단발성 버전의 테스트라면 대부분의 실행에서 통과하여 아무것도 잡아내지 못한 채 배포되었을 것입니다. PR #1269의 순서 테스트 세 개는 각각 반복 0, 반복 0, 반복 2에서 처음 실패했습니다.

수정되지 않은 코드에 대해 먼저 테스트를 실행하여 검증하십시오. 거기서 실패하지 않는다면 그 테스트는 결함을 테스트하는 것이 아닙니다.

### 대신 BTreeMap 선택하기

`BTreeMap`과 `BTreeSet`은 순회 순서가 정의되어 있으므로 컨테이너를 바꾸는 것이 구조적인 수정이며 해당 지점에서 이 부류 전체를 제거합니다. 대가로 O(1) `get`이 O(log n)이 되므로, 조회가 뜨겁지 않은 곳에서는 올바른 선택이고 뜨거운 곳에서는 잘못된 선택입니다. #1277이 반대 방향의 사례입니다. 레지스트리의 노드 맵은 주 요청 경로에서 읽히므로 `HashMap`을 유지하고 네 개의 accessor가 노드 id로 정렬합니다.

**여덟 건의 사례.** 이들은 2026-08-20부터 2026-08-22 사이의 한 번의 리뷰 스윕에서 여섯 개 모듈에 걸쳐 발견되었으며, 툴체인의 어떤 부분도 이를 표시하지 않았습니다.

- **#1265**(PR #1266). 네 개의 `filled_weights` 테스트 fixture가 `WeightMap`을 순회하며 키마다 LCG 시드 하나를 진행시켰기 때문에, 프로세스마다 서로 다른 합성 모델이 만들어졌습니다. 정렬도 인덱스도 보이지 않는, 루프 간에 이어지는 누산기였습니다. 원인을 찾기 전에 이미 추적되는 파일에 두 개의 잘못된 결론이 들어갔으며, 둘 다 `Makefile:604-616`에서 그 자리에서 철회되었습니다.
- **#1267**(PR #1269). `LangBiasYamlConfig::bias`는 `Option<HashMap<String, BiasValueStr>>`였고 이것이 `LangBiasSet.ordered`로 전달되었는데, 이는 우선순위 순서로 문서화되어 있고 첫 번째 언어가 우선하도록 소비됩니다. 그 결과 여러 CJK 언어가 있는 YAML 설정은 공유되는 한자 token에 실행마다 다른 bias를 할당했으며, 문서 주석의 스키마 예시 자체가 CJK 세 개짜리 설정이어서 그것을 복사하기만 해도 문제가 발생했습니다.
- **#1276**(PR #1281). `needs_sanitize`는 `weights.keys()` 순회 도중 처음 만난 마커에서 반환했기 때문에, 두 마커 계열을 모두 가진 체크포인트는 동전 던지기 같은 레이아웃 판정을 받았습니다. 비용이 큰 쪽은 잘못된 방향입니다. 이미 MLX 형식인 가중치에 sanitize 파이프라인을 다시 돌리면 conv 가중치가 두 번 전치되어, 이후 어떤 단계에서도 표시할 수 없는 형태상 유효한 텐서가 만들어집니다.
- **#1277**(PR #1284). 레지스트리 accessor 네 개(`all_nodes`, `nodes_with_role`, `peer_addresses`, `topology_summary`)가 `HashMap` 값을 정렬하지 않은 채 위치 기반으로 인덱싱하거나 입력 순서로 동점을 처리하는 소비자에게 반환했고, 이것이 주 요청 경로의 노드 선택에 도달했습니다.
- **#1286**(PR #1288). `src/distributed/tensor_parallel/cache_manager.rs`, `src/distributed/pipeline/cache_manager.rs`, `src/distributed/request_tracker.rs`의 세 eviction 경로. 세 곳 모두 후보 목록을 정렬했는데도 비결정적이었으며, 위의 안정 정렬 소절이 존재하는 이유가 바로 이 경우입니다.
- **#1293**(PR #1301). `BatchScheduler::select_eviction_victim`은 `LongestFirst`에서는 `generated_tokens.len()`에 대한 `max_by_key`로, `LowestPriority`에서는 우선순위 다음 길이에 대한 `min_by`로 preemption 대상을 골랐으며, 둘 다 `HashMap::values`인 `ActiveBatch::iter_sequences` 위에서 동작했습니다. 여덟 건 중 동점이 키의 조잡화가 아니라 일반적인 상태를 통해 도달 가능한 유일한 사례입니다. 함께 승인된 sequence는 lockstep으로 decode되어 token 수가 같아지고, `PreemptionPolicy::LongestFirst`와 `RequestPriority::Normal`은 모두 기본으로 제공되는 값입니다. 동점인 가장 긴 sequence는 어느 것이든 정책을 만족하므로 손상된 것은 없었지만, 동일하게 구성된 두 worker가 동일한 부하에서 서로 다른 요청을 preempt했고 운영자는 batch 상태로부터 그 선택을 재현할 수 없었습니다. `seq_id`를 덧붙여 수정했으며, `max_by_key`는 마지막 최댓값을, `min_by`는 첫 번째 최솟값을 취하므로 두 분기에서 방향을 반대로 했습니다.

**강제 방법:**

이 규칙에는 `make verify-*` target도, CI 작업도, 스크립트도 없습니다. 이는 누락이 아니라 #1287에서 내린 측정에 근거한 결정이며, 다음 사람이 다시 계산하지 않도록 그 근거 수치를 여기에 기록합니다. [`scripts/ci/check_kernel_dtype_keys.py`](../scripts/ci/check_kernel_dtype_keys.py)와 같은 형태의 후보 검사 두 가지를 프로토타입으로 만들어 `src/` 아래의 `.rs` 파일 1,248개 전체에 대해, 현재 트리와 위 처음 다섯 건의 수정 각각의 수정 전 트리를 재구성한 것에 대해 실행했습니다(#1293은 측정이 실행된 뒤에 발견되었습니다).

- **순서 없는 맵 수신자에 대한 `min_by_key` / `max_by_key`.** 현재 트리에서 네 건이 표시되었고 오탐은 없었습니다. `src/server/prompt_cache/store.rs:315`와 `:336`, `src/server/responses_store.rs:243`, `src/server/conversation_store.rs:151`의 LRU eviction 지점입니다. 위 사례 **8건 중 0건**을 잡습니다. 당시 알려진 일곱 건 중 `min_by_key`나 `max_by_key`를 쓴 것이 없었기 때문에, 재구성한 수정 전 트리는 모두 똑같은 네 건만 표시했고 다른 것은 표시하지 않았습니다. #1293은 `max_by_key`를 사용하는데도 이 섹션의 끝에 기록된 이유로 놓쳤습니다. 이 네 건은 실제 결함도 아닙니다. 네 개의 키는 모두 `std::time::Instant`이고, 각 writer는 호출마다 `Instant::now()`를 한 번 계산하여 하나의 항목에만 기록하므로 나노초 해상도에서는 지금 동점이 발생할 수 없습니다. 이들은 lint 입장에서는 true positive이고 실제 버그는 아닙니다. 이 구분이 중요합니다. 검사가 버그 네 개를 찾았다고 말하는 것은 거짓이며, 누군가 처음 확인하는 순간 규칙의 신뢰를 잃게 됩니다. 이를 관문으로 두면 적용되는 날 네 건 모두를 억제해야 하고, 억제는 패턴이 실제가 되는 변경(키가 더 조잡해지는 경우)에서 다시 활성화되지 않습니다. 이 패턴은 정말 취약하고 알아 둘 가치가 있지만 관문으로 삼을 만큼은 아닙니다.
- **순서 없는 맵 뷰로 만든 `Vec`을 순서에 민감하게 소비하는 경우.** 현재 트리에서 22건이 표시됩니다. PR #1288이 고친 *뒤*의 #1286 지점 세 곳을 모두 표시하는데, 수정은 형태를 유지한 채 정렬 키가 전순서인지 여부만 바꾸었고 이는 어떤 정규식으로도 구분할 수 없기 때문입니다. 8건 중 3건을 잡는다는 수치는 수정 전 트리에 대해서만 성립하며, 이 부류가 고쳐진 트리에서는 그 세 건이 영구적인 오탐이 됩니다. 별도로 접수된 네 건에는 각각 다른 구조적 이유로 도달하지 못합니다. #1265는 `Vec`도 정렬도 인덱스도 없는 루프 간 시드이고, #1267은 `for`에서 맵을 직접 순회하며 `.keys()`나 `.values()`를 전혀 호출하지 않으며, #1276은 소비자를 연결할 `Vec` 바인딩이 없이 `for`에서 일찍 반환하고, #1277은 생산자와 순서에 민감한 소비자가 서로 다른 모듈에 있습니다. `type X = HashMap<...>`를 트리 전체에서 스캔하면 `WeightMap`이 만드는 타입 별칭 사각지대는 해소되므로 그 한계 하나는 고칠 수 있지만, 나머지 셋은 이 수준의 분석으로는 고칠 수 없습니다.

일반적인 형태는 소스만으로는 결정할 수 없습니다. 순회 순서가 중요한지는 다른 모듈에 있는 경우가 많은 소비자에 달려 있기 때문입니다. 따라서 이 규칙은 리뷰와 위의 테스트 규칙으로 강제합니다. 앞으로 두 후보 검사 중 하나가 잡을 수 있었을 형태의 사례가 나오면 이 문제를 다시 검토하십시오.

그 사각지대는 이미 하나의 발견을 놓치게 했으며, 위의 결정이 유리한 내용만이 아니라 정직하게 남도록 여기에 기록합니다. #1293은 `BatchScheduler::select_eviction_victim`에서 발견된 여덟 번째 사례이며, 잠재적이던 네 건과 달리 동점에 도달할 수 있습니다. lockstep으로 decode되는 sequence들이 일상적으로 공유하는 작은 정수인 `generated_tokens.len()`에서 동점이 깨지기 때문입니다. 수신자가 같은 파일에 선언된 맵의 `.values()` 호출이 아니라 모듈 간 accessor 메서드(`ActiveBatch::iter_sequences`)였기 때문에 두 후보 검사 모두 이를 표시하지 못했으며, 이는 #1277에서 설명한 한계와 정확히 같습니다. 사람이 하는 스윕이 이를 찾았습니다. 이것이 검사를 도입하지 않기로 한 결정의 비용이며, 문제를 다시 검토할 때 저울질해야 할 수치입니다.
