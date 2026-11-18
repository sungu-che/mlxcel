# Temperature 0: 일반 디코드 대 블록 기반 KV 히스토리 (#2778)

블록 검증 경로(`--prompt-lookup` 또는 draft 모델)를 거치는 temperature 0 실행은 일반 디코드(plain-decode) 스트림에서 벗어날 수 있습니다. 이 문서는 측정된 내용, 증거가 뒷받침하는 원인, 코드가 변경되지 않은 이유, 그리고 유지보수자가 모든 수치를 재현할 수 있는 명령을 기록합니다. 이 문서는 `docs/benchmarks.md`의 방법론("수치를 변화시키는 변경 평가하기")을 따릅니다. 바이트 단위 동일성은 예/아니오만 말해주므로, 아래의 비교는 teacher-forcing 방식이며 레퍼런스가 얼마나 명확했는지(decided)에 따라 게이트됩니다.

**상태: 알려진 제한 사항, 근본 원인 미결, 코드 변경 없음.** 이슈 #2778은 이것이 0.8.0 릴리스를 차단하지 않는다는 결정을 기록하고 근본 원인을 미결로 남겼습니다. 릴리스 노트에서 temperature 0 speculative 출력이 일반 디코드와 일치한다고 주장해서는 안 됩니다. 일반 디코드는 영향을 받지 않습니다. 이 분석의 범위는 한정되어 있습니다. 아래에서 재현된 반전(flip)을 설명할 뿐, 이슈 본문의 원래 Qwen3 위치 197 변동(일반 디코드는 해당 토큰을 2.0 차이로 선호하고 대부분의 블록 히스토리는 다른 토큰을 약 35 차이로 선호함)을 설명하지는 못하며, 이는 여전히 실제 KV 히스토리 차이일 수 있습니다("제한 사항" 참조).

## 재현된 반전(flip)의 원인

여기서 재현된 반전은 **argmax 동점(tie)**입니다. 동일한 컨텍스트에서 폭 8의 검증 블록은 fp16 KV를 사용하는 Qwen3-0.6B-4bit에서 레퍼런스 자체의 top-1 logit을 최대 0.75만큼 이동시켰으며, 유일한 top-1 불일치는 레퍼런스의 top-2 차이가 0.0인 지점에서 발생했습니다. 자유 실행(free-running) 스트림에서는 반전된 토큰 이후의 나머지 출력이 달라집니다.

제시된 메커니즘은 블록이 첫 번째 이후의 행들을 다른 산술 경로로 계산한다는 것입니다(양자화 projection이 `M = block width`에서 실행되고, MLX는 `M`에 따라 커널을 선택함). 이 실행들에서 이는 하나의 가설입니다. 일치된 레퍼런스 히스토리는 8회의 디코드 스텝이 아니라 1회의 prefill forward였으며, 여기서의 명령은 K/V 텐서가 아닌 logit을 비교합니다("제한 사항"에는 #2778에 보고된 별도의 KV 덤프가 기록되어 있음). KV cache가 양자화되어 저장될 때(`--kv-cache-mode int8`, `fp16+turbo3`, `fp16+turbo4`), 종단 간 스트림은 측정된 6개의 체크포인트/모드 셀 중 4개에서 일반 스트림을 벗어났으며, fp16 KV의 경우는 2개 중 0개였습니다. 이러한 발산은 logit 수준에서 측정되지는 않았습니다.

아래 내용 중 어떤 것도 손상된 KV 내용을 보여주지 않으며, 위치 197 보고에 대해 이를 배제하지도 않습니다. 두 경로가 동점에서 서로 다른 logit을 계산하는 경우, 각 토큰은 자신의 경로가 계산한 logit의 argmax입니다.

각 진술의 근거:

| 주장 | 상태 | 증거 |
|---|---|---|
| 동일 컨텍스트에서 검증 블록은 반올림 스케일로 logit을 이동시키며 정확한 동점만 반전시킴 | 측정됨 (Qwen3-0.6B-4bit, fp16 KV) | 1 |
| 순차적으로 구축된 컨텍스트만 보는 블록의 0번 행은 top-1 logit에서 디코드 스텝과 일치함 | 측정됨 (Qwen3-0.6B-4bit, 48개 위치) | 1 |
| 서로 다른 컨텍스트 길이에서 블록을 폭 1 실행과 비교하면 불일치가 과장됨 | 측정됨 | 1 |
| 블록 자체의 K/V 산술이 1~7번 행의 logit 델타를 유발함 | **가설**; 여기에는 K/V 비교가 없으며 레퍼런스 히스토리는 1회의 prefill forward임 | 1, 제한 사항 |
| 1-토큰 디코드 분할 자체는 무해함 | 측정됨 | 2 |
| 폭 8 트레이스를 반복 실행하면 비트 단위로 동일함 | 측정됨 | 1 |
| fp16 KV에서는 일반 디코드와 `--prompt-lookup`이 동일한 스트림을 방출함 | 측정됨: 2개 체크포인트, 1개 프롬프트, 160 토큰 | 3 |
| 양자화된 KV에서는 일반 디코드와 `--prompt-lookup`이 서로 갈라짐 | 측정됨: 6개 셀 중 4개 | 3 |
| 반올림 수준의 K/V 차이가 양자화에 의해 눈에 띄는 차이로 증폭됨 | **가설**; Gemma 3 logit 수준 프로브는 int8에서 더 큰 델타를 보이지 않음 | 4, 제한 사항 |

## 증거

하드웨어: Apple M1 Pro, Metal. 빌드:
`cargo build --release --features metal,accelerate --example logit_trace` 및
`cargo build --release --features metal,accelerate`. 트레이스는 단일 520 토큰 코퍼스(Qwen3 토크나이저 기준; Gemma 3 토크나이저는 BOS 앵커 포함 536 토큰)에서 `PREFILL=128`, top 8로 teacher-forced 실행되었습니다. 이 실행들의 소스 리비전은 기록되지 않았습니다. 여기서 추적된 폭 범위에서 Metal 3~16행 affine `qmv_mma`를 추가하는 MLX pin 상향(`9f806b36`, #2876) 이전이므로, 수치를 최신으로 취급하기 전에 다시 실행하십시오.

### Fixture

```bash
# 저장소 루트에서 실행하십시오. 이후 블록은 $REPO와 이 임시 디렉터리를 사용합니다.
REPO=$PWD
mkdir -p /tmp/mlxcel_2778 && cd /tmp/mlxcel_2778
cat > passage.txt <<'EOF'
The transformer architecture processes sequences through stacked attention layers. Each layer attends over every preceding token, building contextual representations that the feed-forward network refines. Positional encodings let the model distinguish order, and residual connections carry information forward across depth. Training adjusts billions of parameters so that the final projection predicts the next token from context.
EOF
for i in 1 2 3 4 5 6 7 8; do cat passage.txt; done > corpus_long.txt
{ printf 'Copy the following passage exactly, word for word. Do not summarize, do not paraphrase, do not add commentary. Output only the passage as given:\n\n'; cat passage.txt; } > copy_prompt.txt
```

### 증거 1: 동일 컨텍스트에서의 블록 폭

폭 1은 디코드 스텝이며, 폭 8은 `--prompt-lookup`이 초안을 작성하는 크기의 검증 블록입니다. 폭이 테스트 대상 변수입니다.

폭 8 트레이스와 폭 1 트레이스를 행별로 비교하는 것은 혼란 변수가 있습니다. `logit_trace` 하니스는 모든 청크에 `PREFILL` 토큰 윈도우를 부여하므로, 폭 8 청크의 `j`번 행은 `PREFILL + j` 토큰을 따르는 반면 폭 1 청크는 `PREFILL`을 따릅니다. 컨텍스트 길이만으로도 logit이 움직입니다. 아래의 첫 번째 패스는 그 혼란 효과의 크기를 보여주며, 일치된(matched) 패스는 이를 제거합니다.

첫 번째 패스, 컨텍스트 길이가 일치하지 않음:

```bash
M=~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit
T=$REPO/target/release/examples/logit_trace
$T $M corpus_long.txt 1 384 8 128 > w1.tsv
$T $M corpus_long.txt 8  48 8 128 > w8.tsv
python3 "$REPO/scripts/compare_logit_traces.py" w1.tsv w8.tsv --decided 2
```

| 측정 항목 | 결과 |
|---|---|
| 비교된 위치 수 | 384 |
| top-1 불일치 (전체 위치) | 5 / 384 (1.30%) |
| top-1 불일치, 확실한 위치 (차이 >= 2.0) | 0 / 354 (0.00%) |
| 불일치 위치에서 레퍼런스의 최대 top-2 차이 | 0.000 |
| 후보군에서 레퍼런스의 선택 순위 | 2위 (5개 중 5개) |
| 레퍼런스 자체 선택에 대한 logit 델타 | p50 0.375, p90 1.0, p99 2.0, max 2.375 |
| perplexity, 폭 1 대 폭 8 | 1.0853 대 1.0773 (-0.74%) |

일치된 패스: 모든 폭 8 블록의 `j`번 행을 `PREFILL = 128 + j`에서 추적된 폭 1 실행과 비교하므로, 둘 다 동일한 토큰 히스토리와 동일한 입력 토큰을 봅니다. `scripts/matched_width_trace.sh`가 트레이스를 작성하고 `scripts/compare_matched_width.py`가 이를 비교합니다(폭 8 트레이스 1회와 폭 1 트레이스 8회를 실행하므로 몇 분 소요됩니다):

```bash
LOGIT_TRACE=$T "$REPO/scripts/matched_width_trace.sh" $M corpus_long.txt out_q3     # WIDTH=8 PREFILL=128 BLOCKS=48
python3 "$REPO/scripts/compare_matched_width.py" out_q3
```

| 행 | 위치 수 | top-1 불일치 | 확실한 위치 (차이 >= 2) | 확실한 위치의 불일치 | 최대 logit 델타 |
|---|---|---|---|---|---|
| 0 | 48 | 0 | 45 | 0 | 0.000 |
| 1 | 48 | 1 | 44 | 0 | 0.250 |
| 2 | 48 | 0 | 44 | 0 | 0.375 |
| 3 | 48 | 0 | 45 | 0 | 0.250 |
| 4 | 48 | 0 | 44 | 0 | 0.375 |
| 5 | 48 | 0 | 43 | 0 | 0.375 |
| 6 | 48 | 0 | 46 | 0 | 0.250 |
| 7 | 48 | 0 | 47 | 0 | 0.750 |
| 전체 | 384 | 1 | 358 | 0 | |

- 유일한 불일치는 코퍼스 위치 193, 1번 행에서 발생했으며, 여기서 레퍼런스의 상위 2개 logit은 동일했습니다(차이 0.0).
- 0번 행: 48개 위치 모두 두 실행에서 동일한 레퍼런스 top-1 logit을 가집니다(0이 아닌 델타 없음). 0번 행은 순차적으로 구축된 히스토리만 봅니다.
- 1~7번 행: 336개 위치 중 195개 위치에서 레퍼런스 자체 선택에 0이 아닌 logit 델타 발생; p50 0.125, p90 0.25, p99 0.375, max 0.75.
- Perplexity: 1.0771 (폭 1) 대 1.0773 (폭 8), +0.023%.
- 폭 8 트레이스 반복 실행은 비트 단위로 동일함: 두 번의 독립적인 실행에서 나온 `w8.tsv` 파일을 `cmp`로 비교해도 차이가 보고되지 않음.

일치하지 않는 첫 번째 패스는 `MLXCEL_TRACE_START_TOKEN`을 설정하지 않아 위치 0~383을 추적하는 반면, 일치된 패스는 128~511을 추적합니다. 둘은 동일한 수의 위치를 다루지만 동일한 위치는 아니며, 일치하지 않는 패스의 처음 128개 위치는 더 짧은 컨텍스트 윈도우를 가집니다. 일치하지 않는 1~7번 행(336개 위치)에서는 309개 위치에서 레퍼런스 자체 선택에 0이 아닌 logit 델타(p50 0.375, p90 1.0, max 2.375)와 차이 0.0에서의 5개 top-1 불일치가 발생한 반면, 일치된 패스에서는 336개 중 195개(p50 0.125, p90 0.25, max 0.75)와 1개의 불일치만 있었습니다. 0번 행은 컨텍스트와 입력 토큰이 각 패스의 두 arm 모두에서 동일하므로 두 패스 모두에서 델타가 0입니다(top-1 logit에서 48개 중 48개). 일치된 패스의 더 작은 델타는 컨텍스트 길이가 일치하지 않는 변동의 대부분을 차지한다는 사실과 부합합니다. 남아 있는 것은 1~7번 행에 있으며, 이번 실행에서는 정확한 동점만 반전되었습니다. 여기서 폭 1 히스토리는 8회의 순차적 디코드 스텝이 아니라 1회의 prefill forward입니다.

### 증거 2: 디코드 스텝 분할

`MLXCEL_TRACE_DECODE_STEP=1`은 추적된 블록 직전에 마지막 컨텍스트 토큰을 자체 단일 토큰 forward로 전달합니다(speculative verify 블록이 실행되는 상태, #1793). 블록 폭 64에서 분할이 없는 동일한 컨텍스트 패스와 비교합니다:

```bash
C=corpus_long.txt
MLXCEL_TRACE_DECODE_STEP=1 $T $M $C 64 8 8 256 > armB.tsv   # 분할 있음
$T $M $C 64 8 8 256 > armC.tsv                                            # 분할 없음
python3 "$REPO/scripts/compare_logit_traces.py" armB.tsv armC.tsv --decided 2
```

결과: top-1 불일치 0 / 512, 확실한 위치에서 0 / 444. 분할 자체는 어떠한 변경도 도입하지 않습니다.

대조군은 비교가 민감함을 확인해 줍니다. 컨텍스트 패스가 없는 경우(`PREFILL=0`, 각 64 토큰 청크마다 새로운 캐시, arm B 및 C와 다른 컨텍스트):

```bash
$T $M $C 64 8 8 0 > armA.tsv
python3 "$REPO/scripts/compare_logit_traces.py" armA.tsv armB.tsv --decided 2
```

324 / 512 위치에서 불일치하며(63.28%; 확실한 위치 38 / 114, 불일치 시 최대 차이 3.875), arm C에 대해서도 동일합니다. 이는 블록 산술이 아닌 컨텍스트 길이 민감도 때문입니다.

### 증거 3: 종단 간 일반 디코드 대 `--prompt-lookup`

```bash
PROMPT=$(cat copy_prompt.txt)
for kv in fp16 int8 fp16+turbo3 fp16+turbo4; do
  for arm in plain pl; do
    extra=""; [ "$arm" = pl ] && extra="--prompt-lookup"
    "$REPO/target/release/mlxcel" generate -m "$MODEL" -p "$PROMPT" -n 160 --temp 0 --seed 42 \
      --kv-cache-mode $kv --show-reasoning $extra > ${kv}_${arm}.raw 2>&1
    grep -v -e '^Runtime device' -e '^Wired memory' -e '^KV cache mode' -e '^Loading model' \
      -e '^Detected' -e '^GPU:' -e '^Converting' -e '^Note:' -e '^Model loaded' \
      -e '^Generating' -e '^\[Prompt lookup\]' -e '^\[Generated' ${kv}_${arm}.raw > ${kv}_${arm}.txt
  done
done
python3 - <<'EOF'
P = len(open('copy_prompt.txt').read())          # 577: 프롬프트 에코
def gen(p):
    g = open(p).read()[P:]
    return g[1:] if g.startswith('\n') else g    # prompt-lookup 경로는 여분의 \n 하나를 출력함
def first_diff(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
for kv in ('fp16', 'int8', 'fp16+turbo3', 'fp16+turbo4'):
    print(kv, first_diff(gen(f'{kv}_plain.txt'), gen(f'{kv}_pl.txt')))
EOF
```

`MODEL`은 `~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit`이고, 그다음은 bf16 `Qwen/Qwen3-0.6B` 체크포인트 디렉터리입니다. 생성된 텍스트만 비교합니다. 배너와 통계 줄은 삭제되고, 처음 `len(copy_prompt.txt)`자(프롬프트 에코)와 하나의 선행 개행도 제거됩니다. 토큰 스트림은 거기서 다르지 않습니다. `int8`은 28개 레이어 전체에 적용되는 토큰별 absmax 양자화입니다. `fp16+turboN`은 K를 fp16으로 유지하고 V를 N비트로 저장하며 28개 중 24개 레이어에 적용됩니다.

| 체크포인트 | KV 모드 | 일반 디코드 대 `--prompt-lookup` |
|---|---|---|
| Qwen3-0.6B-4bit | fp16 | 동일함 (748자) |
| Qwen3-0.6B-4bit | int8 | 동일함 (894자) |
| Qwen3-0.6B-4bit | fp16+turbo3 | 동일함 (839자) |
| Qwen3-0.6B-4bit | fp16+turbo4 | 생성된 문자 432에서 갈라짐 |
| Qwen3-0.6B bf16 | fp16 | 동일함 (865자) |
| Qwen3-0.6B bf16 | int8 | 생성된 문자 82에서 갈라짐 |
| Qwen3-0.6B bf16 | fp16+turbo3 | 생성된 문자 276에서 갈라짐 |
| Qwen3-0.6B bf16 | fp16+turbo4 | 생성된 문자 673에서 갈라짐 |

첫 번째 발산 지점은 여러 이어짐이 그럴듯한 자유 형식 추론(reasoning) 텍스트입니다. bf16 int8의 경우 `... without any summary or paraphrase.`(일반 디코드) 대 `... without any summary or commentary.`(prompt lookup), bf16 turbo3의 경우 `The passage talks about` 대 `The passage is about`입니다. KV 모드 자체로도 일반 스트림이 변경되므로(fp16 대 int8, 둘 다 일반 디코드에서 4-bit의 경우 문자 192, bf16의 경우 문자 175에서 발산), 양자화된 스트림은 fp16 스트림의 손상된 복사본이 아니라 서로 다른 스트림입니다. bf16 `fp16+turbo3` 쌍을 다시 실행했을 때 두 스트림이 정확하게 재현되었습니다. 이러한 각 위치가 logit 수준에서 아슬아슬한 동점이었는지는 측정되지 않았습니다. Qwen3 트레이스는 KV 저장 모드를 설치할 수 없기 때문입니다(증거 4).

### 증거 4: 이를 지원하는 모델에서의 트레이스 KV 저장 arm

`MLXCEL_TRACE_KV_CACHE_MODE`는 `--kv-cache-mode`와 동일한 방식으로 KV를 저장합니다. 이는 모델 소유 캐시(배포된 패밀리 중 Gemma 3 및 Gemma 4)에만 도달하며 다른 모델에서는 "this model does not own its KV caches"와 함께 종료됩니다. Qwen3와 Llama가 그러한 모델입니다. 이 arm의 프로브로서, 증거 1의 일치된 폭 비교를 4-bit Gemma 3 270m 체크포인트에서 저장 모드별로 한 번씩 실행했습니다. `$GEMMA`는 로컬 체크포인트입니다(`model_type` `gemma3_text`, hidden size 640, 18개 레이어, 양자화 그룹 크기 64, 4비트). hub id는 기록되지 않았으므로 이 실행은 공개 스냅샷 id에 묶여 있지 않습니다.

```bash
for kv in fp16 int8; do
  LOGIT_TRACE=$T MLXCEL_TRACE_KV_CACHE_MODE=$kv "$REPO/scripts/matched_width_trace.sh" $GEMMA corpus_long.txt out_gm_$kv
  python3 "$REPO/scripts/compare_matched_width.py" out_gm_$kv
done
```

| KV 모드 | top-1 불일치 | 확실한 불일치 | 0번 행 0이 아닌 델타 | 1~7번 행 델타 p50 / p99 / max | perplexity 폭 1 대 8 |
|---|---|---|---|---|---|
| fp16 | 0 / 384 | 0 / 364 | 48개 중 6개 | 0.125 / 1.0 / 2.0 | 1.0433 대 1.0439 (+0.056%) |
| int8 | 0 / 384 | 0 / 365 | 48개 중 44개 | 0.048 / 0.303 / 2.20 | 1.0503 대 1.0500 (-0.034%) |

이 체크포인트에서는 어느 모드에서도 반전이 나타나지 않습니다. int8에서 p50 및 p99 델타는 fp16보다 작고, 최대값은 약간 더 큽니다(2.0 대 2.20). 따라서 이 프로브는 logit 수준에서 증폭 가설을 **뒷받침하지 않습니다**. 단지 해당 arm이 실행되고 그 출력이 어떻게 보이는지만 보여줍니다. Qwen3와 달리 Gemma 3의 0번 행은 어느 모드에서도 비트 단위로 동일하지 않으므로, 위의 "0번 행 일치" 진술은 일반적인 진술이 아니라 Qwen3-0.6B-4bit에서의 관찰 결과입니다.

## 코드가 변경되지 않은 이유

검증 행을 디코드 스텝과 비트 단위로 동일하게 만들려면 해당 폭에서 비트 단위로 동일한 산술 경로가 필요합니다. 그러한 경로가 존재하는지는 MLX가 `M`에 대해 디스패치하는 커널과 하드웨어에 달려 있습니다. `docs/benchmarks.md`는 Apple GPU 세대 13/14에서 12 미만의 블록 폭에 대해 affine 4-bit 타깃이 바이트 단위로 동일하며, 세대 15 이상에서는 `qmv_wide`를 비활성화하기 전까지 폭 2부터 발산한다고 기록하고 있으며, MTP 및 DFlash 경로가 그러한 이유로 런타임 블록 대 체인 정확성 프로브로 게이트됩니다. `--prompt-lookup` 및 풀 모델 drafter 뒤에 있는 토큰 전용 루프에는 그러한 게이트가 없습니다. `docs/speculative-acceptance.md`("오프라인 라우팅")에는 그 스트림이 모든 위치에서 타깃 자체의 argmax이며 일반 디코드와 바이트 단위로 동일함을 보장하지 않는다고 문서화되어 있습니다. 거기에 게이트를 추가하는 것은 이 문서가 결정하지 않는 처리량 간의 트레이드오프입니다. 일치된 폭 트레이스에서 두 arm은 비트 단위로 동일하지 않습니다(1~7번 행 336개 중 195개가 top-1 logit에서 다름).

트레이스 증거에서 발산은 레퍼런스의 상위 2개 logit이 동일했던 위치로 제한됩니다. 그렇다고 해서 회귀(regression)가 없다는 것이 확정되는 것은 아닙니다. #2778은 해당 동작이 #2768 이전부터 있었는지 측정되지 않았다고 기록하며, 여기서도 마찬가지입니다. 증거 3의 양자화된 종단 간 발산은 logit 수준에서 측정되지 않았습니다. 양자화된 KV 저장은 사용자가 정확성과 메모리를 맞바꾸기로 선택한 것이며, 증거 3은 일반 스트림 자체도 KV 모드에 따라 변한다는 것을 보여줍니다.

## 이 증거의 한계

- 원래 보고인 위치 197에서의 Qwen3 `copy`(일반 디코드는 해당 토큰을 2.0 차이로 선호하고 대부분의 블록 히스토리는 다른 토큰을 약 35 차이로 선호함)는 **재현되지 않았습니다**. 원래 실행의 프롬프트 텍스트와 KV 모드는 이슈에 없으며, 여기서의 실행은 M1 Ultra가 아닌 M1 Pro에서 이루어졌습니다. 35 logit 변동은 여기서 측정된 그 어떤 것보다 훨씬 큽니다(일치된 패스에서 가장 큰 불일치 차이 0.0, 최대 logit 델타 일치되지 않은 경우 2.375, 일치된 경우 0.75). 기본 fp16 KV에서 측정된 것이라면 이 문서는 이를 설명하지 못합니다. 이슈를 다시 열기 위한 조건은 동일하게 유지됩니다. 손상된 KV 내용을 보여주는 재현기(reproducer)입니다.
- 이 문서의 명령들은 K/V 텐서를 비교하지 않으므로, 이슈의 첫 번째 수락 기준은 여전히 미결입니다. #2778에 게시된 별도의 하니스(`examples/kv_history_divergence.rs`, 이 트리에 없음)는 Qwen3-0.6B-4bit, M1 Pro 및 64 토큰 컨텍스트에서 폭 1, 2, 4, 8의 1-토큰 추가와 블록 추가 사이에 서로 다른 KV 바이트가 0개이며, 폭 16부터 서로 다른 바이트가 나타나고 해당 구간에서 argmax 반전이 없다고 보고합니다. 그것이 유효하다면 증거 1의 1~7번 행 델타는 블록 폭 8만으로는 설명되지 않으며 레퍼런스의 단일 prefill 히스토리와 블록의 히스토리 간의 차이를 포함합니다. 히스토리가 반올림 스케일에서 다르다는 진술은 여기서 logit에 기반합니다.
- 양자화 증폭 단계(반올림 수준의 K 차이가 양자화 단계를 넘어 완전한 한 단계가 됨)는 logit 수준에서 측정되지 않았습니다. Qwen3 종단 간 증거는 이와 부합하지만(양자화 셀 6개 중 4개 반전, fp16 2개 중 0개), Gemma 3 프로브는 이를 보여주지 않으며, 한 프롬프트에서의 8개 셀은 이를 arm 간의 무관한 차이와 분리해 내지 못합니다.
- Qwen3 스트림은 1개 프롬프트, 2개 체크포인트, 1개 머신에서 160개의 생성된 토큰에 대해 확인되었습니다. 트레이스 증거는 1개 코퍼스입니다.
- 이슈의 Llama `prose` 케이스(위치 137, 차이 0.016)는 진정한 아슬아슬한 동점(near-tie)이며 추가 설명이 필요하지 않습니다.
