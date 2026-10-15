# mlxcel 문서 구성

이 디렉터리는 공개 릴리스 자료를 위한 공용 문서 루트입니다.
다음 항목을 포함할 수 있습니다.

1. 루트 `README.md`에서 직접 링크되는 **GitHub용 Markdown 문서**.
2. MkDocs 전용 소스 트리 아래의 **MkDocs 사이트 콘텐츠**. 이 저장소에는 없습니다.
   아래 "MkDocs 매뉴얼"을 참고하십시오.
3. 유지보수자와 기여자를 위한 **Git/GitHub 워크플로 문서**.

현재 최상위 파일은 루트 README에서 링크되는 GitHub용 문서입니다. 나중에 더 풍부한
MkDocs 페이지가 추가되더라도 독립적인 Markdown 파일로 읽을 수 있어야 합니다.

현재 GitHub용 문서:

1. `installation.md` - macOS, CUDA가 있는 Linux, AMD ROCm(실험적)이 있는 Linux의 플랫폼 요구 사항과 빌드 플래그.
2. `environment-variables.md` - `MLXCEL_*` 런타임, 빌드, 다운로더, 캐시, 진단용 설정.
3. `benchmarks.md` - 벤치마크 방법론과 향후 원시 결과 표에 대한 요구 사항.
4. `supported-models.md` - 관리되는 아키텍처/체크포인트 지원 매트릭스.
5. `architecture.md` - 런타임 아키텍처와 주요 구성 요소. mlxcelverse(`src/lib/mlxcelverse/` 아래의 백엔드별 MLX 오버레이와 mlxcel 커널)를 포함합니다.
6. `distributed.md` - 텐서/파이프라인 병렬 설정과 제한 사항.
7. `turbo-kv-cache.md` - TurboQuant 모드, 통합 paged KV cache, 품질/성능 트레이드오프, 플래그.
8. `CONTINUOUS_BATCHING.md` - continuous batching 스케줄러, paged decode, 분리형(disaggregated) prefill/decode/router 서빙.
9. `responses-api.md` - 구현된 `/v1/responses` 하위 집합과 미구현 사항.
10. `server-features.md` - 서버 라우트, 모델 스토어와 router 의미 체계, LoRA, 설정, 스트리밍, 관측성, 전송, 보안 경계에 대한 운영자 가이드.
11. `audio-api.md` - 구현된 `/v1/audio` 엔드포인트: 오디오 지원 chat 및 Whisper STT 설정, 컨테이너 및 스트리밍 동작, 요청/응답 레퍼런스, WAV 출력, 검증 순서.
12. `adding-models.md` - 새 모델 아키텍처를 위한 기여 가이드.
13. `block-diffusion.md` - DiffusionGemma block-diffusion 생성: 캔버스 denoising 대 autoregressive, CLI 플래그, 처리량, 1단계 제한 사항.
14. `python-client.md` - OpenAI 호환 서버 위에서 동작하는 `mlxcel` Python 클라이언트: managed 및 connect 모드, 스트리밍, chat, 구조화된 출력, `openai_client` 탈출구, 비동기 사용, 문제 해결.
15. `audio-preprocessing.md` - 공유 WAV/MP3/FLAC 모델 입력 정규화, 로드된 패밀리 정책, 리소스 제한, 취소, 메트릭, 현재 XLA 기능 경계.
16. `speculative-acceptance.md` - speculative decoding의 acceptance 규칙, 각 코드 경로가 실행하는 규칙, 분포 보존 보장과 그 RNG 의존성, kill switch, 로그에서 활성 규칙을 읽는 방법.
17. `mla-absorbed-decode.md` - 압축된 latent KV cache 위에서의 DeepSeek 계열 행렬 흡수(matrix-absorbed) MLA decode: 항등식, 캐시 레이아웃, 플래그, 검증된 것과 검증되지 않은 것.
18. `cascade-attention.md` - 공유 프롬프트 접두사(cascade) decode: 여러 동시 시퀀스가 공유하는 접두사를 시퀀스마다가 아니라 step마다 한 번만 계산하는 방식, 2단계 분해, 플래그, 실행이 어떤 경로를 택했는지 확인하는 방법.
19. `sparse-paged-decode.md` - fused v2 decode 커널 위의 페이지 간접 참조(page indirection)로 환원된 sparse attention: 주소 지정 논거, MiniMax-M3 라우팅, DeepSeek Sparse Attention이 라우팅되지 않는 이유, 디스패치 하한이 위치하는 곳, kill switch, 벤치마크 하니스.
20. `mtp-policy-api.md` - 적응형 B=1 MTP 판정에 대한 지원되는 읽기 인터페이스(`GET /v1/internal/mtp-policy`): 응답 본문, 네 가지 상태, unavailable 사유, 스키마 버전 관리 및 호환성 정책.
21. `code-guidelines.md` - 파일 크기와 모듈 분할 임계값(`<name>_helpers.rs`를 추출해야 하는 시점과 인라인 테스트를 형제 `_tests.rs`로 옮기는 시점 포함), 공유 함수에 대한 `// Used by:` 주석 규약, CUDA가 컴파일된 모듈 캐시를 해당 dtype으로 키잉하므로 변하는 모든 입력 dtype이 `template_args`에 나타나야 한다는 JIT 커널 규칙, 그리고 순서에 민감한 소비자의 범위, 전순서가 아닌 키에 대한 안정 정렬이 결과를 고정하지 못하는 이유, 반복마다 새 맵을 사용하는 테스트 요구 사항, 정적 검사로 이를 강제하지 않는 이유를 다루는 `HashMap` 반복 순서 규칙.
22. `embeddings.md` - `/v1/embeddings` 및 `/v1/rerank` 엔드포인트와 `mlxcel embed`, `mlxcel rerank`: 임베딩 및 reranker 감지, pooling과 스코어링, 멀티모달 및 late-interaction 입력, 요청/응답 스키마, 서버 플래그, 오류 코드, 패밀리 등록 지침.
23. `llama-server-compat.md` - 검증된 llama-server b10621 호환성 경계: `compat/llama-server/b10621/` 아래의 기계 판독 가능 매니페스트를 읽는 방법, 호환성 정책 상태, 에픽 #1431에서 사용하는 영역 샤딩, CI 게이트, 결정적 재생성 절차.

## mlxcelverse 유지보수

`mlxcelverse/rocm-overlay.md`는 ROCm 오버레이를 MLX pin 및 그 기반이 되는 NripeshN/mlx 포크와 동기화된 상태로 유지하는 절차이며, `scripts/mlxcelverse/` 도구와 `make verify-rocm-overlay` 게이트를 포함합니다. `mlxcelverse/upstream/`에는 제출되지 않은 채 과거 참고용으로 보관되는 포크 측 수정 패키지가 있습니다. 포크에 대한 수정은 오버레이에 남기며 NripeshN/mlx에는 PR을 보내지 않습니다. 해당 README가 색인입니다.

## Architecture Decision Records

`adr/`에는 번호가 매겨진 Architecture Decision Record가 있으며, 파일당 하나의 중요한 결정을 담고, Accepted 상태가 되면 변경할 수 없습니다. 색인은 `adr/README.md`를 참고하십시오.

## MkDocs 매뉴얼

<https://mlxcel.lablup.ai/en/manual/>에 게시된 매뉴얼은 이 디렉터리에서 빌드되지 않습니다.
소스(`docs/en`, `docs/ko`, `docs/shared`,
`docs/requirements.txt`, `docs/scripts`)는 별도의 문서 트리에서 관리되며
이 저장소에 포함되어 있지 않습니다.

저장소 루트의 mkdocs 설정 네 개(`mkdocs.yml`, `mkdocs.ko.yml`,
`mkdocs.pdf.yml`, `mkdocs.ko.pdf.yml`)는 해당 트리에 속하며, 이곳에서 그 트리와
동기화된 상태로 유지됩니다. 이들의 `docs_dir`, `custom_dir`, `nav:` 항목은 그 트리 내부의 경로로 읽어야 합니다.
이 설정들 중 위에 나열된 어떤 파일도 지칭하지 않으며, 위에 나열된 파일 중
어느 것도 `nav:`에 나타나지 않습니다. 이는 의도된 것이며 불일치가 아닙니다.
여기의 GitHub용 문서는 GitHub에서 일반 Markdown으로 읽히도록 만들어졌고,
매뉴얼은 별도로 작성된 산출물입니다.

`docs-*` Makefile 타깃은 해당 소스에서 매뉴얼을 빌드합니다. 이 저장소에서는
`uv`, 심볼릭 링크 또는 사이트 빌드 도중에 실패하는 대신 설명과 함께 즉시
중단됩니다. 대신 게시된 매뉴얼을 읽으십시오.

향후 레이아웃 예시:

- GitHub 이슈/PR/릴리스 워크플로 노트를 위한 `docs/github/...`.
- 브랜치, 커밋, 태그, 미러링 절차를 위한 `docs/git/...`.

해당 최상위 문서가 의도적으로 리디렉션 형태의 색인 페이지로 대체되지 않는 한 루트 README 링크는 안정적으로 유지하십시오.
