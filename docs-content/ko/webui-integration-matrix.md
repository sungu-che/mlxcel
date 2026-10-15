# WebUI 통합 요구사항 매트릭스 (#1848)

이 매트릭스는 번들 WebUI 릴리스 게이트의 기준(canonical) 인수인계 문서입니다. Mock/단위 테스트 성공은 필수 하드웨어, 설치된 아티팩트, 브라우저, CUDA, Safari/VoiceOver, native-hidden 증거 행을 결코 대체하지 못합니다. `not-run` 또는 `blocked`로 표시된 행은 root가 최종 증거를 기록할 때까지 릴리스 차단 요소로 남습니다.

| 요구사항 | 자동화 / 커밋된 게이트 | 현재 브랜치의 증거 상태 | 남은 릴리스 증거 |
|---|---|---|---|
| 하나의 통합 결정적(deterministic) WebUI 게이트 | `make verify-webui`는 계약 드리프트, 순수 verifier 헬퍼 단위 테스트, 프런트엔드 type/lint/unit, Chromium/Firefox/WebKit Playwright 스캐폴드, 번들 재현성, 범위가 한정된 Rust WebUI/라우터 테스트를 실행합니다. | 이 에이전트에서는 `not-run`입니다. root가 소스 동결 후 실행해야 합니다. | 전체 워크스페이스 `test-fast`, clippy, fmt는 여전히 별도의 root 게이트입니다. |
| Vite 전용이 아닌 설치된 아티팩트 | `make verify-webui-installed WEBUI_SERVER_BIN=... WEBUI_CLI_BIN=...`는 `scripts/webui/verify_installed_artifact.py`를 실행합니다. CI는 GB10 CUDA 기능 세트(`cuda,webui`, `MLX_CUDA_ARCHITECTURES=121`)로 `mlxcel-server`/`mlxcel`을 한 번 빌드하고, 격리된 CWD/HOME에서 배포되는 Rust 아티팩트를 검증합니다. 별도의 설치 헬퍼 타깃이 시작/RSS(`verify-webui-startup`), loopback 리버스 프록시/TLS 권한(`verify-webui-reverse-proxy`), 실제 단일 모델 인수(`verify-webui-single-model`)를 다룹니다. | Root PASS 증거: 설치된 HTTP+TLS/on/off/generated/feature-off는 `integration-1848/installed-f09-52103-both-tls-pass`, 재배치(relocated)/소스 및 아웃바운드 네트워크 차단 음성 대조군은 `integration-1848/offline-52103-pass`, 두 명령 모두에 대한 리버스 프록시는 `integration-1848/proxy-fbf7-52103-pass`, 시작/RSS는 `integration-1848/startup-52103-pass`, 단일 모델 4-arm Hello 캡처는 `integration-1848/single-model-52103-pass`에 있으며, 레거시 자체 SIGINT 반환값 `-2`가 기록되어 있고 drain 증명은 없습니다. | Root/CI는 여전히 최종 CI 아티팩트 URL과 바이너리 SHA-256/기능을 첨부해야 합니다. 세 엔진 전체를 대상으로 하는 CI는 보류 중이며 별도입니다. |
| fake 모델/다운로더 leaf를 사용하는 실제 보안 라우터 | `make verify-webui-integration-fake`는 ignore 처리된 정확한 Rust 테스트 `server::router_server::router_webui_playwright_harness_tests::real_router_browser_harness`를 실행합니다. 이 테스트는 실제 보안 Axum 라우터를 통해 임베드된 번들을 서빙하고, 전용 Playwright 설정만을 통해 `webui/tests/router-real.harness.ts`를 호출합니다. | root가 허가한 한정 실행 하에서 `ff4463c6`에서 PASS: `MLXCEL_WEBUI_ROUTER_ARTIFACTS=/private/tmp/mlxcel-router-restart-gap-2 cargo test --profile test-fast --features metal,accelerate server::router_server::router_webui_playwright_harness_tests::real_router_browser_harness -- --ignored --exact --nocapture`. 증거는 `/private/tmp/mlxcel-router-restart-gap-2/run-948b6f65-897b-4c31-a3d3-df88dd5b28ba/router-real-evidence.json`입니다. | Root는 원시 실행 디렉터리를 보관해야 합니다. 이는 fake 모델/다운로더 증거로 남으며 실제 체크포인트 행을 충족하지 않습니다. |
| 인증, 접두사(prefix), CSP, 캐시/오프라인, 악의적 라우트 | 설치된 아티팩트 스모크와 실제 라우터 하니스는 `/lab` 접두사, 비공개 bearer 인증, unsafe inline/eval이 없는 CSP, 실제 라우터 하니스에서 외부 브라우저 요청 없음, 인증되지 않은 401, UI 어댑터 및 레거시 `/v1/chat/completions`에 대한 악의적 Host/Origin/Fetch Metadata 403, 누락된 자산 404를 검증합니다. 설치된 아티팩트 스모크는 또한 Python 프록시를 비활성화하고, 상속된 `LLAMA_ARG_*`/`MLXCEL_*`/HF 스타일 환경 영향을 제거(sanitize)하며, 파이프 대신 파일에 로그를 쓰고, 정리 시 비공개 키 파일을 삭제합니다. | TLS/리버스 프록시 권한, 재배치/소스 분리, 아웃바운드 차단 음성 대조군에 대한 Root PASS 증거는 위에 명시된 installed/proxy/offline 아카이브에 있습니다. | 최종 PR 본문에 정확한 아티팩트 경로와 빌드 출처(provenance)를 보존하십시오. 리버스 프록시/TLS 증거는 설치된 아티팩트 증거이며 Vite 증거가 아닙니다. |
| UI 비활성화 회귀 | 설치된 아티팩트 스모크는 동일한 바이너리를 `--no-webui`로 시작하고, health는 계속 사용 가능하며 `/webui/`와 UI API는 사용할 수 없을 것을 요구합니다. | `not-run`입니다. 커밋된 게이트에 커버리지가 있습니다. | feature-disabled 빌드는 런타임 `--no-webui`로 암시되지 않으며 root/CI 증거로 남습니다. 릴리스 전에 해당 행을 추가하십시오. |
| 모델 없음 / 네트워크 없음 보장 | 설치된 아티팩트 스모크는 비어 있는 models/cache 루트를 사용하며 체크포인트를 로드하지 않습니다. 실제 라우터 하니스의 fake 다운로더는 소유한 임시 스냅샷만 구체화합니다. | `not-run`입니다. 설계는 skip-green이 아니라 fail-closed입니다. | Root는 빈 라우터 실행에서 provider 초기화가 없음을 검증하고 로그를 보관해야 합니다. |
| 다운로드 → 검사 → 로드 → 채팅 → Stop/drain → 언로드 → 제거 흐름 | 실제 라우터 브라우저 하니스는 UI 로그인/다운로드/로드/채팅, 활성 요청 drain을 관찰하는 로컬 Stop, 인증된 언로드 워커 종료 관찰, fake 모델/다운로더 leaf에 대한 제거를 사용합니다. | `not-run`입니다. 단위/브라우저 mock은 별도이며 단독으로는 불충분합니다. | Root의 실제 모델 증거는 dense, hybrid/MoE, VLM 이미지, 소규모 공개 다운로드를 포함해야 합니다. |
| 상태 완료 및 경쟁 상태(race) | 기존 계약/단위 스위트와 `verify-webui-rust`는 잘못된 전이, 작업 목록, 재시작/갭 픽스처, 다운로드 취소, 스키마 오류를 다룹니다. 실제 라우터 하니스는 추가로 빈 라이브러리 재스캔, 동일 포트에서 실제 새 서버 재시작 SSE, 이벤트 링 갭 SSE, 오래된 리비전 409, 알 수 없는 필드 400, 지원되지 않는 종료 상태 작업 취소를 프로덕션 보안 라우트에서 수행합니다. | `ff4463c6`에서 실제 라우터 fake 하니스에 대해 PASS입니다. 단위/통합 재실행은 root 소관으로 남습니다. | 실패한 시도와 정확한 커밋/바이너리 식별 정보를 최종 증거 표에 보존하십시오. |
| 브라우저 시각/접근성 엔진 | `pnpm --dir webui run browser`는 승인된 Chromium 픽셀 베이스라인/Chromium-AX 스위트를 유지하고, `pnpm --dir webui run browser:all`은 `playwright.engines.config.ts`를 사용하여 Chromium 전용 CDP나 스크린샷 베이스라인을 호출하지 않고 Chromium/Firefox/WebKit의 DOM, 키보드, 레이아웃, axe 커버리지를 제공합니다. CI는 세 엔진을 모두 설치하고 `verify-webui-frontend`를 통해 두 경로를 모두 실행합니다. | mock/Vite Chromium 증거에 대한 Root PASS: 기본 Chromium 테스트 28개, 교차 표면/성능 Chromium 테스트 7개가 `integration-1848/engines-f873-quiet-pass`, `chromium-prefix-default-pass`에 있으며, 최신 실제 DOM+paint 타이머 증거는 `integration-1848/engines-browser-only-timing-pass`에 있습니다. 311개 Vitest, 17개 Node, 49개 계약 테스트, typecheck, lint도 통과했습니다. | **설치된 아티팩트에서의 네이티브 Safari: 사용자 보고 PASS 2026-09-16**. Vite 프리뷰가 아니라 loopback으로 포워딩된 세션을 통해 임베드된 번들을 서빙하는 `mlxcel-server --webui`를 대상으로 했습니다. 범위는 유지관리자가 확인했으며 이 범위로만 한정됩니다. Models, Chat, Activity, Settings 전반의 화면 및 레이아웃 육안 검토, 보이는 포커스가 있는 키보드 Tab 및 Shift-Tab 순서와 Escape로 대화상자와 시트를 닫고 포커스가 연 요소로 복원됨, 390 px에서 가로 페이지 스크롤이 없는 Safari 네이티브 페이지 200% 확대, 증가된 대비에서도 가독성 유지. **수동 VoiceOver: 사용자 보류 2026-09-16, NOT RUN.** 이는 Safari 통과 범위에 포함되지 않았으며, CI의 axe 및 DOM 검사는 스크린 리더가 낭독하는 내용이 아니라 마크업을 검증하므로 다른 어떤 게이트도 이를 대체하지 못합니다. 유지관리자는 UI에 더 시급한 변경이 남아 있다는 이유로 보류했으며, 따라서 #1848이나 에픽을 차단하지 않습니다. 아직 검증되지 않은 범위는 VoiceOver가 기본 내비게이션과 선택된 라우트, 명령 대화상자 제목, 모델 combobox, 상태 배지, 스트리밍 요약, 오류 텍스트를 낭독하는지 여부입니다. 보류를 해제하려면 설치된 아티팩트에 대한 VoiceOver 세션이 필요하며, 어떤 자동화 게이트도 이 행을 닫을 수 없습니다. 세 엔진 전체 CI는 `WebUI bundle` 작업이 다룹니다. 이는 위 표면에 대해서만 2026-09-13 보고서를 대체하며, 해당 보고서는 `#gallery`의 Vite 프리뷰 하에서 디자인 시스템 후보를 다루었습니다. |
| 번들 예산 및 결정적 자산 | `make verify-webui-bundle`은 여전히 초기 JS, 전체 JS, 임베드된 번들 크기, 결정적으로 체크인된 자산, 소스 맵/로컬 경로/활성 외부 origin 없음, 라이선스 고지를 강제합니다. | 여기서는 `not-run`입니다. | Root는 통합 아티팩트의 최종 매니페스트 다이제스트와 번들 크기를 기록해야 합니다. |
| 성능 및 대용량 데이터 | 기존 프런트엔드 테스트는 1,000개 항목 카탈로그와 10,000 토큰 대화 파싱을 다룹니다. `make verify-webui-startup`은 설치된 빈 라우터의 시작/RSS를 기록합니다. `make verify-webui-activity-performance`는 체크포인트와 함께 실제 Activity 오버헤드/native-hidden 하니스를 실행하며, 승인된 호스트/디스플레이 설정으로 호출하면 실제 Linux GB10 headed Chrome의 최소화 → `document.hidden` 경로도 포함합니다. | Root PASS 증거: 설치된 시작/RSS는 `integration-1848/startup-52103-pass`, mock/Vite Chromium 실제 DOM+paint 타이밍은 `integration-1848/engines-browser-only-timing-pass`에 콜드 59.8 ms, 내비게이션 25.8 ms, 1,000개 항목 검색 25.9 ms, 10,000 토큰 새 대화 32.0 ms, CLS 0으로 기록되었습니다. 이전의 가시(visible) Activity 증거는 부분적이며 해당 소스 커밋에만 연결됩니다. GB10 작업은 이제 모든 변경에서 headed 가시 모드 관찰 오버헤드(`--perf-mode visible-only-headed`)를 실행하고 `--expect-hidden-native not-run`으로 요약하므로, 요약은 완전한 인수처럼 읽히지 않고 `status: incomplete` 및 `hidden_native: not-run`을 선언합니다. | **Native hidden 인수: 사용자 보류 2026-09-16.** 보류 범위는 최소화 → `document.hidden` 측정에 한정됩니다. headed 가시 모드 오버헤드는 CI에서 측정되며 모든 수치 예산이 여전히 적용됩니다. 추정이 아니라 GB10 러너에서 확인된 이유는 다음과 같습니다. Xvfb와 openbox는 headed Chromium에 실제 700x900 콘텐츠 영역을 제공하고, 브라우저는 창 상태를 `minimized`로 보고하지만 `document.visibilityState`는 15초 동안 `visible`로 유지됩니다. 같은 컨텍스트의 형제 탭을 전면으로 가져와도 `document.hidden`은 false로 남습니다. 가시성을 억제하는 두 실행 인자(`--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`)를 모두 제거했고, 실행 중인 브라우저 프로세스의 인자를 읽어 제거를 검증했습니다. 이 연쇄에서 검증되지 않은 단계는 Playwright의 형제 페이지가 같은 창의 탭인지 별도의 창인지 여부이며, 별도 창이라면 탭 데이터는 정보를 주지 못합니다. 이는 러너나 제품이 아니라 자동화 브라우저의 속성입니다. 이 행이 보호하려는 가시성 backoff는 `webui/src/state/sync.test.ts`의 단위 테스트 두 개와 `webui/tests/browser.spec.ts`의 `visibilitychange` 케이스가 계속 다룹니다. 보류를 해제하려면 실제 창 가시성 변경을 전파하는 브라우저에서 `--perf-mode full`을 다시 실행해야 합니다. |
| 실제 추론/미디어 | 어떤 커밋된 mock 게이트도 실제 추론을 충족할 수 없습니다. 승인된 Chat source9385 dense/hybrid/VLM 증거는 소스가 baseline6976까지 동일하게 유지된 특정 dense/hybrid/VLM 생성 경로에 대해서만 참조할 수 있습니다(`git diff --name-only 9385bcb7 6976e1df -- src Cargo.toml Cargo.lock build.rs webui/src webui/package.json webui/pnpm-lock.yaml` 결과가 비어 있었음). 이는 해당 경로에 대한 소스 동등성 증거이며, 동일 바이너리 또는 현재 브랜치 전체에 대한 런타임/UI 주장이 아닙니다. #1848에는 프로덕션 URL 접두사/공개 health 변경(`d6b7`, `c629`)과 테스트 하니스 변경도 포함됩니다. | Root 단일 모델 설치 4-arm Hello 캡처 PASS는 `integration-1848/single-model-52103-pass`에 있으며, 의미 검토 범위는 해당 소규모 실제 체크포인트 프롬프트로 한정되고 레거시 자체 SIGINT `-2`가 기록되어 있습니다. 소스 동등한 Chat dense/hybrid/VLM 행은 위 제한 사항과 함께만 승인된 상태로 남습니다. | Root는 여전히 소규모 공개 다운로드, VLM 이미지, 진정한 native-hidden Activity 오버헤드에 대한 정확한 행을 `MLXCEL_REQUIRE_MODELS=1`로 실행하거나 참조해야 하며, 체크포인트 리비전, 출력, 수명 주기 순서, 리소스 전/후를 기록해야 합니다. |
| CUDA 호스트 스모크 | `make verify-webui-cuda WEBUI_CUDA_EVIDENCE=...`는 UI-on, UI-off, 설치된 아티팩트 행에 대한 root 실행 CUDA 증거 JSON을 검증합니다. | `not-run`입니다. macOS로는 충족되지 않습니다. | GB10/CUDA 호스트는 UI-on/UI-off 서버 스모크를 실행하고 결과를 기록해야 합니다. 과거의 장애 면제는 포괄적 허가가 아닙니다. |
| Settings / Chat / Activity 교차 표면 통합 | 기존의 병합된 하위 게이트가 소스별 증거를 제공합니다. 이 매트릭스는 허구의 단일 실행 통과를 상속하지 않고 최종 재실행을 가리킵니다. | 이전 root 메모: Models source22f19 수명 주기/프로필/내비게이션/제거는 generationTested=false로 통과했고, Chat source9385 dense/hybrid/VLM 의미론, Stop active→0, CSP, 언로드가 통과했으며, Activity 가시 쌍(paired) 오버헤드는 부분 통과했고 native hidden은 실행되지 않았습니다. | 최종 통합 아티팩트는 승인된 이전 증거를 재실행하거나 커밋/소스와 제한 사항을 명시하여 참조해야 합니다. Chat 컴포저는 새 턴을 100으로 제한하지만 기록 가져오기/저장은 대화당 200을 허용합니다. Stop은 요청별 취소 영수증이 아니라 집계 Activity를 새로 고칩니다. |
| 호환성 엔드포인트 / CLI 매니페스트 | `verify-webui-rust`, 설치된 아티팩트 스모크, `verify-llama-compat`, feature-off root 빌드가 함께 호환성 표면을 다룹니다. | 여기서는 `not-run`입니다. | Root는 `mlxcel-server`와 `mlxcel serve`, UI-on/UI-off, 단일 모델과 빈 라우터, 생성된/명시적 키, 접두사, 지원되는 경우 TLS/프록시를 포함해야 합니다. |
| 보안 및 개인정보 | 계약/프런트엔드/서버 테스트는 토큰 마스킹, 토큰 비영속화, 기록/이미지 동의, 본문 크기 제한, XSS/Markdown 정제(sanitization), UI 어댑터 오류를 다룹니다. 실제 라우터/설치된 아티팩트 게이트는 프로덕션 CSP/인증/레거시 라우트 공격을 추가합니다. | 여기서는 `not-run`이며, 활성화된 placeholder는 통과로 표시되지 않았습니다. | 보안 PR 지적 사항은 에픽 종료 전에 정확한 테스트/증거 행으로 표현되어야 합니다. |
| 최종 게시 | PR 본문과 최종 보고서는 이 매트릭스를 링크하고 모든 `not-run`, `blocked`, 사용자 보류 항목을 나열해야 합니다. | 이 브랜치가 매트릭스를 소유하며, 보다 폭넓은 사용자 가이드/문서는 #1849가 소유합니다. | 상위 에픽은 필수 게이트가 검증되거나 사용자가 구체적인 날짜/범위와 함께 명시적으로 보류하기 전에는 닫혀서는 안 됩니다. |

## 증거 행 스키마

각 최종 증거 행에는 요구사항 id, 소스 커밋, 바이너리 SHA-256/기능, UI 매니페스트 소스와 번들 다이제스트, OS/하드웨어/브라우저, 해당하는 경우 체크포인트 리비전, 명령 또는 하니스 버전, 결과(`passed`, `failed`, `not-run`, `blocked`, 또는 명시적 사용자 보류), 원시 증거 경로, 필요한 경우 사람의 시각/수동 검토자, 소유자, 잔여 제한 사항을 기록하십시오.

## 소스 동결 후 필요한 root 명령

순수 validator 자체 테스트:

```bash
make verify-webui-helper-tests
```

결정적 통합 게이트:

```bash
make verify-webui
make verify-webui-integration-fake
make verify-webui-installed WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_FEATURE_OFF_SERVER_BIN=/path/to/feature-off/mlxcel-server WEBUI_FEATURE_OFF_CLI_BIN=/path/to/feature-off/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_FEATURE_OFF_BUILD_SOURCE_HEAD=<40-hex> WEBUI_INSTALLED_EVIDENCE=/path/to/webui-installed-evidence.json
make verify-webui-startup WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_BUILD_FEATURES=metal,accelerate,webui WEBUI_STARTUP_EVIDENCE=/path/to/startup-evidence.json
make verify-webui-reverse-proxy WEBUI_SERVER_BIN=/path/to/mlxcel-server WEBUI_CLI_BIN=/path/to/mlxcel WEBUI_BUILD_SOURCE_HEAD=<40-hex> WEBUI_REVERSE_PROXY_EVIDENCE=/path/to/reverse-proxy-evidence.json
make verify-test
make verify-clippy
make verify-fmt
```

validator 전용 선언이 아니라 기존 하니스를 사용하는 root 소유의 실제 모델 및 CUDA 증거 명령:

```bash
# For each accepted dense, hybrid/MoE, and VLM checkpoint row, root starts an isolated bundled server with the model already loaded, then runs the existing real chat browser harness and stores its JSON/PNG artifacts referenced by hardware-evidence.json.
MLXCEL_CHAT_REAL_ISOLATED=1 \
MLXCEL_CHAT_REAL_URL=http://127.0.0.1:<port>/<prefix>/webui/ \
MLXCEL_CHAT_REAL_KEY_FILE=/path/to/private-0600-key \
MLXCEL_CHAT_REAL_MODEL_ID=mdl_... \
MLXCEL_CHAT_REAL_IMAGE_FILE=/path/to/image.png \
  pnpm --dir webui exec playwright test --config playwright.chat-real.config.ts

# For the small public download and native-hidden/startup rows, root runs the approved actual-router/download and Activity native-hidden/startup harnesses on the scheduled host and records raw JSON/log/PNG artifacts.
MLXCEL_REQUIRE_MODELS=1 \
WEBUI_SERVER_BIN=/path/to/mlxcel-server \
WEBUI_CLI_BIN=/path/to/mlxcel \
WEBUI_SINGLE_MODEL=/path/to/checkpoint \
WEBUI_BUILD_SOURCE_HEAD=<40-hex> \
WEBUI_BUILD_FEATURES=metal,accelerate,webui \
WEBUI_SINGLE_MODEL_OUTPUT_DIR=/path/to/single-model-output \
WEBUI_SINGLE_MODEL_STARTUP_REPORT=/path/to/startup-evidence.json \
  make verify-webui-single-model

MLXCEL_REQUIRE_MODELS=1 \
WEBUI_SERVER_BIN=/path/to/mlxcel-server \
WEBUI_ACTIVITY_MODEL=/path/to/checkpoint \
WEBUI_ACTIVITY_CHECKPOINT_REVISION=<checkpoint-revision> \
WEBUI_BUILD_SOURCE_HEAD=<40-hex> \
WEBUI_BUILD_FEATURES=metal,accelerate,webui \
WEBUI_ACTIVITY_EVIDENCE=/path/to/activity-evidence.json \
  make verify-webui-activity-performance

# Then validate that every required row is present, passed, typed, SHA-identified, and backed by existing artifact files.
MLXCEL_REQUIRE_MODELS=1 make verify-webui-hardware WEBUI_HARDWARE_EVIDENCE=/path/to/hardware-evidence.json

# On the GB10/CUDA host, root builds/runs UI-on/UI-off installed artifact smokes and records artifacts, then validates them.
make verify-webui-cuda WEBUI_CUDA_EVIDENCE=/path/to/cuda-evidence.json
```

`verify-webui-hardware`와 `verify-webui-cuda`는 root가 제공한 증거 JSON을 검증하며, 증거가 누락되었거나 불완전하거나 통과하지 않았거나, 중복 행이 있거나, `source_commit`/`binary_sha256`이 유효하지 않거나, 필드 타입이 잘못되었거나, 실행 가능한 명령이 없거나, 존재하지 않는 아티팩트 경로가 있으면 fail-closed로 실패합니다. 이들은 root가 실행하는 하드웨어/CUDA 명령 자체를 대체하지 않습니다.

## 설치된 아티팩트 작업을 위한 GB10 러너 참고 사항

`.github/workflows/ci.yml`의 `WebUI installed artifact` 작업은 개발 호스트이기도 한 self-hosted GB10 러너에서 실행됩니다. 그곳의 두 가지 동작은 의도된 것입니다(lablup/mlxcel#2111).

**호스트 GPU 락.** 러너 호스트의 개발 세션은 `gpu-lock`(`/usr/local/bin/gpu-lock`, `/tmp/gpu-lock-$(id -u)/lock`에 대한 flock)으로 GPU 작업을 직렬화합니다. 러너 서비스는 `PrivateTmp=no`로 동일한 uid에서 실행되므로 이 락 파일을 볼 수 있습니다. Activity 성능 게이트는 세 개의 락을 순서대로 획득합니다. `$HOME/.cache/mlxcel/ci-locks` 아래의 협력적 CI flock, 그다음 `gpu-lock run --tag webui-activity --wait 600`, 그다음 아래에 설명된 fail-closed 호스트 사전 점검입니다. 작업 로그는 대기 전에 락 보유자를 출력하고 게이트 본문이 시작되면 대기한 초를 출력합니다. 600초 내에 `gpu-lock`을 얻지 못하면 해당 단계는 fail-closed로 실패하고 `gpu-lock status`를 출력합니다. `gpu-lock`이 없는 러너에서는 게이트가 중간 락을 건너뛰고 이전처럼 동작합니다. `gpu-lock` 하에서 게이트의 프로세스는 `oom_score_adj=1000`을 가지며, 이는 `gpu-lock`이 실행하는 모든 GPU 작업에 설정합니다. 러너가 대기하기를 원하는 개발 세션은 전체 GPU 실행 동안 `gpu-lock`을 보유해야 합니다. `gpu-lock` 밖에서 시작된 GPU 작업은 여전히 사전 점검의 GPU 검사에서 게이트를 실패시킵니다.

**호스트 사전 점검 및 정리(reap).** 게이트 본문은 `scripts/webui/activity_gate_host.py precheck`로 시작합니다(lablup/mlxcel#1949). 프로세스는 실행 파일이 `$RUNNER_TEMP/mlxcel-webui-installed/` 안에 있거나, 작업 디렉터리 또는 `HOME`이 verifier 실행 디렉터리(`$RUNNER_TEMP/mlxcel-activity-performance-*`; verifier는 서버, node, Playwright가 시작하는 브라우저에 `HOME=<run directory>/home`을 부여하고 Playwright는 브라우저를 자체 프로세스 그룹에서 시작함)일 때 작업의 식별 정보를 가진 것으로 간주합니다. 개발자의 러너 셸처럼 제어 터미널이 있는 프로세스는 절대 건드리지 않으며, 디렉터리 또는 `HOME`으로만 일치한 프로세스는 그룹이 아니라 단독으로만 시그널을 보냅니다. 사전 점검은 먼저 이전 실행이 누수시킨 해당 식별 정보의 프로세스를 중지합니다. 고아(부모 PID 1)이거나, 디렉터리 또는 실행 파일이 `(deleted)`로 읽히거나 실행 디렉터리가 사라진 경우이며, 이는 `RUNNER_TEMP`의 모든 것이 이를 만든 작업이 끝나면 그렇게 되기 때문입니다. 해당 식별 정보를 가진 살아 있는 프로세스나 다른 uid가 소유한 누수는 게이트를 실패시킵니다. 그다음 대기 전에 GPU 컴퓨트 프로세스가 남아 있으면 fail-closed로 실패하여 충돌을 즉시 보고합니다. 10개의 1초 샘플에 걸친 `/proc/loadavg`의 평균 실행 가능 작업 수가 CPU 수의 4분의 1 이하가 될 때까지 최대 600초 기다리며, 호스트가 계속 바쁘면 호스트 스냅샷과 함께 fail-closed로 실패합니다. 러너에서의 동시 컴파일은 이전 점수를 2.8~3.9퍼센트 움직였으며 이는 2퍼센트 예산을 초과합니다. Load average와 `ps %cpu`는 갓 시작된 컴파일에 뒤늦게 반응하므로 사용하지 않습니다. 대기 후에 GPU를 한 번 더 확인합니다. GPU 프로세스는 pid, 프로세스 이름, 메모리, 부모, uid와 함께 명명되며, 이 공개 저장소의 로그와 아티팩트는 누구나 읽을 수 있으므로 실행 파일과 작업 디렉터리는 작업의 식별 정보를 가진 경우에만 출력됩니다. 관찰한 상태(load average 및 컴파일러 pid가 포함된 시작과 끝의 스냅샷, 조용함 대기 샘플, 정리한 모든 것, 모든 거부)는 `webui-activity-host-state.json`에 기록되고, 출력이 검증되기 전에 취득한 `host_state.after_activity` 로드 샘플과 함께 `host_state.gate_start`로 전체 증거에 임베드되며 진단과 함께 업로드됩니다. 요약 증거에는 포함되지 않습니다. verifier는 시작하는 모든 프로세스(Xvfb, openbox, 서버, node activity 스크립트)를 시작 시간 및 boot id와 함께 `$RUNNER_TEMP/webui-activity-gate.pid`에 기록하며, `Reap Activity gate processes` 단계는 `if: always()`로 `activity_gate_host.py reap`을 실행합니다. 이 단계는 pid 파일이 지정한 프로세스 그룹(시작 시간이 바뀐 pid나 재부팅 이전의 기록은 절대 대상으로 삼지 않음)과 작업 식별 정보를 가진 그 외의 모든 것을 중지하고, 시그널을 보내야 했던 대상을 출력하며, 무엇인가 살아남거나 pid 파일을 읽을 수 없으면 작업을 실패시킵니다(식별 정보 스윕은 그래도 실행됨). 통과한 실행에서는 아무것도 찾지 못하며, 요약이 요구하는 정상 종료(`exit_code` 0, `forced` false, `process_group_empty` true)는 변경되지 않습니다.

**증분 컴파일 없음.** 이 작업은 자체 빌드에 `CARGO_INCREMENTAL=0`을 설정합니다. 영속 타깃 디렉터리 `$HOME/.cargo-target/mlxcel-webui-installed-ci`는 한때 하나의 codegen 단위에서 정의되지 않은 `serde_json` 제네릭으로 연속 다섯 번 링크에 실패하는 증분 상태(실행 36988516810)를 보유했고, 해당 상태를 지운 후에야 통과했습니다. 증분 설정이 크레이트 fingerprint의 일부이므로 이 변경 후 첫 실행에서는 워크스페이스 크레이트를 한 번 다시 빌드하고, 이후 실행은 다시 warm 상태가 됩니다. 변경 이전에 남은 증분 상태는 더 이상 읽히지 않으므로 디스크 공간을 회수하기 위해 `$HOME/.cargo-target/mlxcel-webui-installed-ci/test-fast/incremental`을 수동으로 삭제해도 됩니다. 다른 GB10 타깃 디렉터리는 `test-fast` 프로필 기본값을 유지합니다.
