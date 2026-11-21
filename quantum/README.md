# Quantum experiments

Python project layout: `src/modules/` contains the importable package, `tests/` contains unit tests, `notebooks/` contains experiments, and `scripts/` contains the test runner.

From `quantum/` install with `python -m pip install -e .` and run `python -m pytest` (requires pytest). Alternatively run `bash scripts/exe_tests.sh`.

The notebooks prepend `quantum/src` to Python's module search path when started from the project root or its notebook directory.

## Quantum — 2주 실험 계획 및 결과

### 실험 목적

동일한 문제와 평가 예산에서 고전적 최적화와 양자·하이브리드 최적화 접근을 비교하여 **성능, 실행시간, 자원 사용량 및 재현성**을 검증합니다. 시뮬레이터 기반 결과와 실제 양자 하드웨어의 성능은 구분합니다.

### Plan (계획)

- **원본 실험 계획서:** [QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf](QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf)
- **계획의 목적과 검증 기준:** [quantum.README.md](quantum.README.md)
- **최적화 실험 계획 개정본:** [v0.0.1](optimizer_experiment_plan_v.0.0.1.pdf) · [v0.0.2](optimizer_experiment_plan_v.0.0.2.pdf) · [v0.1.0](optimizer_experiment_plan_v.0.1.0.pdf)
- **실험 코드:** [src/modules/optimizers/](src/modules/optimizers/) — Quantum Direction COBYLA, Adaptive Quantum Direction COBYLA 및 방향 샘플러.
- **주요 비교 항목:** 동일 데이터·초기화·난수 시드·손실 평가 예산을 전제로 한 모델별 검증 MSE, 실행시간, 평가 호출 횟수 및 메모리 사용량.
- **추가 실험:** Adaptive 방식의 `theta_group_size` 변화에 따른 θ 개수·메모리·방향 샘플 차이 비교.

### Result (결과 및 검증 상태)

| 항목 | 저장소에 확인되는 산출물 | 현재 확인 상태 |
| --- | --- | --- |
| Adaptive vs Fixed COBYLA | [비교 Notebook](notebooks/adaptive_Q_cobyla_vs_Q_cobyla.ipynb) | 실험 코드와 Notebook 존재. 실행 결과의 통계적 유효성·성능 우위는 별도 검증 필요 |
| θ 그룹 크기 비교 | [그룹 비교 Notebook](notebooks/adaptive_theta_group_size_comparison.ipynb) | 그룹 크기별 비교 및 방향·θ 진단 코드 존재. 확정적 비교 수치는 미기재 |
| 메모리 측정 | [독립 프로세스 실험 실행기](src/modules/profilers/group_experiment_worker.py) | 실험별 프로세스 분리 코드 구현. 재현 가능한 수치 검증 필요 |
| 단위 테스트 | [tests/units/optimizers/](tests/units/optimizers/) | 테스트 코드 존재. 현재 README 작성 과정에서는 테스트 미실행 |
| 2주 프로젝트 최종 결과 | [원본 PDF](QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf) | **원본 PDF 본문의 측정 결과는 확인되지 않음**. 결과로 단정하지 않음 |

> **해석상 주의:** 위 표는 저장소에서 확인된 *산출물과 검증 상태*를 기록한 것입니다. 실제 실행에서 얻은 수치, 개선율, 통계적 유의성은 재현된 측정 자료가 확보되기 전까지 성과로 주장하지 않습니다. 원본 PDF의 세부 내용도 본문 확인 없이 요약하거나 결과를 추정하지 않습니다.

### 재현 방법

```bash
cd quantum
python -m pip install -e .
python -m pip install pytest
python -m pytest tests
```

Notebook은 [notebooks/](notebooks/)에서 실행하며, 모델별·시드별 결과를 확인할 때는 실행환경, 패키지 버전, 원시 측정값 및 실행 여부를 함께 기록합니다.
