# Quantum experiments

Python project layout: `src/modules/` contains the importable package, `tests/` contains unit tests, `notebooks/` contains experiments, and `scripts/` contains the test runner.

From `quantum/` install with `python -m pip install -e .` and run `python -m pytest` (requires pytest). Alternatively run `bash scripts/exe_tests.sh`.

The notebooks prepend `quantum/src` to Python's module search path when started from the project root or its notebook directory.

## Quantum — 2주 프로젝트

### Plan — 2주 실험 계획

- [2주 양자 컴퓨팅 실험 계획 및 프로젝트 보고서(PDF)](QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf)
- [프로젝트 목적과 검증 원칙](quantum.README.md)

2주 실험의 계획은 위 자료를 기준으로 구분합니다. PDF 본문에서 계획과 결과에 해당하는 페이지 범위는 아직 확인되지 않았으므로 임의로 나누어 서술하지 않습니다.

### Result Report — 2주 프로젝트 결과 보고

- [2주 프로젝트 결과 보고서(PDF)](QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf)

현재 계획과 결과에 사용되는 원본 PDF는 동일 파일입니다. 결과의 세부 지표와 해석은 PDF 원문을 참조하며, 본문 미확인 상태에서 수치나 결론을 별도로 단정하지 않습니다.

## 후속 연구 — 양자 방향 최적화

아래 내용은 **2주 프로젝트의 계획 및 결과가 아닌 이후 진행하는 별도 연구**입니다.

### 후속 실험 계획

- [최적화 실험 계획 v0.0.1](optimizer_experiment_plan_v.0.0.1.pdf) · [v0.0.2](optimizer_experiment_plan_v.0.0.2.pdf) · [v0.1.0](optimizer_experiment_plan_v.0.1.0.pdf)
- **Adaptive Quantum Direction COBYLA 비교:** 동일 데이터, 초기화, 난수 시드 및 손실 평가 예산 아래 Fixed Quantum Direction COBYLA와 검증 MSE, 실행시간, 평가 횟수 및 메모리를 비교합니다.
- **θ 그룹 크기 실험:** `theta_group_size`별 θ 개수, 메모리, θ 이동량 및 샘플 방향의 차이를 비교합니다.

### 후속 실험 산출물 및 상태

| 실험 | 관련 산출물 | 상태 |
| --- | --- | --- |
| Adaptive vs Fixed COBYLA | [비교 Notebook](notebooks/adaptive_Q_cobyla_vs_Q_cobyla.ipynb) | 실험 코드 존재. 결과의 재현성과 통계적 유효성은 미검증 |
| θ 그룹 크기 비교 | [그룹 비교 Notebook](notebooks/adaptive_theta_group_size_comparison.ipynb) | 진단 코드 존재. 확정적인 수치 비교는 미기재 |
| 독립 프로세스 메모리 측정 | [실험 실행기](src/modules/profilers/group_experiment_worker.py) | 구현 코드 존재. 측정 결과 검증 필요 |
| 단위 테스트 | [테스트 코드](tests/units/optimizers/) | 코드 존재. 이 문서 수정 시 테스트 미실행 |

### 실행 방법

```bash
cd quantum
python -m pip install -e .
python -m pip install pytest
python -m pytest tests
```
