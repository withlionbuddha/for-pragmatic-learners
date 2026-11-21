# Quantum Python experiments

이 폴더는 Quantum 실험 코드와 결과 문서를 포함합니다. 전체 실험 계획 및 보고 내용은 [저장소 루트 README](../README.md#quantum)를 참조하십시오.

## Python 환경

```bash
cd quantum
python -m pip install -e '.[dev]'
python -m pytest tests
# 또는
bash scripts/exe_tests.sh
```

Python 패키지는 `src/modules/`, 단위 테스트는 `tests/`, 실험 Notebook은 `notebooks/`에 있습니다. Notebook은 `quantum/` 또는 `quantum/notebooks/`에서 실행할 수 있습니다.

## 실험 자료

- [2주 실험 원본 보고서](QUANTUM_COMPUTING_2WEEK_EXPERIMENT_PROJECT.pdf)
- [후속 실험 계획 v0.1.0](optimizer_experiment_plan_v.0.1.0.pdf)
- [Adaptive vs Fixed COBYLA](notebooks/adaptive_Q_cobyla_vs_Q_cobyla.ipynb)
- [θ 그룹 크기 비교](notebooks/adaptive_theta_group_size_comparison.ipynb)
