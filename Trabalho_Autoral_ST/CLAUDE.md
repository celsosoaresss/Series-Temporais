# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Course assignment (Séries Temporais, in Portuguese — keep code/docs in Portuguese): **Self-Supervised Learning for Time Series** — a BERT-style masked-reconstruction Transformer that imputes missing values in time series, pretrained on the UCI ElectricityLoadDiagrams20112014 dataset and designed to be re-evaluated on other datasets. Not a git repository.

## Pipeline

Run the notebooks in order; each consumes the previous one's artifacts:

1. `01_preprocessamento.ipynb` — loads `LD2011_2014.txt`, resamples 15min→1h, keeps 2012–2014, selects 100 clients (seed 42), temporal train/val/test split (2012–13 / 2014-S1 / 2014-S2), per-series z-score with **train-only stats**, windows of 168h → writes `dados_processados/` (janelas.npz, normalizador.json, config_dados.json)
2. `02_treinamento.ipynb` — trains `BERTImputador` (masked MSE, point+block masking, early stopping) → writes `resultados/melhor_modelo.pt`, `historico.json`
3. `03_avaliacao.ipynb` — fixed-seed evaluation masks (seed 123), 6 scenarios (pontual/bloco × 10/25/50%), baselines comparison → `resultados/metricas_teste.{csv,json}`; last section is the template for applying the method to a new dataset

All reusable logic lives in `src/` (notebooks only orchestrate): `dados.py` (load/split/normalize/window — generic over any datetime-indexed DataFrame), `mascaramento.py` (point + geometric block masking, vectorized), `modelo.py` (BERTImputador: bidirectional Transformer encoder, learnable [MASK] token, sinusoidal PE computed dynamically so any window length works), `treino.py` (seeded training loop), `avaliacao.py` (reproducible evaluation protocol), `baselines.py`, `estilo.py` (matplotlib palette).

Reproducibility is a hard requirement: everything seeded, configs saved alongside artifacts, evaluation masks deterministic. Loss/metrics are always computed only on masked positions.

## Commands

```powershell
# execute a notebook headlessly, saving outputs in-place
python -m nbclient  # not a CLI here; use the runner pattern:
python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('01_preprocessamento.ipynb', as_version=4); NotebookClient(nb, timeout=5400).execute(); nbformat.write(nb,'01_preprocessamento.ipynb')"
```

**Gotcha:** plain `pip` on this machine belongs to a conda env (`pat`, Python 3.10) — always use `python -m pip install ...` so packages land in the Python 3.12 that has CUDA-enabled torch.

## Environment

- Python 3.12.5 at `C:\Users\celso\AppData\Local\Programs\Python\Python312` (no venv); GPU: GTX 1080 Ti (CUDA available)
- Key packages: torch 2.7.1+cu118, pandas 2.2.2, numpy 2.1.0, matplotlib 3.10.3, nbformat/nbclient; **statsmodels not installed**

## Dataset: LD2011_2014.txt

UCI ElectricityLoadDiagrams20112014 (~711 MB — never read whole into context): 140,256 rows × 371 cols (timestamp + clients MT_001–MT_370), 15-min kW readings 2011–2015. Semicolon separator, **comma decimal**. Leading zeros = client not yet installed (missing, not zero consumption). Load via `src.dados.carregar_ld2011` (takes minutes).
