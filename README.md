# SpaNav EEG Analysis

Analysis of EEG data from the SpaNav study, investigating the effects of transcranial temporal interference stimulation (tTIS) on spatial navigation in participants healthy and with history of traumatic brain injury (TBI).

## Study overview

The SpaNav study is a collaboration between HES-SO Sion and EPFL Lausanne (Switzerland). It combines:

- **Virtual reality (VR)** spatial navigation tasks
- **EEG** recordings to capture neural activity
- **tTIS** (transcranial temporal interference stimulation), a non-invasive brain stimulation technique

The goal of this analysis is to characterize how tTIS affects neural oscillations related to spatial navigation in TBI patients compared to healthy controls.

## Analysis pipeline

The codebase is organized as a pipeline from raw EEG to statistical inference:

1. **Preprocessing** — EEG cleaning, artifact rejection (via `autoreject`), epoching, and alignment with behavioral data
2. **Processing** — Power spectral density (PSD) and time-frequency representations (TFR) of EEG signals, oscillatory features extracted via FOOOF modeling of the PSD
3. **Analysis** — Mass-univariate statistics with cluster-based permutation tests, linear mixed models (LMM) in an integrated Python and R pipeline
4. **Visualization** — Raw PSD, oscillatory features, TFR

## Data availability

Raw data from the SpaNav study is not included in this repository and is not yet publicly available.

## Setup

Requires Python 3.12.

### Configuration

Before running any scripts, create your personal `config.ini` from the provided template:

```bash
cp config.template.ini config.ini
```

Edit `config.ini` to match your local setup (e.g., data paths). This file is not tracked by git.

### Dependencies

Using pipenv:

```bash
pipenv install
```

Or using pip:

```bash
pip install -r requirements.txt
```

## License

MIT