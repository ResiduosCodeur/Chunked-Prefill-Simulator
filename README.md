# Chunked Prefill Simulator

A simulator that compares **normal prefill** and **chunked prefill** for LLM inference, using latency and throughput metrics. It is based on the SARATHI paper ([arXiv:2308.16369](https://arxiv.org/abs/2308.16369)).

---

## 1. Project overview

### The problem

LLM inference has two phases:

- **Prefill**: the whole prompt is processed in parallel. It keeps the GPU busy (compute-bound).
- **Decode**: tokens are generated one at a time. The GPU is mostly idle waiting for memory (memory-bound).

### The two methods compared

| | Normal prefill | Chunked prefill (SARATHI) |
|---|---|---|
| Prefill | Whole prompt in one pass | Prompt split into chunks of size `C` |
| Batch contents | Prefill-only **or** decode-only | One prefill chunk **plus** one token for every running decode |
| Decodes during a prefill | Stall until it finishes | Keep running ("piggyback" on the chunk) |
| Expected effect | Long stalls, idle GPU during decode | Higher throughput, steadier token latency, slightly slower TTFT for an idle server |

### How the simulation works

The simulator models one GPU serving a stream of requests, one forward pass (iteration) at a time. The duration of each pass comes from a **cost model**:

- **Linear layers:** time is the larger of compute time and weight-loading time (roofline model). Token counts are rounded up to a multiple of the tile size (128) to model tile quantization.
- **Attention:** prefill attention is compute-bound and re-reads the KV cache of earlier chunks. Decode attention is memory-bound and reads each request's KV cache.
- **Calibration:** the hardware constants are tuned so the model matches Table 2 of the paper (LLaMA-13B on an A6000: about 0.229 ms per prefill token and about 50 ms for a decode-only batch of 4 at 1K context).

### Metrics

| Metric | Meaning |
|---|---|
| **TTFT** | Time to first token: first token time minus arrival time (ms) |
| **Throughput** | (prefill + decode tokens) / total run time (tokens/ms) |
| **TBT** | Time between consecutive output tokens (ms), reported as mean and P99 |

### Experiments and graphs

| # | Experiment | Output graph |
|---|---|---|
| 1 | TTFT of a single request vs prompt length (chunking overhead) | `exp1_ttft_vs_prompt.png` |
| 2 | Throughput gain vs prefill:decode ratio (paper Fig. 9) | `exp2_throughput_vs_pd_ratio.png` |
| 3 | Throughput gain vs batch size at balanced P:D | `exp3_throughput_vs_batch.png` |
| 4 | Online serving under rising load: throughput, TTFT, TBT | `exp4_online_load.png` |
| 5 | Decode stalls (iteration timeline) and TBT/TTFT CDFs | `exp5_stalls_and_cdfs.png` |
| 6 | Chunk-size sweep (tile-quantization effect) | `exp6_chunk_size_sweep.png` |

### Key observations (from the simulator's default settings)

- The throughput gain peaks when the number of prefill chunks matches the number of decode iterations, at **P:D ≈ C / (B − 1)**. This matches the paper's analysis.
- Chunking slightly **increases TTFT on an idle server** because of repeated KV-cache reads, and the effect is larger for small chunks.
- **Under load**, chunked prefill reduces mean TTFT and sharply reduces P99 time-between-tokens, because decodes are no longer stalled behind long prefills.

> Numbers come from an analytical simulator, not from measurements on a real GPU. Trends match the paper, but absolute values depend on the constants in `default_config.m`.

---

## 2. Repository structure

```
chunked-prefill-simulator/
├── MATLAB/                      # Simulator + graph generation
│   ├── run_experiments.m        # Entry point: runs experiments, saves CSVs and graphs
│   └── src/
│       ├── default_config.m     # Hardware, model and scheduler settings
│       ├── cost_model.m         # Latency of one forward pass
│       ├── make_workload.m      # Generates requests (arrival, prompt, output lengths)
│       ├── max_batch_size.m     # How many requests fit in GPU memory
│       ├── simulate.m           # Normal and chunked schedulers
│       └── compute_metrics.m    # TTFT, throughput, TBT from a simulation result
├── src/                         # Python port 
│   └── schedulers/              # base / normal / chunked schedulers
├── experiments/                 # Python experiment runner 
├── configs/
│   └── default.yaml             # Shared configuration for the Python version
├── tests/                       # 
├── results/
│   ├── raw/                     # CSV output of the experiments
│   └── graphs/                  # PNG output graphs of the experiments
|   └── sample_graphs/           # contains sample graphs
├── Dockerfile                   # 
├── docker-compose.yml           # 
├── requirements.txt             # Python dependencies
├── README.md
└── .gitignore
```

---

## 3. What each file does

### MATLAB code (`MATLAB/`)

| File | Purpose |
|---|---|
| `run_experiments.m` | The only file you run. It builds workloads, runs both schedulers, collects metrics, writes CSVs to `results/raw/` and saves graphs to `results/graphs/`. Accepts experiment numbers, for example `run_experiments(2)`. |
| `src/default_config.m` | Returns a settings struct: GPU speed, memory, model size, maximum batch size, chunk size, and whether to align chunk size to the tile size. Edit this to simulate different hardware. |
| `src/cost_model.m` | Given the prefill chunks and decode requests in a batch, returns the time of one forward pass. All performance behavior comes from here. |
| `src/make_workload.m` | Creates `n` requests with arrival times (all at 0 or Poisson), and prompt and output lengths from a total length and a P:D ratio. Lengths can be fixed or Zipf-distributed. |
| `src/max_batch_size.m` | Computes the largest batch whose KV caches fit in GPU memory: `B = floor((M_gpu − M_model) / (L · m_kv))`. |
| `src/simulate.m` | The simulator core. Admits requests first-come-first-served up to the batch limit, then runs either the `'normal'` or `'chunked'` scheduler iteration by iteration and records when each token was produced. |
| `src/compute_metrics.m` | Converts a simulation record into TTFT, throughput and TBT statistics (mean, P50, P99), plus raw samples for CDF plots. |

### Other folders and files

| Path | Purpose |
|---|---|
| `src/`, `src/schedulers/` | Reserved for the Python port: `models.py`, `workload.py`, `cost_model.py`, `simulator.py`, `metrics.py`, and `schedulers/{base,normal,chunked}.py`. |
| `experiments/` | Reserved for `run_experiments.py`. |
| `configs/default.yaml` | Parameter file for the Python version. Keep it in sync with `default_config.m`. |
| `tests/` | Unit tests. |
| `results/raw/` | CSV files, one per experiment. |
| `results/graphs/` | PNG figures produced by the experiments. |
| `Dockerfile`, `docker-compose.yml` | Container packaging . |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Files excluded from version control. |

---

## 4. How to run (MATLAB)

1. Open MATLAB and set the **Current Folder** to the `MATLAB/` directory.
2. Run all experiments:
   ```matlab
   run_experiments
   ```
   Or run selected experiments:
   ```matlab
   run_experiments(2)        % only experiment 2
   run_experiments([1 4 5])  % experiments 1, 4 and 5
   ```
3. Open `results/graphs/` for the figures and `results/raw/` for the CSV data.

If MATLAB reports `Undefined function 'default_config'`, check that the helper files are inside `MATLAB/src/` and that `run_experiments.m` adds that folder with `addpath`.

### Changing settings

Edit `MATLAB/src/default_config.m`:

| Setting | Meaning | Default |
|---|---|---|
| `cfg.sched.chunk` | Chunk size `C` | 256 |
| `cfg.sched.maxBatch` | Maximum concurrent requests | 18 |
| `cfg.sched.tileAlign` | Use `chunk = C − #decodes` so the total is a multiple of the tile size | `true` |
| `cfg.hw.flops`, `cfg.hw.bw` | Effective GPU compute and memory bandwidth | calibrated to A6000 |
| `cfg.model.*` | Layers, hidden size, parameters (LLaMA-13B) | 40 / 5120 / 13B |

---

## 5. References

- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*, 2023. https://arxiv.org/abs/2308.16369
- Docker: https://www.docker.com/