function cfg = default_config()
%DEFAULT_CONFIG  Hardware, model and scheduler parameters (LLaMA-13B on A6000).
%
% The hardware numbers are *effective* values calibrated so that cost_model()
% reproduces Table 2 of the SARATHI paper:
%   prefill-only, 1024 tokens  -> ~234.8 ms  (0.229 ms/token)
%   decode-only, batch 4, 1K   -> ~50 ms/iteration
% Change them here to simulate another GPU / model.

% ---- hardware -----------------------------------------------------------
cfg.hw.flops     = 113e12;   % effective dense matmul FLOP/s
cfg.hw.bw        = 587e9;    % effective HBM bandwidth (bytes/s)
cfg.hw.tile      = 128;      % matmul tile size (tile quantization)
cfg.hw.attnEff   = 0.40;     % attention kernels reach this fraction of flops
cfg.hw.overhead  = 0.3e-3;   % fixed per-iteration overhead (s)
cfg.hw.usableMem = 41.5e9;   % bytes usable for weights + KV cache

% ---- model (LLaMA-13B) ----------------------------------------------------
cfg.model.layers = 40;
cfg.model.hidden = 5120;
cfg.model.params = 13e9;
cfg.model.bytes  = 2;        % fp16

% ---- scheduler ------------------------------------------------------------
cfg.sched.maxBatch               = 18;    % max concurrent requests (KV limit)
cfg.sched.chunk                  = 256;   % chunk size C for chunked prefill
cfg.sched.tileAlign              = true;  % chunk = C - #decodes (SARATHI sec 4.4)
cfg.sched.normalMaxPrefillTokens = 2048;  % normal prefill: pack whole prompts up to this

% ---- output ---------------------------------------------------------------
cfg.out.showFigures = false;  % true -> keep figures open instead of closing
end
