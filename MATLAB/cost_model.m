function dt = cost_model(cfg, pChunk, pCtx, dCtx)
%COST_MODEL  Latency (seconds) of ONE forward pass over a (hybrid) batch.
%
%   pChunk : vector, number of prefill tokens of each prefill piece in the batch
%   pCtx   : vector, tokens of that request already in the KV cache (earlier chunks)
%   dCtx   : vector, current context length of each decode request in the batch
%
% Linear layers (preproj, postproj, ffn): all tokens of the batch are fused into
% one matmul, so   t_lin = max( compute time , weight-load time )   (roofline).
%   - decode-only batch  -> tiny T  -> memory bound  (weights loaded for 1 token each)
%   - prefill / hybrid   -> big T   -> compute bound (decodes ride along "for free")
% T is rounded up to a multiple of the tile size (tile quantization).
%
% Attention is NOT fused across requests:
%   - prefill piece : compute  ~ 4*p*(ctx+p/2)*H*L ; memory = re-read of prior KV
%                     (this re-read is the overhead of chunking, sec 4.2)
%   - decodes       : memory bound, read the whole KV cache of every request

hw = cfg.hw;  m = cfg.model;
pChunk = pChunk(:);  pCtx = pCtx(:);  dCtx = dCtx(:);

T = sum(pChunk) + numel(dCtx);
if T == 0
    dt = 0;  return;
end

% ---- linear operators -------------------------------------------------------
Tpad  = ceil(T / hw.tile) * hw.tile;
tComp = 2 * Tpad * m.params / hw.flops;
tMem  = m.params * m.bytes / hw.bw;
tLin  = max(tComp, tMem);

% ---- attention --------------------------------------------------------------
kvPerTok = 2 * m.hidden * m.layers * m.bytes;      % K and V, all layers
tAttn = 0;
if ~isempty(pChunk)
    flopsA = 4 * pChunk .* (pCtx + pChunk / 2) * m.hidden * m.layers;
    tAc    = flopsA / (hw.flops * hw.attnEff);
    tAm    = pCtx * kvPerTok / hw.bw;
    tAttn  = tAttn + sum(max(tAc, tAm));
end
if ~isempty(dCtx)
    tAttn = tAttn + sum(dCtx) * kvPerTok / hw.bw;
end

dt = tLin + tAttn + hw.overhead;
end
