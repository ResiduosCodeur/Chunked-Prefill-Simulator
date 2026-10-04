function B = max_batch_size(cfg, seqLen)
%MAX_BATCH_SIZE  Largest batch whose KV caches fit in GPU memory (SARATHI sec 4.3.1):
%   B = floor( (M_G - M_S) / (L * m_kv) )
% M_G = usable GPU memory, M_S = model weights, m_kv = KV bytes per token.
m   = cfg.model;
mkv = 2 * m.hidden * m.layers * m.bytes;
B   = floor((cfg.hw.usableMem - m.params * m.bytes) / (seqLen * mkv));
B   = max(B, 1);
end
