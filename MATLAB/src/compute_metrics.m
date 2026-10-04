function m = compute_metrics(res)
%COMPUTE_METRICS  Latency / throughput metrics from a simulate() result.
%
%   tput_tok_per_ms : (prefill+decode tokens) / makespan   (paper's end-to-end throughput)
%   out_tok_per_s   : generated tokens per second
%   ttft_*          : time-to-first-token (ms) = first token time - arrival
%   tbt_*           : time-between-tokens (ms) over all consecutive output tokens
%   ttftAll, tbtAll : raw samples (for CDF plots)

r = res.reqs;
makespan = max(res.finish) - min(r.arrival);

m.makespan_s       = makespan;
m.tput_tok_per_ms  = (sum(r.P) + sum(r.D)) / (makespan * 1e3);
m.out_tok_per_s    = sum(r.D) / makespan;

ttft = (res.firstTok - r.arrival) * 1e3;
gaps = diff(res.tok, 1, 2) * 1e3;
gaps = gaps(~isnan(gaps));

m.ttftAll     = ttft;
m.tbtAll      = gaps;
m.ttft_mean   = mean(ttft);
m.ttft_p50    = pct(ttft, 50);
m.ttft_p99    = pct(ttft, 99);
if isempty(gaps)
    m.tbt_mean = NaN;  m.tbt_p99 = NaN;
else
    m.tbt_mean = mean(gaps);
    m.tbt_p99  = pct(gaps, 99);
end
m.e2e_mean_s = mean(res.finish - r.arrival);
end

function v = pct(x, q)
x = sort(x(:));
idx = max(1, ceil(q / 100 * numel(x)));
v = x(idx);
end
