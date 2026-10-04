function run_experiments(which)
%RUN_EXPERIMENTS  Normal prefill vs chunked prefill: run sweeps, save CSVs + graphs.
%
%   run_experiments          % run everything
%   run_experiments(2)       % only experiment 2
%   run_experiments([1 4])   % experiments 1 and 4
%
%   1  TTFT vs prompt length (single request)            -> chunking overhead
%   2  Throughput gain vs P:D ratio (1K/2K/3K)           -> paper Fig. 9
%   3  Throughput gain vs batch size (balanced P:D)      -> paper Fig. 10 / Table 4
%   4  Online load sweep: throughput, TTFT, TBT          -> Poisson arrivals
%   5  Iteration-time timeline + TBT CDF                 -> decode stalls
%   6  Chunk-size sweep (tile quantization)              -> paper Fig. 13 / sec 4.4
%
% CSVs  -> results/raw/     PNGs -> results/graphs/

if nargin < 1, which = 1:6; end
here = fileparts(mfilename('fullpath'));   % the MATLAB/ folder
root = fileparts(here);                    % repo root
addpath(fullfile(here, 'src'));            % MATLAB/src
rawDir   = fullfile(root, 'results', 'raw');
graphDir = fullfile(root, 'results', 'graphs');
if ~exist(rawDir,   'dir'), mkdir(rawDir);   end
if ~exist(graphDir, 'dir'), mkdir(graphDir); end

cfg = default_config();
for e = which(:)'
    fprintf('\n=== Experiment %d ===\n', e);
    switch e
        case 1, exp1_ttft_vs_prompt(cfg, rawDir, graphDir);
        case 2, exp2_pd_ratio(cfg, rawDir, graphDir);
        case 3, exp3_batch_size(cfg, rawDir, graphDir);
        case 4, exp4_online_load(cfg, rawDir, graphDir);
        case 5, exp5_stalls(cfg, rawDir, graphDir);
        case 6, exp6_chunk_sweep(cfg, rawDir, graphDir);
    end
end
fprintf('\nDone. CSVs in %s\nGraphs in %s\n', rawDir, graphDir);
end

% =============================================================================
% Experiment 1: TTFT of a single request vs prompt length
% =============================================================================
function exp1_ttft_vs_prompt(cfg, rawDir, graphDir)
Ls = 256:128:4096;
Cs = [128 256 512];
cfg.sched.maxBatch = 1;
ttft = zeros(numel(Ls), 1 + numel(Cs));
for i = 1:numel(Ls)
    reqs = one_request(Ls(i), 16);
    ttft(i,1) = compute_metrics(simulate(reqs, cfg, 'normal')).ttft_mean;
    for j = 1:numel(Cs)
        ttft(i,1+j) = compute_metrics(simulate(reqs, with_chunk(cfg, Cs(j), false), 'chunked')).ttft_mean;
    end
end
save_csv(fullfile(rawDir,'exp1_ttft_vs_prompt.csv'), ...
    {'prompt_len','normal_ms','chunk128_ms','chunk256_ms','chunk512_ms'}, [Ls(:) ttft]);

fig = new_fig([100 100 1000 420]);
subplot(1,2,1); hold on;
plot(Ls, ttft(:,1), 'k-o', 'LineWidth', 1.8, 'MarkerSize', 4);
for j = 1:numel(Cs), plot(Ls, ttft(:,1+j), '-s', 'LineWidth', 1.3, 'MarkerSize', 3); end
xlabel('Prompt length (tokens)'); ylabel('TTFT (ms)');
title('TTFT of a single request'); grid on; box on;
legend(['Normal prefill', arrayfun(@(c) sprintf('Chunked (C=%d)',c), Cs, 'UniformOutput', false)], 'Location','northwest');
subplot(1,2,2); hold on;
for j = 1:numel(Cs), plot(Ls, 100*(ttft(:,1+j)./ttft(:,1) - 1), '-s', 'LineWidth', 1.3, 'MarkerSize', 3); end
xlabel('Prompt length (tokens)'); ylabel('TTFT increase vs normal (%)');
title('Chunking overhead on TTFT'); grid on; box on;
legend(arrayfun(@(c) sprintf('C=%d',c), Cs, 'UniformOutput', false), 'Location','northeast');
finish_fig(fig, fullfile(graphDir,'exp1_ttft_vs_prompt.png'), cfg);
end

% =============================================================================
% Experiment 2: normalized throughput vs P:D ratio (paper Fig. 9)
% =============================================================================
function exp2_pd_ratio(cfg, rawDir, graphDir)
Ls     = [1024 2048 3072];
ratios = [1 2 4 8 14 20 28 40 60 80 100 130 160 200];
Cs     = [128 256 512];

fig = new_fig([100 100 1300 400]);
for a = 1:numel(Ls)
    L = Ls(a);  B = max_batch_size(cfg, L);
    cB = cfg;  cB.sched.maxBatch = B;
    n  = max(120, 8*B);
    gain = zeros(numel(ratios), numel(Cs));
    for i = 1:numel(ratios)
        reqs = make_workload(n, 'seqLen', L, 'pdRatio', ratios(i));
        mN = compute_metrics(simulate(reqs, cB, 'normal'));
        for j = 1:numel(Cs)
            mC = compute_metrics(simulate(reqs, with_chunk(cB, Cs(j), true), 'chunked'));
            gain(i,j) = mC.tput_tok_per_ms / mN.tput_tok_per_ms;
        end
    end
    save_csv(fullfile(rawDir, sprintf('exp2_pd_ratio_L%d_B%d.csv', L, B)), ...
        {'pd_ratio','gain_C128','gain_C256','gain_C512'}, [ratios(:) gain]);

    subplot(1,3,a); hold on;
    plot([0 max(ratios)], [1 1], 'k--');
    for j = 1:numel(Cs), plot(ratios, gain(:,j), '-o', 'LineWidth', 1.4, 'MarkerSize', 3); end
    xlabel('Prefill / Decode ratio'); ylabel('Throughput (chunked / normal)');
    title(sprintf('Seq len = %dK, batch size = %d', L/1024, B));
    ylim([0.95 1.4]); grid on; box on;
    if a == 1, legend('Normal = 1.0', 'C=128', 'C=256', 'C=512', 'Location','northeast'); end
    [pk, ip] = max(gain(:));  [ir, ic] = ind2sub(size(gain), ip);
    fprintf('L=%d B=%d: peak gain %.3fx at P:D=%d with C=%d\n', L, B, pk, ratios(ir), Cs(ic));
end
finish_fig(fig, fullfile(graphDir,'exp2_throughput_vs_pd_ratio.png'), cfg);
end

% =============================================================================
% Experiment 3: throughput gain vs batch size at the balanced P:D = C/(B-1)
% =============================================================================
function exp3_batch_size(cfg, rawDir, graphDir)
Ls = [1024 2048 3072];
Cs = [256 512];
fig = new_fig([100 100 1000 420]);
for c = 1:numel(Cs)
    C = Cs(c);
    subplot(1,2,c); hold on;
    for a = 1:numel(Ls)
        L = Ls(a);  Bmax = max_batch_size(cfg, L);
        Bs = 2:Bmax;
        gain = zeros(numel(Bs),1);
        for i = 1:numel(Bs)
            cB = cfg;  cB.sched.maxBatch = Bs(i);
            reqs = make_workload(max(120, 8*Bs(i)), 'seqLen', L, 'pdRatio', C/(Bs(i)-1));
            mN = compute_metrics(simulate(reqs, cB, 'normal'));
            mC = compute_metrics(simulate(reqs, with_chunk(cB, C, true), 'chunked'));
            gain(i) = mC.tput_tok_per_ms / mN.tput_tok_per_ms;
        end
        save_csv(fullfile(rawDir, sprintf('exp3_batch_C%d_L%d.csv', C, L)), {'batch','gain'}, [Bs(:) gain]);
        plot(Bs, gain, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
        fprintf('C=%d L=%d: peak gain %.3fx at B=%d\n', C, L, max(gain), Bs(find(gain==max(gain),1)));
    end
    plot(xlim, [1 1], 'k--');
    xlabel('Batch size'); ylabel('Throughput (chunked / normal)');
    title(sprintf('Balanced P:D = C/(B-1), chunk size C = %d', C)); grid on; box on;
    legend('1K','2K','3K','Normal = 1.0','Location','southeast');
end
finish_fig(fig, fullfile(graphDir,'exp3_throughput_vs_batch.png'), cfg);
end

% =============================================================================
% Experiment 4: online serving, Poisson arrivals, Zipf lengths
% =============================================================================
function exp4_online_load(cfg, rawDir, graphDir)
cfg.sched.maxBatch = 6;
n = 200;
wl = @(rate) make_workload(n, 'dist','zipf', 'minLen',1024, 'maxLen',3072, ...
                           'pdRatio',10, 'rate',rate, 'seed',7);
mOff = compute_metrics(simulate(wl(0), cfg, 'normal'));
capReq = n / mOff.makespan_s;                    % saturation rate of the baseline
fracs = [0.2 0.4 0.6 0.8 0.9 1.0 1.1 1.2];
rates = capReq * fracs;
fprintf('baseline saturation ~ %.2f req/s\n', capReq);

cC = with_chunk(cfg, cfg.sched.chunk, true);
M = zeros(numel(rates), 9);
for i = 1:numel(rates)
    reqs = wl(rates(i));
    a = compute_metrics(simulate(reqs, cfg, 'normal'));
    b = compute_metrics(simulate(reqs, cC,  'chunked'));
    M(i,:) = [rates(i), a.tput_tok_per_ms, b.tput_tok_per_ms, a.ttft_mean, b.ttft_mean, ...
              a.ttft_p99, b.ttft_p99, a.tbt_p99, b.tbt_p99];
end
save_csv(fullfile(rawDir,'exp4_online_load.csv'), {'rate_req_s','tput_normal','tput_chunked', ...
    'ttft_mean_normal','ttft_mean_chunked','ttft_p99_normal','ttft_p99_chunked', ...
    'tbt_p99_normal','tbt_p99_chunked'}, M);

fig = new_fig([100 100 1100 800]);
names = {'Throughput (tokens/ms)','Mean TTFT (ms)','P99 TTFT (ms)','P99 TBT (ms)'};
cols  = {[2 3],[4 5],[6 7],[8 9]};
for p = 1:4
    subplot(2,2,p); hold on;
    plot(M(:,1), M(:,cols{p}(1)), 'r-o', 'LineWidth', 1.6, 'MarkerSize', 4);
    plot(M(:,1), M(:,cols{p}(2)), 'b-s', 'LineWidth', 1.6, 'MarkerSize', 4);
    xlabel('Arrival rate (requests/s)'); ylabel(names{p}); grid on; box on;
    if p > 1, set(gca,'YScale','log'); end
    if p == 1, legend('Normal prefill','Chunked prefill','Location','southeast'); end
end
finish_fig(fig, fullfile(graphDir,'exp4_online_load.png'), cfg);
end

% =============================================================================
% Experiment 5: decode stalls (iteration timeline) and TBT CDF
% =============================================================================
function exp5_stalls(cfg, rawDir, graphDir)
cfg.sched.maxBatch = 6;
n = 200;
reqs0 = make_workload(n, 'dist','zipf', 'minLen',1024, 'maxLen',3072, 'pdRatio',10, 'seed',7);
capReq = n / compute_metrics(simulate(reqs0, cfg, 'normal')).makespan_s;
reqs = make_workload(n, 'dist','zipf', 'minLen',1024, 'maxLen',3072, 'pdRatio',10, ...
                     'rate', 0.8*capReq, 'seed',7);
rN = simulate(reqs, cfg, 'normal');
rC = simulate(reqs, with_chunk(cfg, cfg.sched.chunk, true), 'chunked');
mN = compute_metrics(rN);  mC = compute_metrics(rC);

fig = new_fig([100 100 1100 800]);
tmax = 30;
subplot(2,2,1);
sel = rN.iterLog(:,1) <= tmax;
stem(rN.iterLog(sel,1), rN.iterLog(sel,2)*1e3, 'r.', 'MarkerSize', 4);
xlabel('Time (s)'); ylabel('Iteration time (ms)'); title('Normal prefill: iteration times');
grid on; box on;
subplot(2,2,2);
sel = rC.iterLog(:,1) <= tmax;
stem(rC.iterLog(sel,1), rC.iterLog(sel,2)*1e3, 'b.', 'MarkerSize', 4);
xlabel('Time (s)'); ylabel('Iteration time (ms)'); title('Chunked prefill: iteration times');
grid on; box on;
yl = [0 max(max(rN.iterLog(:,2)), max(rC.iterLog(:,2)))*1e3*1.05];
subplot(2,2,1); ylim(yl); subplot(2,2,2); ylim(yl);

subplot(2,2,3); hold on;
plot_cdf(mN.tbtAll, 'r');  plot_cdf(mC.tbtAll, 'b');
set(gca,'XScale','log'); xlabel('Time between tokens (ms)'); ylabel('CDF');
title('TBT distribution'); legend('Normal','Chunked','Location','southeast'); grid on; box on;
subplot(2,2,4); hold on;
plot_cdf(mN.ttftAll, 'r');  plot_cdf(mC.ttftAll, 'b');
set(gca,'XScale','log'); xlabel('TTFT (ms)'); ylabel('CDF');
title('TTFT distribution'); legend('Normal','Chunked','Location','southeast'); grid on; box on;
finish_fig(fig, fullfile(graphDir,'exp5_stalls_and_cdfs.png'), cfg);

save_csv(fullfile(rawDir,'exp5_summary.csv'), ...
    {'sched','tput_tok_ms','ttft_mean','ttft_p99','tbt_mean','tbt_p99'}, ...
    [1 mN.tput_tok_per_ms mN.ttft_mean mN.ttft_p99 mN.tbt_mean mN.tbt_p99; ...
     2 mC.tput_tok_per_ms mC.ttft_mean mC.ttft_p99 mC.tbt_mean mC.tbt_p99]);
end

% =============================================================================
% Experiment 6: chunk-size sweep (tile quantization, overhead vs piggybacking)
% =============================================================================
function exp6_chunk_sweep(cfg, rawDir, graphDir)
L = 1024;  B = max_batch_size(cfg, L);
cfg.sched.maxBatch = B;
Cs = 64:16:1024;
reqs   = make_workload(160, 'seqLen', L, 'pdRatio', 14);
mN     = compute_metrics(simulate(reqs, cfg, 'normal'));
one    = one_request(L, 16);
cfg1   = cfg;  cfg1.sched.maxBatch = 1;
ttftN  = compute_metrics(simulate(one, cfg1, 'normal')).ttft_mean;

G = zeros(numel(Cs), 4);  % gain aligned, gain unaligned, ttft aligned-ratio(single req) , unused
for i = 1:numel(Cs)
    a = compute_metrics(simulate(reqs, with_chunk(cfg, Cs(i), true),  'chunked'));
    u = compute_metrics(simulate(reqs, with_chunk(cfg, Cs(i), false), 'chunked'));
    t = compute_metrics(simulate(one,  with_chunk(cfg1, Cs(i), false), 'chunked'));
    G(i,:) = [a.tput_tok_per_ms/mN.tput_tok_per_ms, u.tput_tok_per_ms/mN.tput_tok_per_ms, ...
              t.ttft_mean/ttftN, 0];
end
save_csv(fullfile(rawDir,'exp6_chunk_sweep.csv'), {'chunk','gain_aligned','gain_unaligned','ttft_ratio_single'}, [Cs(:) G(:,1:3)]);

fig = new_fig([100 100 1000 420]);
subplot(1,2,1); hold on;
plot(Cs, G(:,1), 'b-o', 'LineWidth', 1.4, 'MarkerSize', 3);
plot(Cs, G(:,2), 'r--s', 'LineWidth', 1.2, 'MarkerSize', 3);
plot(xlim, [1 1], 'k:');
xlabel('Chunk size C (tokens)'); ylabel('Throughput (chunked / normal)');
title(sprintf('Throughput vs chunk size (L=%d, B=%d, P:D=14)', L, B));
legend('chunk = C - #decodes (tile-aligned)', 'chunk = C (unaligned)', 'Location','southeast'); grid on; box on;
subplot(1,2,2);
plot(Cs, G(:,3), 'm-o', 'LineWidth', 1.4, 'MarkerSize', 3);
xlabel('Chunk size C (tokens)'); ylabel('TTFT (chunked / normal)');
title(sprintf('TTFT of a single %d-token prompt', L)); grid on; box on;
finish_fig(fig, fullfile(graphDir,'exp6_chunk_size_sweep.png'), cfg);
end

% =============================================================================
% helpers
% =============================================================================
function reqs = one_request(P, D)
reqs.arrival = 0;  reqs.P = P;  reqs.D = D;
end

function c = with_chunk(cfg, C, align)
c = cfg;  c.sched.chunk = C;  c.sched.tileAlign = align;
end

function fig = new_fig(pos)
fig = figure('Visible', 'off', 'Position', pos, 'Color', 'w');
end

function finish_fig(fig, path, cfg)
saveas(fig, path);
fprintf('saved %s\n', path);
if cfg.out.showFigures, set(fig, 'Visible', 'on'); else, close(fig); end
end

function plot_cdf(x, colour)
x = sort(x(:));
plot(x, (1:numel(x))' / numel(x), colour, 'LineWidth', 1.6);
end

function save_csv(path, header, M)
fid = fopen(path, 'w');
fprintf(fid, '%s\n', strjoin(header, ','));
fmt = [repmat('%.6g,', 1, size(M,2)-1) '%.6g\n'];
fprintf(fid, fmt, M');
fclose(fid);
end
