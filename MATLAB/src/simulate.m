function res = simulate(reqs, cfg, scheduler)
%SIMULATE  Iteration-level simulation of one GPU serving a stream of requests.
%
%   res = simulate(reqs, cfg, 'normal')   % whole-prompt prefill, prefill-only / decode-only batches
%   res = simulate(reqs, cfg, 'chunked')  % SARATHI: one prefill chunk + piggybacked decodes
%
% Both schedulers admit requests FCFS up to cfg.sched.maxBatch concurrent requests
% (KV-cache limit). Each loop iteration is ONE forward pass whose duration comes
% from cost_model().
%
% NORMAL   : if any admitted request still needs its prefill -> run a prefill-only
%            iteration over whole prompts (packed up to normalMaxPrefillTokens);
%            otherwise run a decode-only iteration over all running requests.
%            Decodes stall while a prefill runs.
% CHUNKED  : every iteration = ONE prefill chunk of the oldest unfinished prefill
%            + one decode token for EVERY running request (decode-maximal batching).
%            chunk = C - #decodes if tileAlign (so chunk+decodes = C), else C.
%
% The prefill's final pass produces the request's first token (TTFT).

n = numel(reqs.P);  P = reqs.P;  D = reqs.D;  arr = reqs.arrival;
B = cfg.sched.maxBatch;

prefilled = zeros(n,1);
decoded   = zeros(n,1);               % output tokens produced so far
firstTok  = nan(n,1);
finish    = nan(n,1);
tok       = nan(n, max(D));           % tok(i,j) = time token j of request i was produced

active = [];  nextAdmit = 1;  nDone = 0;  t = 0;
cap = sum(D) + sum(P) + 10;
iterLog = zeros(cap, 4);  k = 0;      % [t_end, dt, prefillTokens, decodeTokens]

while nDone < n
    % ---- admit (FCFS) ------------------------------------------------------
    while numel(active) < B && nextAdmit <= n && arr(nextAdmit) <= t
        active(end+1) = nextAdmit;  %#ok<AGROW>
        nextAdmit = nextAdmit + 1;
    end
    if isempty(active)                 % idle: jump to next arrival
        t = arr(nextAdmit);
        continue;
    end

    needPre = prefilled(active) < P(active);
    inPre = active(needPre);
    inDec = active(~needPre);

    pre = [];  preTok = [];  dec = [];

    switch scheduler
        case 'normal'
            if ~isempty(inPre)
                pre = inPre(1);  tot = P(pre);
                for j = 2:numel(inPre)
                    if tot + P(inPre(j)) > cfg.sched.normalMaxPrefillTokens, break; end
                    pre(end+1) = inPre(j);  tot = tot + P(inPre(j));  %#ok<AGROW>
                end
                preTok = P(pre);
                dt = cost_model(cfg, preTok, zeros(size(preTok)), []);
            else
                dec = inDec;
                dt = cost_model(cfg, [], [], P(dec) + decoded(dec));
            end

        case 'chunked'
            dec = inDec;
            dCtx = P(dec) + decoded(dec);
            if ~isempty(inPre)
                i = inPre(1);
                if cfg.sched.tileAlign
                    c = cfg.sched.chunk - numel(dec);
                else
                    c = cfg.sched.chunk;
                end
                c = max(1, min(c, P(i) - prefilled(i)));
                pre = i;  preTok = c;
                dt = cost_model(cfg, c, prefilled(i), dCtx);
            else
                dt = cost_model(cfg, [], [], dCtx);
            end

        otherwise
            error('unknown scheduler %s', scheduler);
    end

    t = t + dt;

    % ---- apply decode tokens ----------------------------------------------
    if ~isempty(dec)
        decoded(dec) = decoded(dec) + 1;
        tok(sub2ind(size(tok), dec(:), decoded(dec(:)))) = t;
    end
    % ---- apply prefill progress -------------------------------------------
    if ~isempty(pre)
        prefilled(pre) = prefilled(pre) + preTok(:);
        done = pre(prefilled(pre) >= P(pre));
        decoded(done)  = 1;
        firstTok(done) = t;
        tok(done, 1)   = t;
    end

    % ---- retire finished requests -----------------------------------------
    fin = active(decoded(active) >= D(active) & prefilled(active) >= P(active));
    if ~isempty(fin)
        finish(fin) = t;
        nDone = nDone + numel(fin);
        active = active(~ismember(active, fin));
    end

    k = k + 1;
    iterLog(k,:) = [t, dt, sum(preTok), numel(dec)];
end

res.reqs     = reqs;
res.firstTok = firstTok;
res.finish   = finish;
res.tok      = tok;
res.iterLog  = iterLog(1:k,:);
res.scheduler = scheduler;
end
