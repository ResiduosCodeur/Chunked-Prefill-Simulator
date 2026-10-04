function reqs = make_workload(n, varargin)
%MAKE_WORKLOAD  Generate n requests (arrival time, prompt tokens P, output tokens D).
%
%   reqs = make_workload(n, 'seqLen',1024, 'pdRatio',14, 'rate',0, ...)
%
% Options (name/value):
%   seqLen    total tokens per request P+D            (default 1024)
%   pdRatio   prefill : decode token ratio            (default 10)
%   rate      Poisson arrival rate in req/s; 0 = all requests arrive at t=0
%   dist      'fixed' or 'zipf' (lengths 'minLen':128:'maxLen', Zipf theta)
%   zipfTheta Zipf parameter                          (default 0.4, as in paper)
%   minLen, maxLen                                    (default 1024, 4096)
%   seed      RNG seed                                (default 1)

o = struct('seqLen',1024,'pdRatio',10,'rate',0,'dist','fixed', ...
           'zipfTheta',0.4,'minLen',1024,'maxLen',4096,'seed',1);
for k = 1:2:numel(varargin)
    o.(varargin{k}) = varargin{k+1};
end
try, rng(o.seed); catch, rand('twister', o.seed); end %#ok<NOCOM>

switch o.dist
    case 'fixed'
        L = repmat(o.seqLen, n, 1);
    case 'zipf'
        vals = (o.minLen:128:o.maxLen)';
        w    = (1:numel(vals)).^(-o.zipfTheta);
        cdf  = cumsum(w) / sum(w);
        idx  = sum(rand(n,1) > cdf(:)', 2) + 1;
        idx  = min(idx, numel(vals));
        L    = vals(idx);
    otherwise
        error('unknown dist %s', o.dist);
end

r = o.pdRatio;
P = max(1, round(L * r / (1 + r)));
D = max(1, L - P);

if o.rate > 0
    arrival = cumsum(-log(rand(n,1)) / o.rate);
else
    arrival = zeros(n,1);
end

reqs.arrival = arrival;
reqs.P = P;
reqs.D = D;
end
