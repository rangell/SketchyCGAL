%%  Test setup for MaxCut SDP - Solved with SketchyCGAL
%% Alp Yurtsever (alp.yurtsever@epfl.ch - alpy@mit.edu)

%% Choose data
% NOTE: You need to download data from GSET and locate them to under the
% "FilesMaxCut/data/G/" folder (resp. DIMACS10, "FilesMaxCut/data/DIMACS10/").

% maxcut_data = 'G/G1'; % you can choose other data files as well
% maxcut_data = 'DIMACS10/belgium_osm';

%% Preamble
rng(0,'twister');
addpath utils;
addpath solver;

%% Load data
maxcut_data = 'G/G22';
load(['./FilesMaxCut/data/',maxcut_data]);

n = size(Problem.A,1);
C = spdiags(Problem.A*ones(n,1),0,n,n) - Problem.A;
C = 0.5*(C+C'); % symmetrize if not symmetric
C = (-0.25).*C;

clearvars Problem;

%% Create warm-start data and primitives
warmStartFrac = 0.99;

warmStartn = floor(warmStartFrac * n);
warmStartIndices = 1:warmStartn; % we can change what subset later
warmStartC = C(warmStartIndices, warmStartIndices);

Primitive1 = @(x) warmStartC*x;
Primitive2 = @(y,x) y.*x;
Primitive3 = @(x) sum(x.^2,2);
a = warmStartn;
b = ones(warmStartn,1);

% Compute scaling factors
SCALE_X = 1/warmStartn;
SCALE_C = 1/norm(warmStartC,'fro');

err{1} = 'cutvalue'; % name for error
err{2} = @(u) round(warmStartC,u); % function definition at the bottom of this script

%% Start memory logging
% NOTE: This works only on Unix systems. 

hmL = memLog([mfilename,'_',maxcut_data]);
hmL.start;
MEMBEGIN = hmL.prompt;

%% Solve using SketchyCGAL

R = 100; % rank/sketch size parameter
beta0 = 1; % we didn't tune - choose 1 - you can tune this!
K = inf;
maxit = 1e6; % limit on number of iterations

timer = tic;
cputimeBegin = cputime;

% Initialize the decision variable
warmStartInit.mySketch = NystromSketch(warmStartn, R, 'real');

% Initialize the dual
warmStartInit.z = zeros(size(b,1),1);
warmStartInit.y = zeros(size(b,1),1);
warmStartInit.pobj = 0;

[out, U, D, y, AX, pobj] = CGAL(warmStartn, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
    'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
    'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
    'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
    'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
    'warmstartinit', warmStartInit,... % warm start initialization of state variables
    'errfncs',err,... % err defines the spectral rounding for maxcut
    'stoptol',0.001); % algorithm stops when 1e-2 relative accuracy is achieved


%% Create test primitives
Primitive1 = @(x) C*x;
Primitive2 = @(y,x) y.*x;
Primitive3 = @(x) sum(x.^2,2);
a = n;
b = ones(n,1);

% Compute scaling factors
SCALE_X = 1/n;
SCALE_C = 1/norm(C,'fro');

%% Expand the warm-start solution
XHat = U * D * U';

% Pad XHat
XHat = padarray(XHat, [n - warmStartn, n - warmStartn], 'replicate', 'post');

% Create a new appropriately-sized sketch
warmStartInit.mySketch = NystromSketch(n, R, 'real');

% Instantiate all other warm-start variables
warmStartInit.mySketch.S = XHat * warmStartInit.mySketch.Omega;
warmStartInit.z = diag(XHat);
warmStartInit.y = cat(1, y, ones(n - warmStartn, 1));
warmStartInit.pobj = trace(C * XHat);

err{1} = 'cutvalue'; % name for error
err{2} = @(u) round(C,u); % function definition at the bottom of this script

[out, U, D, y, AX, pobj] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
    'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
    'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
    'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
    'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
    'warmStartInit', warmStartInit,... % warm start initialization of state variables
    'tstart', 1,... % hand-tuned starting t
    'errfncs',err,... % err defines the spectral rounding for maxcut
    'stoptol',0.001); % algorithm stops when 1e-3 relative accuracy is achieved

cputimeEnd = cputime;
totalTime = toc(timer);

out.totalTime = totalTime;
out.totalCpuTime = cputimeEnd - cputimeBegin;

%% Stop memory logging

MEMEND = hmL.prompt;
hmL.stop;
out.memory = (MEMEND - MEMBEGIN)/1000; %in MB

%% Evaluate errors

out.cutvalue = out.info.cutvalue(end);
out.primalObj = out.info.primalObj(end);
out.primalFeas = out.info.primalFeas(end);

%% Save results

if ~exist(['results/MaxCut/',maxcut_data],'dir'), mkdir(['results/MaxCut/',maxcut_data]); end
save(['results/MaxCut/',maxcut_data,'/SketchyCGAL.mat'],'out','-v7.3');

%% Implement rounding
% NOTE: Defining a function in a MATLAB script was not available in older
% versions. If you are using an old version of MATLAB, you might need to
% save this function as a seperate ".m" file in your path.

function cutvalue = round(C,u)
cutvalue = 0;
for t = 1:size(u,2)
    sign_evec = sign(u(:,t));
    rankvalue = -(sign_evec'*(C*sign_evec));
    cutvalue = max(cutvalue, rankvalue);
end
end

%% Last edit: Alp Yurtsever - July 24, 2020
