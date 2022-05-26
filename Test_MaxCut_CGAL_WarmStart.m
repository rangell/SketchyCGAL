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

%% Construct the Black Box Oracles for MaxCut SDP

Primitive1 = @(x) C*x;
Primitive2 = @(y,x) y.*x;
Primitive3 = @(x) sum(x.^2,2);
a = n;
b = ones(n,1);

% Compute scaling factors
SCALE_X = 1/n;
SCALE_C = 1/norm(C,'fro');

%% Implement rounding

err{1} = 'cutvalue'; % name for error
err{2} = @(u) round(C,u); % function definition at the bottom of this script

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
warmstartinit.mySketch = NystromSketch(n, R, 'real');

% Initialize the dual
warmstartinit.z = zeros(size(b,1),1);
warmstartinit.y = zeros(size(b,1),1);
warmstartinit.pobj = 0;

[out, U, D, y, AX, pobj] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
    'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
    'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
    'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
    'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
    'warmstartinit', warmstartinit,... % warm start initialization of state variables
    'errfncs',err,... % err defines the spectral rounding for maxcut
    'stoptol',0.01); % algorithm stops when 1e-2 relative accuracy is achieved


%% Test the warm start
XHat = U * D * U';
warmstartinit.mySketch.S = XHat * warmstartinit.mySketch.Omega;
warmstartinit.z = diag(XHat);
warmstartinit.y = y;
warmstartinit.pobj = trace(C * XHat);

[out, U, D, y, AX, pobj] = CGAL( n, Primitive1, Primitive2, Primitive3, a, b, R, maxit, beta0, K, ...
    'FLAG_MULTRANK_P1',true,... % This flag informs that Primitive1 can be applied to find AUU' for any size U. 
    'FLAG_MULTRANK_P3',true,... % This flag informs that Primitive3 can be applied to find (A'y)U for any size U.
    'SCALE_X',SCALE_X,... % SCALE_X prescales the primal variable X of the problem
    'SCALE_C',SCALE_C,... % SCALE_C prescales the cost matrix C of the problem
    'warmstartinit', warmstartinit,... % warm start initialization of state variables
    'tstart', 2500,... % hand-tuned starting t
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
