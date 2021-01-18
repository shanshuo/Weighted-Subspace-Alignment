function acc = WSADA(Xs,Xt,Ys,Yt,d,mode,sigma)
% Visual Domain Adaptation using Weighted Subspace Alignment
% Xs: source samples, a row represents a sample
% Xt: target samples, a row represents a sample
% Ys: source label, column vector
% Yt: target label, column vector
% d:  subspace dimension
% mode: 'sa'-ICCV 2013  'wsada'-my algorithm
%       'gauss'-Gaussian model  'laplace'-Laplacian model
% simga: model parameter


DIST = pdist2(Xt, Xs);
v = min(DIST,[],2);
w = ones(size(Xs,1),1);
% w = zeros(size(Xs,1),1);

for i=1:size(DIST,2)
    for j=1:size(DIST,1)
        if strcmp(mode,'sa')
            break
        end
        if strcmp(mode,'wsada')
            if abs(DIST(j,i) - v(j)) <= 0
                w(i) = w(i) + 1;
            end
        end
        if strcmp(mode,'gauss')
            w(i) = w(i) + exp(-(DIST(j,i)-v(j))^2/sigma); % Gaussian model
        end
        if strcmp(mode,'laplace')
           w(i) = w(i) + exp(-abs(DIST(j,i)-v(j))/sigma); % Laplacian model
        end
    end
    if w(i) > 1
        w(i) = w(i) - 1;
    end
end

% m = size(Xs,1);

% W = diag(w);

% w = logsig(w);

Source_Data = Xs;

Source_label = Ys;

Ps = weightedPCA(Source_Data,w);
% weighted PCA

Pt = pca(Xt);
acc = Subspace_Alignment(Xs,Xt,Source_label,Yt,Ps(:,1:d),Pt(:,1:d));   
acc = acc/length(Yt);

function [accuracy_sa_nn] =  Subspace_Alignment(Source_Data,Target_Data,Source_label,Target_label,Xs,Xt)
Target_Aligned_Source_Data = Source_Data*(Xs * Xs'*Xt);
Target_Projected_Data = Target_Data*Xt;
NN_Neighbours = 1; %  neares neighbour classifier
% predicted_Label = cvKnn(Target_Projected_Data', Target_Aligned_Source_Data', Source_label, NN_Neighbours);
mdl = ClassificationKNN.fit(Target_Aligned_Source_Data, Source_label,'Numneighbors',NN_Neighbours,'distanceweight','squaredinverse');
predicted_Label = predict(mdl,Target_Projected_Data);
r=find(predicted_Label==Target_label);
accuracy_sa_nn = length(r); 

function coeff = weightedPCA(Source_Data, w)
w = reshape(w,1,length(w));
mu = w*Source_Data ./sum(w);
Source_Data = bsxfun(@minus,Source_Data,mu);
OmegaSqrt = sqrt(w);
Source_Data = bsxfun(@times,Source_Data,OmegaSqrt');
% [U,~,coeff] = svd(Source_Data,'econ');
% U = bsxfun(@times,U,1./OmegaSqrt');
[coeff, eigValueDiag] = eig(Source_Data'*Source_Data);
[eigValues, idx] = sort(diag(eigValueDiag), 'descend');
coeff = coeff(:, idx);
coeff = coeff(:,1:end-1);
[~,maxind] = max(abs(coeff), [], 1);
[d1, d2] = size(coeff);
colsign = sign(coeff(maxind + (0:d1:(d2-1)*d1)));
coeff = bsxfun(@times, coeff, colsign);