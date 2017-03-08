% Compute point laplacian basis and joint laplacian for 16 shape categories
% The 16 categories are a subset of ShapeNetCore with part annotations
% Referece:
%   Chang A X, Funkhouser T, Guibas L, et al. Shapenet: An information-rich 3d model repository[J]. arXiv preprint arXiv:1512.03012, 2015.
%   Yi L, Kim V G, Ceylan D, et al. A scalable active framework for region annotation in 3d shape collections[J]. ACM Transactions on Graphics (TOG), 2016, 35(6): 210.
%
% Contact: ericyi@stanford.edu

%% generate synset list
datapath = '../Data/Categories';
jointspecpath = '../Data/JointSpec';
synsets = importdata('../Data/CategoryList.txt');

%% compute point Laplacian basis for each synset
for s=1:length(synsets)
    synset = synsets{s};
    curdatapath = fullfile(datapath,synset);
    fileList = dir(fullfile(curdatapath,'*.mat'));
    fileList = extractfield(fileList,'name');
    for f=1:length(fileList)
        %%%% load data
        %%%% data.v: vertex functions with last three columns corresponding to scaled xyz coordinates
        %%%% data.label: vertex label
        %%%% data.pts: point clouds
        %%%% data.modelList: model names in shapenet
        data = load(fullfile(curdatapath,fileList{f}),'pts');
        if isfield(data,'V') && isfield(data,'D')
            continue
        end
        nNode = length(data.pts);
        V = cell(nNode,1);
        D = cell(nNode,1);
        parfor ii = 1:nNode
            pt = data.pts{ii};
            pt = bsxfun(@rdivide,pt,max(abs(pt),[],1)+eps);
            kk = 6;
            while 1
                try
                    L = compute_pc_laplacian(pt,kk);
                    [Vtmp,Dtmp] = eigs(L,100,'sm');
                    break;
                catch
                    kk = round(kk*1.2);
                end
            end
            Dtmp = diag(Dtmp);
            [Dtmp,isx] = sort(Dtmp,'ascend');
            Vtmp = Vtmp(:,isx);
            Vtmp = single(Vtmp); Dtmp = single(Dtmp);
            V{ii} = Vtmp; D{ii} = Dtmp;
        end
        save(fullfile(curdatapath,fileList{f}),'V','D','-append');
    end
end

%% compute joint Laplacian
ngrid = 20;
if ~exist(jointspecpath,'dir')
    mkdir(jointspecpath);
end
for s=1:length(synsets)
    synset = synsets{s};
    fileList = dir(fullfile(datapath,synset,'*.mat'));
    fileList = extractfield(fileList,'name');
    res = 2/(ngrid-1);
    W = zeros(ngrid^3); % weight matrix for average shape
    vidx = zeros(ngrid^3,1); % the frequency of voxel being occupied
    k = 7;
    for i=1:length(fileList)
        load(fullfile(datapath,synset,fileList{i}));
        for j=1:length(pts)
            pt = v{j}(:,end-2:end);
            pt = bsxfun(@rdivide,pt,max(abs(pt),[],1)+eps);
            %%%% generate point-voxel correspondences
            vv_grid = ceil((pt+1+eps)/res);
            corres = [1:size(vv_grid,1);(vv_grid(:,3)'-1)*ngrid*ngrid+(vv_grid(:,2)'-1)*ngrid+vv_grid(:,1)'];
            %%%% generate point connection and corresponding weights
            [mm,imx] = pdist2(pt,pt,'Euclidean','Smallest',k);
            mm = mm(2:end,:);
            imx = imx(2:end,:);
            mm = 1./((mm+1e-3).^2);
            %%%% construct weight matrix for average shape
            idx = sub2ind(size(W),reshape(repmat(corres(2,:),k-1,1),(k-1)*size(corres,2),1),corres(2,imx(:))');
            mm = mm(:);
            uidx = unique(idx);
            [cc,ubinidx] = histc(idx,uidx);
            W(uidx) = W(uidx)+accumarray(ubinidx,mm);
            
            ucorres = unique(corres(2,:));
            ucc = histc(corres(2,:),ucorres);
            vidx(ucorres) = vidx(ucorres)+ucc(:);
        end
    end
    %%%% compute Laplacian basis for average shape
    vidx(vidx>0) = [1:sum(vidx>0)]';
    W = W(vidx>0,vidx>0);
    W = sparse(W+W')/2;
    W = W-diag(diag(W));
    L = speye(size(W,1)) - diag(sum(W,2).^(-1/2)) * W * diag(sum(W,2).^(-1/2));
    [V,D] = eigs(L,100,'sm');
    V = single(V);
    save(fullfile(jointspecpath,[synset '.mat']),'V','D','vidx');
end