function L = compute_pc_laplacian(v,k)
%%%% compute normalized distance laplacian for point cloud
%%%% v - point coordinates
%%%% k - parameter for KNN graph construction
if size(v,1)==3
    v = v';
end
n = size(v,1);
[mm,imx] = pdist2(v,v,'Euclidean','Smallest',k);
mm = mm(2:end,:);
imx = imx(2:end,:);
mm = 1./(mm.^2);
W = zeros(n);
for i=1:k-1
    W(sub2ind(size(W),[1:n],imx(i,:))) = mm(i,:);
end
W = sparse(W+W')/2; 
L = speye(n) - diag(sum(W,2).^(-1/2)) * W * diag(sum(W,2).^(-1/2));