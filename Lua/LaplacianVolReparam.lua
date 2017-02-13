--[[ Generate volumetric reparameterization of graph laplacian eigenbasis, which will be used as input for functional map prediction

  nDim           - Number of laplacian eigenbasis to be reparameterized
  nGrid          - Voxelization resolution, the reparameterization will be represented in an (nGrid x nGrid x nGrid) space
  specParam[i].V - Laplacian eigenbasis of graph_i, of shape (x_i, m_i), with each column corresponding to one basis, x_i is the number of vertices for graph_i, m_i is the number of spectral basis used for graph_i, i=1,...,N, N is the batch size
  input          - Input graph vertex functions, of shape (x_1 + x_2 + ... + x_N, c), c is the number of signal channels. The last three columns of input corresponds to graph vertex positions in R3, which will be used to generate graph voxelization. The rest graph vertex functions are not used in this reparameterization step.

  output         - Output volumetric reparameterization of graph laplacian eigenbasis, of shape (N * nDim, 1, nGrid, nGrid, nGrid)

]]--

local LaplacianVolReparam, parent = torch.class('nn.LaplacianVolReparam', 'nn.Module')

function LaplacianVolReparam:__init(nDim,nGrid)
   parent.__init(self)
   self.nDim = nDim or 30
   self.nGrid = nGrid or 20
   self.specParam = {}
end

function LaplacianVolReparam:setSpecParam(specParam)
   self.specParam = specParam
   return self
end

function LaplacianVolReparam:updateOutput(input) 
   -- split index for input (spatial representation)
   local splitimx = {}
   for i=1,#self.specParam do
      splitimx[i] = self.specParam[i].V:size(1)
   end
   splitimx = torch.Tensor(splitimx)
   self.splitimx = splitimx
   assert(splitimx:sum()==input:size(1),'Input and specParam do not match!')
   local ndim = self.nDim
   local ngrid = self.nGrid
   local res = 2/(ngrid-1)
   local count = 0
   local nfea = input:size(2)
   
   self.output = torch.Tensor(#self.specParam,ndim,ngrid,ngrid,ngrid):typeAs(input):fill(0)   
   for i=1,#self.specParam do
        
      -- normalize point cloud
      v = input[{{count+1,count+splitimx[i]},{nfea-2,nfea}}]:clone()
      local vmax = torch.abs(v):max(1)[1]
      v[{{},{1}}] = v[{{},{1}}]/vmax[1]
      v[{{},{2}}] = v[{{},{2}}]/vmax[2]
      v[{{},{3}}] = v[{{},{3}}]/vmax[3]
        
      -- compute which voxels will be occupied
      local v_grid = ((v+1+1e-4)/res):ceil()
      local fea = self.specParam[i].V[{{},{1,ndim}}]:clone()
      if input:type()=="torch.CudaTensor" then
          v_grid = v_grid[{{},{3}}]:add(-1):mul(ngrid*ngrid):add( v_grid[{{},{2}}]:add(-1):mul(ngrid) ):add( v_grid[{{},{1}}] ):type("torch.CudaLongTensor")
      elseif input:type()=="torch.DoubleTensor" then
          v_grid = v_grid[{{},{3}}]:add(-1):mul(ngrid*ngrid):add( v_grid[{{},{2}}]:add(-1):mul(ngrid) ):add( v_grid[{{},{1}}] ):type("torch.LongTensor")
      end
      
      -- generate volumetric reparameterization of laplacian basis
      local tmp = torch.Tensor(ndim,ngrid*ngrid*ngrid):typeAs(input):fill(0):scatter(2,v_grid:t():expand(ndim,v_grid:size(1)),fea:t())
      self.output[i] = tmp:view(ndim,ngrid,ngrid,ngrid)
      count = count+splitimx[i]
   end
   self.output = self.output:view(#self.specParam*ndim,1,ngrid,ngrid,ngrid)
   return self.output
end

function LaplacianVolReparam:updateGradInput(input, gradOutput)
   self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):fill(0)
   return self.gradInput
end

function LaplacianVolReparam:clearState()
   return parent.clearState(self)
end