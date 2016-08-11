local PairwiseConv = torch.class('similarityMeasure.Conv')
dofile 'CsDis.lua'

function PairwiseConv:__init(config)
  self.mem_dim       = config.mem_dim       or 150 --200
  self.learning_rate = config.learning_rate or 0.001
  self.batch_size    = config.batch_size    or 1 --25
  self.num_layers    = config.num_layers    or 1
  self.reg           = config.reg           or 1e-4
  self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
  self.sim_nhidden   = config.sim_nhidden   or 150
  self.task          = config.task          or 'qa' 
  self.neg_mode = config.neg_mode or 2 -- 1 is random, 2 is max, 3 is mix
  self.num_pairs = config.num_pairs or 8
  self.loss_mode = 2 -- 1 is margin loss, 2 is self-defined loss	
  self.dropout_mode = config.dropout_mode or 1 -- 1 is to add dropout layer, dropout prob is 0.5
  -- word embedding
  self.emb_vecs = config.emb_vecs
  self.emb_dim = config.emb_vecs:size(2)
  self.cos_dist = nn.CsDis()
  -- number of similarity rating classes
  if self.task == 'qa' then
    self.num_classes = 2
  else
    error("not possible task!")
  end
  if self.neg_mode == 1 then
    print('Random generate negative samples')
  elseif self.neg_mode == 2 then
    print('Max generate negative samples')
  else
    print('Mix generate negative samples')
  end  
  print('number of negative pairs: '..self.num_pairs)	
  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }
  dofile 'PairwiseLossCriterion.lua'
  if self.loss_mode == 1 then
    self.criterion = nn.MultiMarginCriterion() --PairwiseLossCriterion()--nn.ClassNLLCriterion();
    print('set MultiMargin Loss Criterion\n')
  elseif self.loss_mode == 2 then
    self.criterion = nn.PairwiseLossCriterion()--nn.ClassNLLCriterion();
    print('set Self-Defined Standard Loss Criterion\n')
  end
  ----------------------Combination of ConvNets.---------------------------
  dofile 'models.lua'
  print('<model> creating a fresh model')
  self.epoch_counter = 0
  -- Type of model; Size of vocabulary; Number of output classes
  local modelName = 'deepQueryRankingNgramSimilarityOnevsGroupMaxMinMeanLinearExDGpPoinPercpt'
  --print(modelName)
  self.ngram = 3
  self.length = self.emb_dim
  self.rankModel2 = nn.ParallelTable()
  self.posModel = nn.Sequential()
  self.convModel = createModel(modelName, 10000, self.length, self.num_classes, self.ngram) 
  --print(self.convModel) 
  self.linearLayer = self:LinearLayer()
  self.posModel:add(self.convModel)
  self.posModel:add(self.linearLayer)
  self.params, self.grad_params = self.posModel:getParameters()
  -- clone negative model
  self.negModel = self.posModel:clone('weight','bias','gradWeight','gradBias')
  -- combine pos and neg model
  self.rankModel2:add(self.posModel)
  self.rankModel2:add(self.negModel)
  self.rankModel = nn.Sequential()
  self.rankModel:add(self.rankModel2)
  self.rankModel:add(nn.JoinTable(1))
  --print(self.posModel:parameters()[1]:norm())
  --print(self.negModel:parameters()[1]:norm())
  --print(self.convModel:parameters()[1][1]:norm())
  --print(self.softMaxC:parameters()[1][1]:norm())
end

function PairwiseConv:LinearLayer()
  local maxMinMean = 3
  local separator = (maxMinMean+1)*self.mem_dim
  local modelQ1 = nn.Sequential()	
  local ngram = self.ngram
  local items = (ngram+1)*3  		
  --local items = (ngram+1) -- no Min and Mean
  local NumFilter = self.length --300
  local conceptFNum = 20	
  --inputNum = 2*items*items/3+NumFilter*items*items/3+3*NumFilter*(2+ngram+1)+(2+NumFilter)*3*ngram*conceptFNum+3*NumFilter*(2+ngram*conceptFNum) --PoinPercpt model!
  inputNum = 2*items*items/3+NumFilter*items*items/3+6*NumFilter+(2+NumFilter)*2*ngram*conceptFNum -- old EMNLP model inputNum
  if self.dropout_mode == 1 then 
    modelQ1:add(nn.Dropout(0.5))
    print('Add dropout layer')
  end
  modelQ1:add(nn.Linear(inputNum, 1))
  --modelQ1:add(nn.Tanh())
  --modelQ1:add(nn.Linear(50, 1))
  return modelQ1
end

function PairwiseConv:trainCombineOnly(dataset)
  --local classes = {1,2}
  --local confusion = optim.ConfusionMatrix(classes)
  --confusion:zero()
  self.rankModel:training()
  self.epoch_counter = self.epoch_counter+1
  train_looss = 0.0
  local training_sample = 0
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  local features = nil 
  if self.epoch_counter > 1 and ( self.neg_mode == 2 or self.neg_mode == 3) then
    print('get CNN features')
    features = self:getTrainingFeatures(indices, dataset)
  end
  for i = 1, dataset.size, self.batch_size do

    local batch_size = 1 --math.min(i + self.batch_size - 1, dataset.size) - i + 1
    -- get target distributions for batch
    local targets = torch.zeros(batch_size, self.num_classes)
    for j = 1, batch_size do
      local sim  = -0.1
      if self.task == 'qa' then
        sim = dataset.labels[indices[i + j - 1]] + 1 
      else
	error("not possible!")
      end
      local ceil, floor = math.ceil(sim), math.floor(sim)
      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil - sim
        targets[{j, ceil}] = sim - floor
      end--]]
    end
    
    local feval = function(x)
      self.grad_params:zero()
      local loss = 0
      local batch_loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        -- skip if we met a negative <query, doc> pair
        if dataset.labels[idx] > 0 then
          local query, pos_doc = dataset.lsents[idx], dataset.rsents[idx]
          local query_vector = self.emb_vecs:index(1, query:long()):double()
          local pos_vector = self.emb_vecs:index(1, pos_doc:long()):double()
 	  local numrels = self:getNumRels(idx, dataset)
          local negIdxs = nil
          if self.neg_mode == 1 then
            negIdxs = self:randomNegSample(idx, dataset, self.num_pairs)
          elseif self.neg_mode == 2 then
            negIdxs = self:selectNegSample(idx, dataset, features, self.num_pairs)
          elseif self.neg_mode == 3 then
            negIdxs = self:mixNegSample(idx, dataset, features, self.num_pairs)
          end
  	  --print(negIdxs)
	  for negIdx, cos_score in pairs(negIdxs) do	  
	    training_sample = training_sample+1
            local neg_doc = dataset.rsents[negIdx]
	    local neg_vector = self.emb_vecs:index(1, neg_doc:long()):double()
	    local output = self.rankModel:forward({{query_vector, pos_vector}, {query_vector, neg_vector}})
            local weight = nil
            if self.loss_mode == 1 then
              weight = 1
            elseif self.loss_mode == 2 then
              weight = {query_weight = 1.0, pair_weight = 1.0}
            end
            local loss = self.criterion:forward(output, weight)
            -- print following info for debug	    
            --local loss = self.criterion:forward(output, 1)
	    --print(self.posModel:parameters()[1]:norm())
  	    --print(self.negModel:parameters()[1]:norm())
	    --print('pos index:' .. idx .. 'neg index:' .. negIdx)
	    --print('query weight:' .. weight['query_weight'], ', pair weight:' .. weight['pair_weight'])  
            --print(query, pos_doc, neg_doc)
	    --print(output[1], output[2], loss)
	    batch_loss = loss + batch_loss
 	    train_looss = loss + train_looss
	    local sim_grad = self.criterion:backward(output, weight) 
            self.rankModel:backward({{query_vector, pos_vector}, {query_vector, neg_vector}}, sim_grad)	    
	  end
        end
      end
      -- regularization
      batch_loss = batch_loss/5 + 0.5 * self.reg * self.params:norm()^2
      self.grad_params:add(self.reg, self.params)
      return batch_loss, self.grad_params
    end
    _, fs  = optim.sgd(feval, self.params, self.optim_state)
    --train_looss = train_looss + fs[#fs]
  end
  print('Num of training sample: ' .. training_sample)
  print('Loss: ' .. train_looss)
end

-- Get Intermediate Conv Features for each training instance during a training epoch
function PairwiseConv:getTrainingFeatures(indices, dataset)
  local features = {} 
  local start = sys.clock()
  for i = 1, dataset.size, 1 do
    local idx = indices[i]
    local query, doc = dataset.lsents[idx], dataset.rsents[idx]
    local query_vec = self.emb_vecs:index(1, query:long()):double()
    local doc_vec = self.emb_vecs:index(1, doc:long()):double()
    local feat = self.convModel:forward({query_vec, doc_vec}):clone()
    features[idx] = feat:clone()
  end
  return features
end

-- Randomly generate negative documents
function PairwiseConv:randomNegSample(posIndex, dataset, num_sample)
  local neg_docs = self:getNegIndices(posIndex, dataset)
  local negIdxs = {}
  local neg_count = 0
  while neg_count < math.min(num_sample, #neg_docs) do
    local randomIdx = neg_docs[math.random(#neg_docs)]
    if negIdxs[randomIdx] == nil then
      negIdxs[randomIdx] = 1
      neg_count = neg_count + 1
    end
  end
  return negIdxs
end

-- Select the most similar negative documents
function PairwiseConv:selectNegSample(posIndex, dataset, features, num_sample)
  if features == nil or #features == 0 then
    return self:randomNegSample(posIndex, dataset, num_sample)
  end
  local dist_table = self:getSortedDistTable(posIndex, dataset, features)
  local negIdxs = {}
  for ii = 1, math.min(num_sample, #dist_table) do
  end
  table.sort(dist_table, function(a,b) return a[2]>b[2] end)
  return dist_table
end

function PairwiseConv:getNumRels(posIndex, dataset)
  if dataset.boundary ~= nil then
    for tmpIndex = 1, dataset.boundary:size(1) do
      if dataset.boundary[tmpIndex] >= posIndex then
        return dataset.numrels[tmpIndex-1]
      end
    end
  end
  print('error in getNumRels')
  return -1
end

function PairwiseConv:getNegIndices(posIndex, dataset)
   -- step 1: Find the query boundary, i.e, [1, 1000] 
  local queryStartIdx, queryEndIdx = 0, 0
  if dataset.boundary ~= nil then
    for tmpIndex = 1, dataset.boundary:size(1) do
      if dataset.boundary[tmpIndex] >= posIndex then
	queryStartIdx = dataset.boundary[tmpIndex-1]+1
	queryEndIdx = dataset.boundary[tmpIndex]
        break
      end
    end
  end
  -- Step 2: Generate the negative index randomly in the query boundary.
  local queryInterval = queryEndIdx - queryStartIdx
  local neg_docs = {}
  for queryIter = queryStartIdx, queryEndIdx, 1 do
    if dataset.labels[queryIter] == 0 then
      neg_docs[#neg_docs+1] = queryIter
    end
  end
  return neg_docs 
end

-- Predict the similarity of a sentence pair.
function PairwiseConv:predictCombination(lsent, rsent)
  self.rankModel:evaluate()
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()
  
  local part2 = self.convModel:forward({linputs, rinputs})
  local output = self.linearLayer:forward(part2)
  local val = -1.0
  if self.task == 'qa' then
    return output
  else
    error("not possible task")
  end
  return val
end

-- Produce similarity predictions for each sentence pair in the dataset.
function PairwiseConv:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
  end
  return predictions
end

function PairwiseConv:print_config()
  local num_params = self.params:nElement()

  print('num params: ' .. num_params)
  print('word vector dim: ' .. self.emb_dim)
  print('regularization strength: ' .. self.reg)
  print('minibatch size: ' .. self.batch_size)
  print('learning rate: ' .. self.learning_rate)
  print('model structure: ' .. self.structure)
  print('number of hidden layers: ' .. self.num_layers)
  print('number of neurons in hidden layer: ' .. self.mem_dim)
end

function PairwiseConv:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predictCombination(lsent, rsent)
  end
  return predictions
end

function PairwiseConv:save(path)
  local config = {
    batch_size    = self.batch_size,
    emb_vecs      = self.emb_vecs:float(),
    learning_rate = self.learning_rate,
    num_layers    = self.num_layers,
    sim_nhidden   = self.sim_nhidden,
    reg           = self.reg,
    structure     = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end
