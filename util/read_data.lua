--[[

  Functions for loading data from disk.

--]]

function similarityMeasure.read_embedding(vocab_path, emb_path)
  local vocab = similarityMeasure.Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function similarityMeasure.read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(math.max(len,3))
    local counter = 0
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    if len < 3 then
      for i = len+1, 3 do
	sent[i] = vocab:index('unk') -- sent[len]
      end
    end
    if sent == nil then print('line: '..line) end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function similarityMeasure.read_relatedness_dataset(dir, vocab, task)
  local dataset = {}
  dataset.vocab = vocab
  if task == 'twitter' then
	file1 = 'tokenize_query2.txt'
	file2 = 'tokenize_doc2.txt'
  else 
	file1 = 'a.toks'
	file2 = 'b.toks'
  end
  dataset.lsents = similarityMeasure.read_sentences(dir .. file1, vocab)
  dataset.rsents = similarityMeasure.read_sentences(dir .. file2, vocab)
  dataset.size = #dataset.lsents
  local id_file = io.open(dir .. 'id.txt', 'r')
  local sim_file = torch.DiskFile(dir .. 'sim.txt')
  dataset.ids = {}
  dataset.labels = torch.Tensor(dataset.size)
  if task == 'twitter' or task == 'qa' then  
    local boundary_file, _ = io.open(dir .. 'boundary.txt')
    local numrels_file = torch.DiskFile(dir .. 'numrels.txt')
    local boundary, counter = {}, 0
    while true do
      line = boundary_file:read()
      if line == nil then break end
      counter = counter + 1
      boundary[counter] = tonumber(line)
    end
    boundary_file:close()  
    dataset.boundary = torch.IntTensor(#boundary)
    for counter, bound in pairs(boundary) do
      dataset.boundary[counter] = bound
    end  
    -- read numrels data
    dataset.numrels = torch.IntTensor(#boundary-1)
    for i = 1, #boundary-1 do
      dataset.numrels[i] = numrels_file:readInt()
    end
    numrels_file:close()
  end

  for i = 1, dataset.size do
    dataset.ids[i] = id_file:read()
    if task == 'sic' then
    	dataset.labels[i] = 0.25 * (sim_file:readDouble() - 1) -- sic data
    elseif task == 'vid' then
	dataset.labels[i] = 0.2 * (sim_file:readDouble()) -- vid data
    else
    	dataset.labels[i] = (sim_file:readDouble()) -- twi and msp
    end
  end
  id_file:close()
  sim_file:close()
  return dataset
end

