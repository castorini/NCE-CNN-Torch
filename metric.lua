require('torch')

function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end

function match(x, y)
  local match = torch.sum(torch.eq(x, y))
  return match / x:size(1)
end

function mrr(score, qrels, boundary, num_rels) 
  local mrr_score = 0
  for qid = 1, boundary:size(1)-1 do
    local num_pairs = boundary[qid+1]-boundary[qid] --number of query-doc pairs
    local slice_score = torch.Tensor(num_pairs):copy(score[{{boundary[qid]+1, boundary[qid+1]}}])
    local sort_score, sort_index = torch.sort(slice_score, true) -- sort score from high to low    
    local new_qrels = torch.Tensor(num_pairs)
    local tp, ap = 0, 0
    --printf("qid:%d, size:%d\n", qid, -boundary[qid]+boundary[qid+1])  
    for i = 1, num_pairs do
      new_qrels[i] = qrels[boundary[qid]+sort_index[i]]
      if new_qrels[i] >= 1 then
        mrr_score = mrr_score + 1.0/i
        break
      end
    end
  end
  return mrr_score/(boundary:size(1)-1)
end

function map(score, qrels, boundary, num_rels)
  local map_score = 0
  for qid = 1, boundary:size(1)-1 do -- per query
    local num_pairs = boundary[qid+1]-boundary[qid] --number of query-doc pairs
    local slice_score = torch.Tensor(num_pairs):copy(score[{{boundary[qid]+1, boundary[qid+1]}}])
    local sort_score, sort_index = torch.sort(slice_score, true) -- sort score from high to low    
    local new_qrels = torch.Tensor(num_pairs)
    local tp, ap = 0, 0
    --printf("qid:%d, size:%d\n", qid, -boundary[qid]+boundary[qid+1])	
    for i = 1, num_pairs do
      new_qrels[i] = qrels[boundary[qid]+sort_index[i]]
      if new_qrels[i] >= 1 then
        tp = tp + 1
        ap = ap + tp/i
      end
    end
    --printf("qid:%d, tp:%d, num_rels:%d, map:%.4f\n", qid, tp, num_rels[qid], ap/num_rels[qid])
    if num_rels[qid] == 0 then
      map_score = map_score + 0
    else
      map_score = map_score + ap/num_rels[qid]
    end
  end
  return map_score/(boundary:size(1)-1)
end

function p_30(score, qrels, boundary)
  local p30_score = 0
  for qid = 1, boundary:size(1)-1 do -- per query
    local num_pairs = boundary[qid+1]-boundary[qid] --number of query-doc pairs
    local slice_score = torch.Tensor(num_pairs):copy(score[{{boundary[qid]+1, boundary[qid+1]}}])
    local sort_score, sort_index = torch.sort(slice_score, true) -- sort score from high to low
    local new_qrels = torch.Tensor(num_pairs)
    local tp = 0
    for i = 1, math.min(30, num_pairs) do
      new_qrels[i] = qrels[boundary[qid]+sort_index[i]]
      if new_qrels[i] >= 1 then
          tp = tp + 1
      end
    end
    --printf("qid:%d, p30:%.4f\n", qid, tp/30)
    p30_score = p30_score + tp/30
  end
  return p30_score/(boundary:size(1)-1)
end

