local PairwiseLossCriterion, parent = torch.class('nn.PairwiseLossCriterion', 'nn.Criterion')

function PairwiseLossCriterion:__init()
    parent.__init(self)
end

function PairwiseLossCriterion:updateOutput(input, weight)
  return weight['query_weight']*weight['pair_weight']*math.max(0, 1 - (input[1] - input[2]))/2
end

function PairwiseLossCriterion:updateGradInput(input, weight)
    local diff = 1 - (input[1] - input[2])
    self.gradInput = torch.zeros(2)
    if diff > 0 then
    	self.gradInput[1] = -0.5
	self.gradInput[2] = 0.5
    end
    return self.gradInput:mul(weight['query_weight']*weight['pair_weight'])
end
