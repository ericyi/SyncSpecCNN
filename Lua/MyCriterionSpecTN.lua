require 'nn'
local MyCriterionSpecTN, parent = torch.class('nn.MyCriterionSpecTN', 'nn.Criterion')

function MyCriterionSpecTN:__init()
    parent.__init(self)
    self.MSECriterion = nn.MSECriterion()
end

function MyCriterionSpecTN:updateOutput(input, target)
    self.output = self.MSECriterion:updateOutput(input,target)
    return self.output
end

function MyCriterionSpecTN:updateGradInput(input, target)
    self.gradInput = self.MSECriterion:updateGradInput(input,target)
    return self.gradInput
end

function MyCriterionSpecTN:type(t)
    parent.type(self, t)
    self.MSECriterion:type(t)
    return self
end