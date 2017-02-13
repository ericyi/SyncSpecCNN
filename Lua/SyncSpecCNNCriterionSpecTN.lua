require 'nn'
local SyncSpecCNNCriterionSpecTN, parent = torch.class('nn.SyncSpecCNNCriterionSpecTN', 'nn.Criterion')

function SyncSpecCNNCriterionSpecTN:__init()
    parent.__init(self)
    self.MSECriterion = nn.MSECriterion()
end

function SyncSpecCNNCriterionSpecTN:updateOutput(input, target)
    self.output = self.MSECriterion:updateOutput(input,target)
    return self.output
end

function SyncSpecCNNCriterionSpecTN:updateGradInput(input, target)
    self.gradInput = self.MSECriterion:updateGradInput(input,target)
    return self.gradInput
end

function SyncSpecCNNCriterionSpecTN:type(t)
    parent.type(self, t)
    self.MSECriterion:type(t)
    return self
end