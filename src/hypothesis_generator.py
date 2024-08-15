import dspy

class HypothesisGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_hypothesis = dspy.ChainOfThought('observation, jargon_definitions, context, retrieved_passages -> reasoning, novel_hypothesis')

    def forward(self, observation, jargon_definitions, context, retrieved_passages):
        result = self.generate_hypothesis(
            observation=observation,
            jargon_definitions=jargon_definitions,
            context=context,
            retrieved_passages=retrieved_passages
        )
        return result.reasoning, result.novel_hypothesis
