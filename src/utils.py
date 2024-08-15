import dspy
import random

def generate_and_load_trainset(num_examples=20):
    observations = [
        'An increase in atmospheric CO2 correlates with a rise in global temperatures.',
        'Marine biodiversity is declining at an alarming rate due to overfishing.',
        'AI is increasingly being used in personalized medicine to predict patient outcomes.',
        # Add more scientific observations here
    ]
    hypotheses = [
        'Increasing atmospheric CO2 levels may enhance plant photosynthesis rates under certain conditions, potentially affecting food security.',
        'Overfishing might lead to a shift in marine ecosystems, favoring species with rapid reproduction cycles.',
        'AI-driven models could identify novel biomarkers for disease prediction and treatment customization.',
        # Add corresponding hypotheses here
    ]
    trainset = []
    for _ in range(num_examples):
        idx = random.randint(0, len(observations) - 1)
        example = dspy.Example(observation=observations[idx], hypothesis=hypotheses[idx])
        trainset.append(example.with_inputs('observation'))  # Specify 'observation' as input
    return trainset
