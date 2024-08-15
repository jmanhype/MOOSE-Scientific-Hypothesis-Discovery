import os
from dotenv import load_dotenv
import logging
import json
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from scientific_discovery import ScientificHypothesisDiscovery
from evaluation import hypothesis_evaluation, evaluate
from utils import generate_and_load_trainset

# Load environment variables and setup logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Setup and compilation
    dataset = generate_and_load_trainset()
    trainset = dataset[:-5]  # Use all but last 5 examples as train set
    devset = dataset[-5:]  # Use last 5 examples as dev set

    # Create a new ScientificHypothesisDiscovery instance
    discovery_instance = ScientificHypothesisDiscovery()
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=hypothesis_evaluation,
        num_candidate_programs=5,
        max_bootstrapped_demos=3,
        max_labeled_demos=10,
        max_rounds=2,
        num_threads=1,
        max_errors=10
    )

    try:
        compiled_discovery = teleprompter.compile(discovery_instance, trainset=trainset, valset=devset)
    except Exception as e:
        logging.error(f"Error during compilation: {e}")
        compiled_discovery = discovery_instance

    # Save the compiled program
    try:
        compiled_program_json = json.dumps(compiled_discovery.dump_state(), indent=2)
        with open("compiled_scientific_hypothesis_discovery.json", "w") as f:
            f.write(compiled_program_json)
        print("Program saved to compiled_scientific_hypothesis_discovery.json")
    except Exception as e:
        logging.error(f"Error saving compiled program: {e}")

    # Evaluate the compiled program
    try:
        results, avg_score = evaluate(compiled_discovery, devset)
        print("Evaluation Results:")
        print(f"Binary results: {results}")
        print(f"Average score: {avg_score:.4f}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        print("An error occurred during evaluation. Please check the logs for details.")

    # Interactive loop
    while True:
        input_type = input("Enter 'text' for text input or 'voice' for voice input (or 'quit' to exit): ")

        if input_type.lower() == 'quit':
            break

        if input_type.lower() == 'voice':
            file_path = input("Enter the path to your audio file: ")
            try:
                observation = compiled_discovery.transcribe(file_path)
                print(f"Transcribed observation: {observation}")
            except Exception as e:
                logging.error(f"Error during transcription: {e}")
                print("An error occurred while transcribing the audio. Please try again.")
                continue
        else:
            observation = input("Enter an observation: ")

            try:
                prediction = compiled_discovery(observation)
                print(f"Observation: {prediction.observation}")
                print(f"Identified Jargon Terms:")
                for term, definitions in prediction.jargon_definitions.items():
                    print(f"  - {term}:")
                    for source, definition in definitions.items():
                        print(f"    {source}: {definition}")
                print(f"Identified Context: {prediction.context}")
                print(f"Reasoning:")
                print(prediction.reasoning)
                print(f"Hypothesis: {prediction.hypothesis}")
                print("Retrieved Passages:")
                for i, passage in enumerate(prediction.retrieved_passages, 1):
                    print(f"Passage {i}: {passage[:200]}...")  # Print first 200 characters of each passage

                # Ask if the user wants voice output
                voice_output = input("Do you want to hear the hypothesis spoken? (yes/no): ")
                if voice_output.lower() == 'yes':
                    compiled_discovery.speak(prediction.hypothesis)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                print("An error occurred while processing the observation. Please try again.")
                print("" + "-"*50 + "")  # Add a separator between iterations
    print("Thank you for using ScientificHypothesisDiscovery. Goodbye!")

if __name__ == '__main__':
    main()