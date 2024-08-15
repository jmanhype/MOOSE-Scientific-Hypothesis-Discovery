import dspy
import asyncio
import json
import logging
import openai
from query_jargon import QueryScientificJargon
from hypothesis_generator import HypothesisGenerator
from pydub import AudioSegment
from pydub.playback import play
import io
import os
from dotenv import load_dotenv

load_dotenv()

class ScientificHypothesisDiscovery(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.query_jargon_dictionary = QueryScientificJargon()
        
        # Initialize LM
        llm = dspy.OpenAI(model='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
        dspy.settings.configure(lm=llm)
        
        # Initialize RM
        try:
            colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
            dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
            logging.info("Successfully configured ColBERTv2 retrieval model")
        except Exception as e:
            logging.error(f"Failed to configure ColBERTv2 retrieval model: {e}")
            logging.warning("Falling back to default retrieval method")
        
        self.retrieve = dspy.Retrieve(k=num_passages)
        logging.info(f"Successfully initialized Retrieve module with k={num_passages}")
        
        self.identify_jargon = dspy.Predict("observation -> jargon_terms")
        self.identify_context = dspy.Predict("observation -> context")
        self.hypothesis_generator = HypothesisGenerator()
        
        # Set up OpenAI client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            logging.warning('OPENAI_API_KEY not found in environment variables. Some features may not work.')

    def forward(self, observation):
        try:
            jargon_terms = self.identify_jargon(observation=observation).jargon_terms.strip().split(',')
            jargon_terms = [term.strip() for term in jargon_terms if len(term.strip().split()) <= 3]  # Limit to terms with 3 words or less
            logging.info(f'Identified jargon terms: {jargon_terms}')
        except Exception as e:
            logging.error(f'Error in identify_jargon: {e}')
            jargon_terms = []
        try:
            jargon_definitions = asyncio.run(self.query_jargon_dictionary(jargon_terms))
            logging.info(f'Retrieved jargon definitions: {json.dumps(jargon_definitions, indent=2)}')
        except Exception as e:
            logging.error(f'Error in query_jargon_dictionary: {e}')
            jargon_definitions = {}
        try:
            context = self.identify_context(observation=observation).context.strip()
            logging.info(f'Identified context: {context}')
        except Exception as e:
            logging.error(f'Error in identify_context: {e}')
            context = ''
        relevant_passages = self.retrieve_relevant_passages(observation)
        if not relevant_passages:
            logging.warning('No relevant passages retrieved. Using a generic passage.')
            relevant_passages = ['This is a generic passage to provide some context for hypothesis generation.']
        try:
            reasoning, hypothesis = self.hypothesis_generator(
                observation=observation,
                jargon_definitions=json.dumps(jargon_definitions),
                context=context,
                retrieved_passages=json.dumps(relevant_passages)
            )
            logging.info(f'Generated hypothesis: {hypothesis}')
            logging.debug(f'Reasoning: {reasoning}')
        except Exception as e:
            logging.error(f'Error in generate_hypothesis: {e}')
            reasoning = 'Unable to generate reasoning due to an error.'
            hypothesis = 'Unable to generate a hypothesis at this time.'
        return dspy.Prediction(
            observation=observation,
            jargon_definitions=jargon_definitions,
            context=context,
            reasoning=reasoning,
            hypothesis=hypothesis,
            retrieved_passages=relevant_passages
        )

    def retrieve_relevant_passages(self, observation):
        try:
            result = self.retrieve(observation)
            if hasattr(result, 'passages'):
                logging.info(f'Successfully retrieved {len(result.passages)} passages')
                return result.passages
            elif isinstance(result, list):
                logging.info(f'Successfully retrieved {len(result)} passages')
                return result
            elif hasattr(result, 'topk'):
                logging.info(f'Successfully retrieved {len(result.topk)} passages')
                return result.topk
            else:
                logging.warning(f'Unexpected return type from retrieve method: {type(result)}')
                return self.fallback_retrieval(observation)
        except Exception as e:
            logging.error(f'Error in retrieve method: {str(e)}')
            return self.fallback_retrieval(observation)

    def fallback_retrieval(self, observation):
        logging.warning('Using fallback retrieval method')
        keywords = observation.split()[:5]  # Use first 5 words as keywords
        fallback_passages = [
            f'Passage related to {' '.join(keywords)}...',
            'General scientific knowledge passage...',
            'Placeholder for relevant scientific context...'
        ]
        logging.info(f'Generated {len(fallback_passages)} fallback passages')
        return fallback_passages

    def validate_passages(self, passages):
        if not passages:
            return False
        if not isinstance(passages, list):
            return False
        if not all(isinstance(p, str) for p in passages):
            return False
        return True

    def transcribe(self, file_path):
        with open(file_path, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                model='whisper-1',
                file=audio_file,
            )
        return transcript.text

    def generate_voice_audio(self, text: str):
        response = openai.audio.speech.create(
            model='tts-1-hd', voice='shimmer', input=text, response_format='mp3'
        )
        return response.content

    def speak(self, text: str):
        audio_bytes = self.generate_voice_audio(text)
        audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        play(audio)