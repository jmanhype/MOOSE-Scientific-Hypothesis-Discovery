import dspy
import asyncio
import aiohttp
from cachetools import TTLCache
import logging
import backoff

class QueryScientificJargon(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.rate_limit = 1.0
        self.local_dictionary = {
            'Hypothetical induction': 'A reasoning process where scientists propose hypotheses to explain observations.',
            'Open-domain': 'Refers to data or questions that are not confined to a specific subject area.',
            'AI': 'Artificial Intelligence; the simulation of human intelligence processes by machines.',
            'Personalized medicine': 'A medical model that separates people into different groups—with medical decisions, practices, and/or products being tailored to the individual patient.',
            'Patient outcomes': 'The results of medical treatment, including quality of life, side effects, and mortality rates.',
            'Marine biodiversity': 'The variety of life in marine ecosystems, including the diversity of plants, animals, and microorganisms.',
            'Overfishing': 'The removal of a species of fish from a body of water at a rate that the species cannot replenish, resulting in diminished fish populations.',
            'Climate change': 'Long-term shifts in temperatures and weather patterns, primarily caused by human activities.',
            'Global warming': 'The long-term heating of Earth's surface observed since the pre-industrial period due to human activities.',
        }

    async def forward(self, jargon_terms):
        jargon_definitions = {}
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_jargon_definition(term, session) for term in jargon_terms]
            results = await asyncio.gather(*tasks)
        for term, definitions in results:
            jargon_definitions[term] = definitions
        return jargon_definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def get_jargon_definition(self, term, session):
        if term in self.cache:
            return term, self.cache[term]
        logging.info(f'Querying for term: {term}')

        # Check local dictionary first
        if term.lower() in self.local_dictionary:
            self.cache[term] = {'local': self.local_dictionary[term.lower()]}
            return term, self.cache[term]
        definitions = {
            'scientific_sources': await self.query_scientific_sources(term, session),
        }
        # Remove None values
        definitions = {k: v for k, v in definitions.items() if v is not None}
        if not definitions:
            # Use GPT-3 as a fallback for definition
            definitions['gpt'] = await self.query_gpt(term)
        self.cache[term] = definitions
        return term, definitions

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def query_scientific_sources(self, term, session):
        try:
            await asyncio.sleep(self.rate_limit)  # Rate limiting
            url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{term}'
            async with session.get(url, headers={'User-Agent': 'ScienceHypothesisBot/1.0'}) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('extract')
                else:
                    logging.warning(f'Scientific source returned status {response.status} for term {term}')
        except Exception as e:
            logging.error(f'Error querying scientific sources for {term}: {e}')
        return None

    async def query_gpt(self, term):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f'Provide a brief definition for the term '{term}' in the context of scientific research:'
                response = dspy.Predict('term -> definition')(term=prompt).definition
                return response.strip()
            except Exception as e:
                logging.warning(f'Error querying GPT for {term} (attempt {attempt + 1}/{max_retries}): {e}')
                if attempt == max_retries - 1:
                    logging.error(f'Failed to query GPT for {term} after {max_retries} attempts')
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
