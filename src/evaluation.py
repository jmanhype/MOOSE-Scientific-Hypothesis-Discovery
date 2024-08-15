from sentence_transformers import SentenceTransformer, util
from rouge import Rouge
import logging

def hypothesis_evaluation(example, pred, trace=None, frac=0.5):
    rouge = Rouge()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_text(text):
        return ' '.join(text.lower().split())

    def calculate_rouge(prediction, ground_truth):
        scores = rouge.get_scores(prediction, ground_truth)
        return scores[0]['rouge-l']['f']

    def calculate_semantic_similarity(prediction, ground_truth):
        embeddings1 = model.encode([prediction], convert_to_tensor=True)
        embeddings2 = model.encode([ground_truth], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2).item()

    prediction = normalize_text(pred.hypothesis)
    ground_truth = normalize_text(example.hypothesis)
    rouge_score = calculate_rouge(prediction, ground_truth)
    semantic_similarity = calculate_semantic_similarity(prediction, ground_truth)
    combined_score = (rouge_score + semantic_similarity) / 2
    logging.info(f'Evaluation scores - ROUGE-L: {rouge_score:.4f}, Semantic Similarity: {semantic_similarity:.4f}, Combined: {combined_score:.4f}')
    logging.info(f'Generated hypothesis: {prediction}')
    logging.info(f'Ground truth: {ground_truth}')
    return combined_score >= frac, combined_score  # Return both the boolean result and the actual score

def evaluate(compiled_module, devset):
    results = []
    scores = []
    for example in devset:
        pred = compiled_module(example.observation)
        result, score = hypothesis_evaluation(example, pred)
        results.append(result)
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    logging.info(f'Average evaluation score: {avg_score:.4f}')
    return results, avg_score
