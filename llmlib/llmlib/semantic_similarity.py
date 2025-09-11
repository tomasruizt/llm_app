from sentence_transformers import SentenceTransformer, util


class SemanticSimilarity:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def cos_sim(self, s1: str, s2: str) -> float:
        """Compute the cosine similarity between two strings"""
        embedding1 = self.model.encode(s1, convert_to_tensor=True)
        embedding2 = self.model.encode(s2, convert_to_tensor=True)
        return util.cos_sim(embedding1, embedding2).item()
