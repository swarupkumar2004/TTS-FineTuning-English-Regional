import numpy as np
from scipy.spatial import distance

def compare_phonemes(reference, output):
    # Compare phoneme representations
    return distance.hamming(reference, output)

def calculate_mos_score(predictions, targets):
    # Mock MOS score calculation (for subjective evaluation, need human input)
    scores = np.random.uniform(4.0, 5.0, len(predictions))
    return np.mean(scores)

