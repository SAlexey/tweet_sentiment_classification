import gzip
import numpy as np


def normalized_compression_distance(seq1: str, seq2: str) -> float:
    """
    Calculate the normalized compression distance between two strings.
    The normalized compression distance is a measure of the similarity between two strings.
    It is calculated as the difference between the length of the compressed concatenation of the two strings
    and the length of the compressed versions of the two strings, divided by the length of the longer of the two
    compressed strings.
    The compressed strings are calculated using the gzip algorithm.

    Args:
        seq1 (str): The first string.
        seq2 (str): The second string.

    Returns:
        float: The normalized compression distance between the two strings.
    """

    seq1_comp = gzip.compress(seq1.encode())
    seq2_comp = gzip.compress(seq2.encode())
    seq1seq2_comp = gzip.compress(f"{seq1} {seq2}".encode())
    ncd = (len(seq1seq2_comp) - min(len(seq1_comp), len(seq2_comp))) / max(
        len(seq1_comp), len(seq2_comp)
    )
    return ncd


def shanons_entropy(seq: str) -> float:
    """
    Calculate the Shannon entropy of a string.
    The Shannon entropy is a measure of the amount of information in a string.
    It is calculated as the sum of the probability of each character in the string multiplied by the logarithm
    of the probability of that character.

    Args:
        seq (str): The string for which to calculate the Shannon entropy.

    Returns:
        float: The Shannon entropy of the string.
    """

    # Calculate the frequency of each character in the string
    freqs = {char: seq.count(char) / len(seq) for char in set(seq)}

    # Calculate the Shannon entropy
    entropy = -sum(prob * np.log2(prob) for prob in freqs.values())
    return entropy


def information_content(seq: str) -> float:
    """
    Calculate the information content of a string.
    The information content is a measure of the amount of information in a string.
    It is calculated as the Shannon entropy of the string divided by the maximum possible Shannon entropy
    for a string of the same length.

    Args:
        seq (str): The string for which to calculate the information content.

    Returns:
        float: The information content of the string.
    """

    # Calculate the Shannon entropy of the string
    entropy = shanons_entropy(seq)

    # Calculate the maximum possible Shannon entropy for a string of the same length
    max_entropy = np.log2(len(set(seq)))

    # Calculate the information content
    info_content = entropy / max_entropy
    return info_content
