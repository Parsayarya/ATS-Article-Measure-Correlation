import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import time

def encode(texts):
    """
    Encode a list of texts using TF-IDF vectorization.

    Parameters:
    texts (list of str): The texts to be encoded.

    Returns:
    np.array: An array of TF-IDF encoded vectors.
    """
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()

def calculate_correlation(encodings, reference_encoding):
    """
    Calculate the cosine correlation between a set of encodings and a reference encoding.

    Parameters:
    encodings (np.array): The encoded vectors.
    reference_encoding (np.array): The reference vector for correlation calculation.

    Returns:
    np.array: Array of correlation values.
    """
    return np.dot(encodings, reference_encoding) / (np.linalg.norm(reference_encoding) * np.linalg.norm(encodings, axis=1))

def main():
    start_time = time.time()

    # Load datasets
    df = pd.read_csv('MeasureCorpus.csv')
    text1, text2 = df[df['Document_Number'].isin([360, 321])].Content.values
    encoding1, encoding2 = encode([text1, text2])

    df2 = pd.read_csv('FinalCorpus4.csv')
    encodings = encode(df2.text.tolist())

    # Calculate correlations
    M360, M321 = calculate_correlation(encodings, encoding1), calculate_correlation(encodings, encoding2)

    # Update DataFrame and save
    df2['M360_Correlation'], df2['M321_Correlation'] = M360, M321
    df2.drop(columns=['text'], inplace=True)

    df2.sort_values(by=['M360_Correlation'], inplace=True).to_csv('M360_Correlation_sorted_tfidf.csv', index=False)
    df2.sort_values(by=['M321_Correlation'], inplace=True).to_csv('M321_Correlation_sorted_tfidf.csv', index=False)

    # Cleanup
    del df, df2, encodings, M321, M360
    gc.collect()

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
