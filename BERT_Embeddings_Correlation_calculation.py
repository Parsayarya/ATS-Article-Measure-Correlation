import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import gc
from tqdm import tqdm
import time

# Initialize BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode(texts, max_length=512, batch_size=64):
    """
    Encode a list of texts using BERT.

    Parameters:
    texts (list of str): The texts to be encoded.
    max_length (int): Maximum length of the tokenized text.
    batch_size (int): The batch size for processing.

    Returns:
    np.array: An array of BERT embeddings.
    """
    all_embeddings = []

    # Process texts in batches to manage memory usage
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)

def main():
    start_time = time.time()

    # Load and process datasets
    df = pd.read_csv('MeasureCorpus.csv')
    text1, text2 = df[df['Document_Number'].isin([360, 321])].Content.values
    encoding1, encoding2 = encode([text1, text2])

    df2 = pd.read_csv('FinalCorpus4.csv')
    encodings = encode(df2.text.tolist())
    enc = pd.DataFrame(encodings)
    enc.to_csv('corpus_encodings.csv', index=False)

    # Calculate correlations
    M360 = np.dot(encodings, encoding1) / (np.linalg.norm(encoding1) * np.linalg.norm(encodings, axis=1))
    M321 = np.dot(encodings, encoding2) / (np.linalg.norm(encoding2) * np.linalg.norm(encodings, axis=1))

    # Update DataFrame and save
    df2['M360_Correlation'], df2['M321_Correlation'] = M360, M321
    df2.drop(columns=['text'], inplace=True)

    df2.sort_values(by=['M360_Correlation'], inplace=True).to_csv('M360_Correlation_sorted_base.csv', index=False)
    df2.sort_values(by=['M321_Correlation'], inplace=True).to_csv('M321_Correlation_sorted_base.csv', index=False)

    # Cleanup
    del df, df2, encodings, M321, M360
    gc.collect()

    # End timing
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
