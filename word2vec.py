from datasets import load_dataset
import numpy as np
import nltk
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from itertools import permutations


try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

ds = load_dataset("okite97/news-data")
ds_train = ds['train']

categories_to_drop = ['tech', 'entertainment', 'health']

filtered_ds = ds_train.filter(lambda example: example['Category'] not in categories_to_drop)

unique_categories = sorted(list(set(filtered_ds['Category'])))

excerpts = filtered_ds['Excerpt']


cat_to_id = {cat: i for i, cat in enumerate(unique_categories)}
y_true = np.array([cat_to_id[cat] for cat in filtered_ds['Category']])

tokenized_excerpts = [nltk.word_tokenize(excerpt.lower()) for excerpt in excerpts]

word2vec_model = Word2Vec(sentences=tokenized_excerpts, vector_size=1000, window=5, min_count=2, workers=4)
print("Word2Vec trained.")
word2vec_vector_size = word2vec_model.vector_size

def get_sentence_vector(tokens, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    index2word_set = set(model.wv.index_to_key)
    for word in tokens:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

X_word2vec = np.array([get_sentence_vector(tokens, word2vec_model, word2vec_vector_size) for tokens in tokenized_excerpts])

valid_indices = np.where(X_word2vec.sum(axis=1) != 0)[0]
if len(valid_indices) < X_word2vec.shape[0]:
    X_word2vec = X_word2vec[valid_indices]
    y_true = y_true[valid_indices]

k = 3 
max_pca_components = min(X_word2vec.shape[0], X_word2vec.shape[1])
pca_n_components_list = [1000]

result = {}

for n_components in pca_n_components_list:
    print(f"\n--- n_components = {n_components} ---")
    X_reduced = X_word2vec.copy()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    cluster_labels_pred = kmeans.fit_predict(X_reduced)
    best_accuracy = 0.0
    best_mapping = {}

    cluster_label_set = sorted(list(set(cluster_labels_pred)))
    
    true_label_set = sorted(list(set(y_true)))
    perms = permutations(true_label_set)


    for p in perms:
        current_mapping = {cluster_idx: true_label_val for cluster_idx, true_label_val in zip(cluster_label_set, p)}
        mapped_labels = np.array([current_mapping[pred_label] for pred_label in cluster_labels_pred])
        
        current_accuracy = accuracy_score(y_true, mapped_labels)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_mapping = current_mapping
    result={'n_components': n_components, 'accuracy': best_accuracy, 'mapping': best_mapping}


print("N_Components | Accuracy")
print("------------------------")
accuracies = []
components = []
print(f"{result['n_components']:<12} | {result['accuracy']:.4f}")
