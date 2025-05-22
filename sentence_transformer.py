from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter
from itertools import permutations
import matplotlib.pyplot as plt

ds = load_dataset("okite97/news-data")
ds_train = ds['train']

categories_to_drop = ['tech', 'entertainment', 'health']
print(f"Original dataset size: {len(ds_train)}")
print(f"Original categories: {Counter(ds_train['Category'])}")

filtered_ds = ds_train.filter(lambda example: example['Category'] not in categories_to_drop)
print(f"Filtered dataset size: {len(filtered_ds)}")
print(f"Categories after filtering: {Counter(filtered_ds['Category'])}")

unique_categories = sorted(list(set(filtered_ds['Category'])))

excerpts = filtered_ds['Excerpt']
cat_to_id = {cat: i for i, cat in enumerate(unique_categories)}
id_to_cat = {i: cat for cat, i in cat_to_id.items()}
y_true_original = np.array([cat_to_id[cat] for cat in filtered_ds['Category']])


st_model_name = 'all-MiniLM-L6-v2'
st_model = SentenceTransformer(st_model_name)
st_embedding_dim = st_model.get_sentence_embedding_dimension()



X_embeddings = st_model.encode(excerpts, show_progress_bar=True, batch_size=64)


valid_indices = np.where(X_embeddings.sum(axis=1) != 0)[0]
y_true = y_true_original
if len(valid_indices) < X_embeddings.shape[0]:
    X_embeddings = X_embeddings[valid_indices]
    y_true = y_true_original[valid_indices]
else:
    print("no zero vec")




pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_embeddings)

plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i, category_name in id_to_cat.items():
    plt.scatter(X_pca_2d[y_true == i, 0], X_pca_2d[y_true == i, 1],
                color=colors[i % len(colors)], label=category_name, alpha=0.7)
plt.title('Sentence Embeddings after PCA to 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()



pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_embeddings)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, category_name in id_to_cat.items():
    ax.scatter(X_pca_3d[y_true == i, 0], X_pca_3d[y_true == i, 1], X_pca_3d[y_true == i, 2],
                color=colors[i % len(colors)], label=category_name, alpha=0.7)
ax.set_title('Sentence Embeddings after PCA to 3 Components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()



pca_full = PCA(n_components=None, random_state=42)
pca_full.fit(X_embeddings)
explained_variance_ratio_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative, marker='.', linestyle='-')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Explained Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Explained Variance')

n_comp_90 = np.argmax(explained_variance_ratio_cumulative >= 0.90) + 1
plt.axvline(x=n_comp_90, color='r', linestyle=':', label=f'{n_comp_90} comps for 90%')


n_comp_95 = np.argmax(explained_variance_ratio_cumulative >= 0.95) + 1
plt.axvline(x=n_comp_95, color='g', linestyle=':', label=f'{n_comp_95} comps for 95%')

plt.legend(loc='best')
plt.ylim(0, 1.05)
if len(explained_variance_ratio_cumulative) > 50:
    plt.xticks(np.arange(0, len(explained_variance_ratio_cumulative)+1, step=max(1, len(explained_variance_ratio_cumulative)//20) ))
plt.tight_layout()
plt.show()



k = len(unique_categories)

max_pca_components = min(X_embeddings.shape[0], X_embeddings.shape[1])
pca_n_components_list = [x for x in range(1, 384) if x % 24 == 0]

if not pca_n_components_list and max_pca_components > 1:
    pca_n_components_list = [max_pca_components-1 if max_pca_components > 1 else 1]
    if max_pca_components <=1 : pca_n_components_list = []
elif max_pca_components <= 1:
    pca_n_components_list = []

if X_embeddings.shape[1] > 0:
    if not pca_n_components_list or X_embeddings.shape[1] > max(pca_n_components_list, default=0):
        if X_embeddings.shape[1] not in pca_n_components_list:
            pca_n_components_list.append(X_embeddings.shape[1])
    pca_n_components_list = sorted(list(set(pca_n_components_list)))

print(f"pcacomponents: {pca_n_components_list}")

results = []

if not X_embeddings.shape[0] > k:
    print(f"Not enough samples")
else:
    for n_components in pca_n_components_list:
        print(f"\n--- n_components: {n_components} ---")

        X_reduced = None
        if n_components == X_embeddings.shape[1]:
            X_reduced = X_embeddings.copy()
        elif n_components < X_embeddings.shape[1] and n_components > 0:
            if n_components > X_embeddings.shape[0]:
                continue
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X_embeddings)
            print(f"Shape after PCA: {X_reduced.shape}")


        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels_pred = kmeans.fit_predict(X_reduced)

        best_accuracy = 0.0
        best_mapping = {}
        cluster_label_set = sorted(list(set(cluster_labels_pred)))
        true_label_set = sorted(list(set(y_true)))

        current_k_pred = len(cluster_label_set)

        if current_k_pred < k:
            perms = permutations(true_label_set, current_k_pred) 
        else:
            perms = permutations(true_label_set)

        for p_tuple in perms:

            current_mapping = {cluster_idx: p_val for cluster_idx, p_val in zip(cluster_label_set, p_tuple)}
            mapped_labels = np.array([current_mapping.get(pred_label, -1) for pred_label in cluster_labels_pred]) 
            if -1 in mapped_labels and current_k_pred < k :
                continue


            current_accuracy = accuracy_score(y_true, mapped_labels)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_mapping = current_mapping
        
        results.append({'n_components': n_components, 'accuracy': best_accuracy, 'mapping': best_mapping})



print("N_Components | Accuracy")
print("------------------------")
accuracies = []
components_labels = []
for res in results:
    print(f"{res['n_components']:<12} | {res['accuracy']:.4f}")
    if res['n_components'] == X_embeddings.shape[1]:
        components_labels.append(f"Full ST ({res['n_components']})")
    else:
        components_labels.append(str(res['n_components']))
    accuracies.append(res['accuracy'])


plt.figure(figsize=(12, 7))
plt.plot(components_labels, accuracies, marker='o', linestyle='-')
plt.title('K-Means Clustering Accuracy vs. Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Clustering Accuracy')
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
