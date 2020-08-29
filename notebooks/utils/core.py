import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def remove_missing_features(X, threshold=0.8, verbose=False):
    features = X.columns
    features_to_remove = []
    dataset_size = X.shape[0]
    for f in features:
        missing_count = X[X[f].isna()].shape[0]
        missing_ratio = missing_count / dataset_size
        if missing_ratio > threshold:
            features_to_remove.append(f) 
            if verbose:
                print("{:14}{:.3f}%".format(f, missing_ratio * 100))
    return X.drop(features_to_remove, axis=1)


def remove_single_values(X, verbose=False):
    features = X.columns
    features_to_remove = []
    for f in features:
        value_count = X[f].nunique()
        if value_count == 1:
            features_to_remove.append(f)
            if verbose:
                print('Removing ', f)
    return X.drop(features_to_remove, axis=1)


def plot_correlation_triangle(X):
    corr = X.corr()

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize=(15, 10))
    sns.heatmap(corr, mask=mask, center=0, cmap= 'coolwarm')
    plt.show()


def remove_highly_correlate_features(X, threshold=0.9):
    corr = X.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    to_remove = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(to_remove, axis=1)


def features_with_missing_values(X):
    features = X.columns
    dataset_size = X.shape[0]
    to_impute = []
    for f in features:
        feature_count = X[X[f].isna()].shape[0]
        if (feature_count != dataset_size) and (feature_count != 0):
            to_impute.append(f)
    return to_impute


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def root_mean_log_error(y_true, y_pred):
    try:
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    except ValueError:
        return -1000


def cross_val_regression(clf, X, y, cv=3):
    neg_mse_scores = cross_val_score(clf, X, y, scoring='neg_mean_squared_error', cv=cv)
    neg_log_scores = cross_val_score(clf, X, y, scoring='neg_mean_squared_log_error', cv=cv)
    return {
        'rmse': (np.sqrt(-neg_mse_scores)).mean(),
        'rmlse': (np.sqrt(-neg_log_scores)).mean()
    }


def kmeans_elbow_approach(max_clusters, X):
    sum_squared_distances = []
    for c in range(2, max_clusters + 1):
        kmeans = KMeans(random_state=RANDOM_STATE, n_clusters=c)
        kmeans.fit(X)
        sum_squared_distances.append(kmeans.inertia_)
    return sum_squared_distances

def plot_elbow_approach(max_clusters, sum_squared_distances):
    r = range(2, max_clusters + 1)
    plt.plot(r, sum_squared_distances, '-o')
    plt.xlabel('N Clusters')
    plt.ylabel('Sum of Squared Errors')
    plt.show()