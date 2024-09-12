
#It is more efficient to normalise data before calling Z
def Z(a,c):
    """
    Calculate the sum of exponentials of dot products between a unit vector and a cluster of points.
    
    Parameters
    ----------
    a : numpy.ndarray
        A unit vector.
    c : numpy.ndarray
        A cluster of points, where each row represents a point.

    Returns
    -------
    float
        The sum of exponentials of the dot product between a and each point in c.
    """
    def func(d):
        return jnp.exp(a.T @ d)

    # Vectorize func over c
    return jnp.sum(vmap(func)(c))


def _global_mean(X, labels, isotropy_function, **kwargs):
    """
    Calculate the weighted mean isotropy value of a clustering.
    
    Parameters
    ----------
    X : numpy.ndarray
        The data points.
    labels : numpy.ndarray
        The cluster labels for each data point.
    isotropy_function : callable
        The function to compute isotropy values for each cluster.
    **kwargs : dict
        Additional keyword arguments to pass to the isotropy function.

    Returns
    -------
    float
        The global mean isotropy value.

    Raises
    ------
    ValueError
        If the length of the labels array is not equal to the number of data points in X.
    """
    if len(labels) != X.shape[0]:
        raise ValueError(f"Invalid labels recieved. Expected {X.shape[0]} labels. Received {len(labels)}")
    def isotropy_for_cluster(label):
        cluster = X[labels == label]
        isotropy_val = isotropy_function(cluster, **kwargs)
        return isotropy_val, cluster.shape[0]

    # Vectorize isotropy_for_cluster over unique_labels
    isotropy_vals, cluster_sizes = vmap(isotropy_for_cluster)(labels)

    # Compute the weighted sum of isotropy values
    weighted_sum = jnp.sum(isotropy_vals * cluster_sizes)

    # Return the weighted sum normalized by the total number of data points in X
    return weighted_sum / X.shape[0]
    
    
def i_rnd(data, labels=None, unit_samples=1000):
    """
    Calculate the isotropy ratio, I_{rnd} of a dataset using random projections.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data points.
    labels : numpy.ndarray, optional
        The cluster labels for each data point. If given, calculate the isotropy ratio
        of the clustering instead of the dataset.
    unit_samples : int, optional
        The number of unit samples to generate for random projections.

    Returns
    -------
    float
        The isotropy ratio of the dataset or clustering.
    """
    
    if labels is not None:
        return _global_mean(data, labels, i_rnd, unit_samples=unit_samples)

    rng, key = jax.random.split(jax.random.PRNGKey(0))
    projections = jax.random.uniform(key, (data.shape[1], unit_samples))
    # Scale to unity
    projections = projections/norm(projections, ord=2, axis=1, keepdims=True)

    # Normalise data
    data = data-data.mean(axis=0)
    data = data/jnp.mean(norm(data, axis=1))

    # Define a function that computes Z(c[:,None], data)
    def compute_Z(c):
        return Z(c[:,None], data)

    # Vectorize compute_Z over projections.T
    z_vals = vmap(compute_Z)(projections.T)

    return float(jnp.min(z_vals) / jnp.max(z_vals))

    
def i_vec(data, labels=None):
    """
    Calculate the isotropy ratio, I_{vec} of a dataset using random projections.
    
    Parameters
    ----------
    data : numpy.ndarray
        The data points.
    labels : numpy.ndarray, optional
        The cluster labels for each data point. If given, calculate the isotropy ratio
        of the clustering instead of the dataset.
    
    Returns
    -------
    float
        The isotropy ratio of the dataset or clustering.
    """
    if labels is not None:
        return _global_mean(data, labels, i_vec)

    # Normalise data
    data = data-data.mean(axis=0)
    data = data/jnp.mean(norm(data, axis=1))

    u, s, v = jnp.linalg.svd(data.T @ data)

    # Define a function that computes Z(c, data)
    def compute_Z(c):
        return Z(c, data)

    # Vectorize compute_Z over u.T
    z_vals = vmap(compute_Z)(u.T)

    return float(jnp.min(z_vals) / jnp.max(z_vals))
    
def fractional_anisotropy(X,labels=None):
    """
    Calculate the fractional anisotropy of a dataset.
    
    Parameters
    ----------
    X : numpy.ndarray
        The data points.
    labels : numpy.ndarray, optional
        The cluster labels for each data point. If given, calculate the fractional anisotropy
        of the clustering instead of the dataset.

    Returns
    -------
    float
        The fractional anisotropy of the dataset or clustering.
    """
    
    if labels is not None:
        return _global_mean(X, labels, fractional_anisotropy)
    
    covariance_matrix = jnp.cov(X, rowvar=False)

    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance_matrix)

    # Calculate the trace of the tensor
    trace = jnp.sum(eigenvalues)

    # Normalize the eigenvalues
    normalised_eigenvalues = eigenvalues / trace

    # Calculate the fractional anisotropy
    numerator =  jnp.var(normalised_eigenvalues)
    denomentator = jnp.mean(normalised_eigenvalues**2)    
    return jnp.sqrt(numerator/denomentator)
    

def var_lambda(X,labels=None):
    """
    Calculate the variance of normalized eigenvalues of the covariance matrix.

    Parameters
    ----------
    X : numpy.ndarray
        The data points.
    labels : numpy.ndarray, optional
        The cluster labels for each data point. If given, calculate the variance of normalized
        eigenvalues of the clustering instead of the dataset.

    Returns
    -------
    float
        The variance of normalized eigenvalues of the dataset or clustering.
    """
    if labels is not None:
        return _global_mean(X, labels, var_lambda)
    
    covariance_matrix = jnp.cov(X, rowvar=False)

    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance_matrix)

    # Calculate the trace of the tensor
    trace = jnp.sum(eigenvalues)

    # Normalize the eigenvalues
    normalised_eigenvalues = eigenvalues / trace

    return jnp.var(normalised_eigenvalues)
    