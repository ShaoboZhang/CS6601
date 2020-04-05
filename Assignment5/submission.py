import numpy as np


def return_your_name():
    return 'Xingbo Song'


def get_initial_means(array, k):
    """
    Picks k random points from the 2D array (without replacement) to use as initial cluster means
    """
    idx = np.random.choice(array.shape[0], k, replace=False)
    return array[idx]


def k_means_step(X, k, means):
    """
    A single update/step of the K-means, predict clusters for each of the pixels and calculate new means.
    returns: new_means, clusters
    """
    dists = np.array([np.sum((X - mean) * (X - mean), axis=1) for mean in means])  # k*m
    clusters = np.argmin(dists, axis=0)
    new_means = np.array([np.mean(X[clusters == i, :], axis=0) for i in range(k)])
    return new_means, clusters


def k_means_segment(image_values: np.ndarray, k=3, initial_means=None):
    """
    Separate the provided RGB values into k separate clusters using the k-means algorithm,
    return an updated image with the original values replaced with the corresponding cluster values.
    """
    if initial_means is None:
        initial_means = get_initial_means(image_values.reshape(-1, 3), k)
    r, c, ch = image_values.shape
    new_values = image_values.copy().reshape(-1, 3)
    while True:
        next_means, clusters = k_means_step(new_values, k, initial_means)
        diff = np.sum(next_means - initial_means)
        if not diff:
            break
        initial_means = next_means
    for i, mean in enumerate(initial_means):
        new_values[clusters == i] = mean
    return new_values.reshape(r, c, ch)


def initialize_parameters(X: np.ndarray, k):
    """
    Return initial values for training of the GMM
    returns: MU, SIGMA, PI
    """
    idx = np.random.choice(X.shape[0], k, replace=False)
    mu = X[idx]
    sigma = compute_sigma(X, mu)
    pi = np.ones(k) / k
    return mu, sigma, pi


def compute_sigma(X: np.ndarray, MU):
    """
    Calculate covariance matrix, based in given X and MU values
    """
    k = MU.shape[0]
    m, n = X.shape
    a = np.expand_dims(X, axis=0).repeat(k, axis=0) - MU.reshape(k, 1, -1)
    b = a.transpose((0, 2, 1))
    res = np.matmul(b, a) / m
    return res


def prob(x: np.ndarray, mu, sigma):
    """Calculate the probability of a single data point x"""
    n = mu.shape[0]
    inv = np.linalg.inv(sigma)
    den = np.sqrt(np.linalg.det(sigma)) * np.power(2 * np.pi, n / 2)
    dif = (x - mu).reshape(1, -1)
    num = -0.5 * dif @ inv @ dif.T
    num = np.exp(num)[0][0]
    return num / den


def E_step(X: np.ndarray, MU, SIGMA, PI, k):
    """
    E-step - Calculate responsibility for each of the data points, for the given  MU, SIGMA and PI.
    returns: responsibility - k x m
    """
    m, n = X.shape
    den = np.sqrt(np.linalg.det(SIGMA)) * np.power(2 * np.pi, n / 2)
    inv = np.expand_dims(np.linalg.inv(SIGMA), axis=1).repeat(m, axis=1)
    a = np.expand_dims(X, axis=0).repeat(k, axis=0) - MU.reshape(k, 1, -1)
    a = np.expand_dims(a, axis=2)
    b = a.transpose((0, 1, 3, 2))
    num = -0.5 * np.squeeze(np.matmul(np.matmul(a, inv), b))
    res = np.exp(num) * np.expand_dims(PI / den, axis=1)
    res /= np.sum(res, axis=0)
    return res


def M_step(X, r, k):
    """
    M-step - Calculate new MU, SIGMA and PI matrices based on the given responsibilities.
    returns: new_MU - k x n, new_SIGMA - k x n x n, new_PI - k x 1
    """
    rat = np.sum(r, axis=1).reshape(k, -1)
    MU = r.dot(X) / rat
    a = np.expand_dims(X, axis=0).repeat(k, axis=0) - MU.reshape(k, 1, -1)
    b = a.transpose((0, 2, 1))
    c = np.expand_dims(r, axis=2)
    SIGMA = np.matmul(b, c * a) / rat.reshape(k, 1, 1)
    PI = np.mean(r, axis=1)
    return MU, SIGMA, PI


def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the trained model based on the following formula for posterior probability:
    log10(Pr(X | weights, mean, stdev)) = sum((n=1 to N), log10(sum((k=1 to K), weights[k] * N(x[n] | mu[k],sig[k]))))
    returns: log_likelihood = float
    """
    m, n = X.shape
    den = np.sqrt(np.linalg.det(SIGMA)) * np.power(2 * np.pi, n / 2)
    inv = np.expand_dims(np.linalg.inv(SIGMA), axis=1).repeat(m, axis=1)
    a = np.expand_dims(X, axis=0).repeat(k, axis=0) - MU.reshape(k, 1, -1)
    a = np.expand_dims(a, axis=2)
    b = a.transpose((0, 1, 3, 2))
    num = -0.5 * np.squeeze(np.matmul(np.matmul(a, inv), b))
    res = np.exp(num) * np.expand_dims(PI / den, axis=1)
    res = np.sum(np.log10(np.sum(res, axis=0)))
    return res


def train_model(X, k, convergence_function, initial_values=None):
    """
    Train the mixture model using the expectation-maximization algorithm until convergence.
    Convergence is reached when convergence_function returns terminate as True,
    params: convergence_function = func, initial_values = None or (MU, SIGMA, PI)
    returns: new_MU, new_SIGMA, new_PI, responsibility
    """
    if initial_values is None:
        initial_values = initialize_parameters(X, k)
    pre_prob = -1.0
    conv_ctr = 0
    while True:
        mu, sigma, pi = initial_values
        cur_prob = likelihood(X, pi, mu, sigma, k)
        res = E_step(X, mu, sigma, pi, k)
        conv_ctr, flag = convergence_function(pre_prob, cur_prob, conv_ctr)
        if flag:
            return mu, sigma, pi, res
        initial_values = M_step(X, res, k)
        pre_prob = cur_prob


def cluster(r: np.ndarray):
    """
    Assign each datapoint to a cluster based.
    params: r - k x m
    return: clusters - m x 1
    """
    return np.argmax(r, axis=0)


def segment(X, MU, k, r):
    """
    Returns a matrix where each data point is replaced with its max-likelihood component mean.
    params: X - m x n, MU - k x n, r - k x m
    returns: new_X - m x n
    """
    cls = cluster(r)
    new_x = X.copy()
    for i in range(k):
        new_x[cls == i, :] = MU[i]
    return new_x


def best_segment(X, k, iters):
    """Determine the best segmentation of the image by repeatedly training the model and calculating its likelihood.
    Return the segment with the highest likelihood.
    params: X - m x n, iters = int
    returns: likelihood, segment
    """
    best_lk = float('-inf')
    MU = np.zeros((k, X.shape[1]))
    R = np.zeros((k, X.shape[0]))
    for _ in range(iters):
        mu, sigma, pi, r = train_model(X, k, default_convergence)
        lk = likelihood(X, pi, mu, sigma, k)
        if lk > best_lk:
            best_lk = lk
            MU, R = mu, r
    return best_lk, segment(X, MU, k, R)


def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 < abs(new_likelihood) < abs(prev_likelihood) * 1.1)
    if increase_convergence_ctr:
        conv_ctr += 1
    else:
        conv_ctr = 0
    return conv_ctr, conv_ctr > conv_ctr_cap


def improved_initialization(X, k):
    """
    give better means to start with, based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs) to a uniform values
    returns: MU, SIGMA, PI
    """
    new_values = X.copy()
    best_like = float('-inf')
    MU, SIGMA, PI = None, None, None
    for _ in range(10):
        initial_means = get_initial_means(new_values, k)
        pi = np.full(k, 1 / k)
        while True:
            mu, clusters = k_means_step(new_values, k, initial_means)
            diff = np.sum(mu - initial_means)
            if not diff:
                sigma = compute_sigma(X, mu)
                break
            initial_means = mu
        mu, sigma, pi, res = train_model(X, k, default_convergence, (mu, sigma, pi))
        lk = likelihood(X, pi, mu, sigma, k)
        if lk > best_like:
            best_like = lk
            MU = mu
            SIGMA = sigma
            PI = pi
    return MU, SIGMA, PI


def new_convergence_function(previous_variables, new_variables, conv_ctr, conv_ctr_cap=20):
    """
    when all variables vary by less than 10% from the previous iteration's variables, increase the convergence counter.
    """
    for pre, new in zip(previous_variables, new_variables):
        dif1 = np.abs(new) - 0.9 * np.abs(pre)
        dif2 = 1.1 * np.abs(pre) - np.abs(new)
        if not (dif1 > 0).all() or not (dif2 > 0).all():
            return 0, False
    return conv_ctr + 1, conv_ctr + 1 > conv_ctr_cap


def train_model_improved(X, k, convergence_function=new_convergence_function, initial_values=None):
    if initial_values is None:
        initial_values = initialize_parameters(X, k)
    count = conv_ctr = 0
    while True:
        mu, sigma, pi = initial_values
        res = E_step(X, mu, sigma, pi, k)
        initial_values = M_step(X, res, k)
        conv_ctr, flag = convergence_function((mu, sigma, pi), initial_values, conv_ctr)
        if flag:
            return mu, sigma, pi, res
        count += 1


def bayes_info_criterion(X, PI, MU, SIGMA, k):
    m, n = X.shape
    lk = likelihood(X, PI, MU, SIGMA, k)
    res = np.log10(m) * (k * n * n + k * n + k) - 2 * lk
    return int(res)


def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components corresponding to the minimum BIC
    and maximum likelihood with respect to image_matrix and comp_means.
    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)
    returns: n_comp_min_bic, n_comp_max_likelihood
    """
    m, n = image_matrix.shape
    best_bic = float('inf')
    best_lk = float('-inf')
    n_comp_min_bic = n_comp_max_likelihood = 0
    for i in range(len(comp_means)):
        k = comp_means[i].shape[0]
        mu, sigma, pi, res = train_model(image_matrix, k, default_convergence)
        lk = likelihood(image_matrix, pi, mu, sigma, k)
        bic = np.log10(m) * (n * n + n + 1) * k - 2 * lk
        if lk > best_lk:
            best_lk = lk
            n_comp_max_likelihood = k
        if bic < best_bic:
            best_bic = bic
            n_comp_min_bic = k
    return n_comp_min_bic, n_comp_max_likelihood


def bonus(points_array: np.ndarray, means_array: np.ndarray):
    """
    Return the distance from every point in points_array to every point in means_array.
    returns: dists = numpy array of float
    """
    x, n = points_array.shape
    y = means_array.shape[0]
    a = np.expand_dims(points_array, axis=1).repeat(y, axis=1)
    b = np.expand_dims(means_array, axis=0).repeat(x, axis=0)
    dif = a - b
    c = np.expand_dims(dif, axis=2)
    d = np.expand_dims(dif, axis=3)
    res = np.sqrt(np.matmul(c,d).squeeze())
    # res = np.sqrt(np.sum((a-b)*(a-b), axis=2))
    return res
