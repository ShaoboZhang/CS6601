import numpy as np
from collections import Counter


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left=None, right=None, decision_function=None, class_label=None):
        """
        Create a decision function to select between left and right nodes.
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Return: Class label if a leaf node, otherwise a child node.
        """
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if class_index == -1:
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif class_index == 0:
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns: The root node of the decision tree.
    """
    root_left = DecisionNode(decision_function=lambda a: a[2] != a[3])
    root_left.left = DecisionNode(class_label=0)
    root_left.right = DecisionNode(class_label=1)

    root = DecisionNode(decision_function=lambda a: a[0] == 0)
    root.left = root_left
    root.right = DecisionNode(class_label=1)

    return root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    """
    output = np.array(classifier_output)
    labels = np.array(true_labels)
    p = output[labels == 1]
    n = output[labels == 0]

    tp = len(p[p == 1])
    fn = len(p[p == 0])
    fp = len(n[n == 1])
    tn = len(n[n == 0])
    return np.array([[tp, fn], [fp, tn]])


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as: true_positive/ (true_positive + false_positive)
    """
    output = np.array(classifier_output)
    labels = np.array(true_labels)
    p = output[labels == 1]
    tp = len(p[p == 1])
    return tp / len(output[output == 1])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as: true_positive/ (true_positive + false_negative)
    """
    output = np.array(classifier_output)
    labels = np.array(true_labels)
    p = output[labels == 1]
    tp = len(p[p == 1])
    return tp / len(p)


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as: correct_classifications / total_number_examples
    """
    output = np.array(classifier_output)
    labels = np.array(true_labels)
    p = output[labels == 1]
    n = output[labels == 0]

    tp = len(p[p == 1])
    tn = len(n[n == 0])
    return (tp + tn) / len(labels)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    Returns: Floating point number representing the gini impurity.
    """
    if not len(class_vector):
        return 1
    labels = np.array(class_vector)
    p = len(labels[labels == 1]) / len(class_vector)
    return 1 - p ** 2 - (1 - p) ** 2


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has 0 and 1 values).
    """
    prev = gini_impurity(previous_classes)
    total = len(previous_classes)
    curr = [gini_impurity(cls) * len(cls) / total for cls in current_classes]
    return prev - sum(curr)


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        self.root: DecisionNode = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        self.root = None
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns: Root node of decision tree.
        """
        aa = Counter(classes)
        # if there's only one class
        if len(aa) == 1:
            return DecisionNode(class_label=classes[0])

        # if there's no more feature or reach the limit of depth
        if not features.shape[0] or depth > self.depth_limit:
            return DecisionNode(class_label=max(aa, key=aa.get))

        # select best alpha
        best_alpha = thresh = 0
        max_gain = float('-inf')
        for alpha in range(features.shape[1]):
            feat = features[:, alpha]
            if min(feat) == max(feat):
                continue
            th = np.mean(feat)
            gain = gini_gain(classes, [classes[feat <= th], classes[feat > th]])
            if gain > max_gain:
                max_gain = gain
                best_alpha = alpha
                thresh = th

        if max_gain == float('-inf'):
            return DecisionNode(class_label=max(aa, key=aa.get))

        # create decision node
        feat = features[:, best_alpha]
        feat_l = features[feat<=thresh]
        feat_r = features[feat>thresh]
        class_l = classes[feat <= thresh]
        class_r = classes[feat > thresh]
        root = DecisionNode(decision_function=lambda a: a[best_alpha] <= thresh)
        root.left = self.__build_tree__(feat_l, class_l, depth + 1)
        root.right = self.__build_tree__(feat_r, class_r, depth + 1)

        return root

    def classify(self, features):
        """
        Use the fitted tree to classify a list of example features.
        """
        class_labels = np.array([self.root.decide(feat) for feat in features], dtype=int)

        return class_labels


def generate_k_folds(dataset, k):
    """Randomly split data into k equal subsets.
    Each fold is a tuple (training_set, test_set).
    Each set is a tuple (features, classes).
    """
    features, labels = dataset
    indices = list(range(labels.shape[0]))
    np.random.shuffle(indices)

    num_per_fold = labels.shape[0] // k
    res = []
    for i in range(k):
        train_idx = indices[(i + 1) * num_per_fold:] + indices[:i * num_per_fold]
        train_set = (features[train_idx], labels[train_idx])
        test_idx = indices[i * num_per_fold:(i + 1) * num_per_fold]
        test_set = (features[test_idx], labels[test_idx])
        res.append((train_set, test_set))

    return res


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=5, depth_limit=5, example_subsample_rate=0.5, attr_subsample_rate=0.5):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

        self.attr_idxs = []

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        tree = DecisionTree()
        tree.fit(features, classes)
        self.trees.append(tree)

    def classify(self, features):
        """
        Classify a list of features based on the trained random forest.
        """
        return self.trees[0].classify(features)


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        self.tree = DecisionTree()

    def fit(self, features, classes):
        self.tree.fit(features, classes)

    def classify(self, features):
        return self.tree.classify(features)



class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to itself.
        """
        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] + data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to itself.
        """
        non_vectorized = data * data + data
        return non_vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row with the max sum.
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]
            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row
        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row with the max sum.
        """
        temp_data = data.sum(axis=1)[:100]
        max_index = temp_data.argmax()
        return temp_data[max_index], max_index

    def non_vectorized_flatten(self, data):
        """
        Display occurrences of positive numbers using loops.
        """
        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """
        Display occurrences of positive numbers using vectorization.
        Returns: List of occurrences [(integer, number of occurrences), ...]
        """
        data = data[data > 0].astype(np.int16)
        count = Counter(data)
        return [(val, times) for val, times in count.items()]


def return_your_name():
    # return your name
    return 'Xingbo Song'


if __name__ == '__main__':
    datasets = load_csv('challenge_train.csv', class_index=0)
    # datasets = load_csv('part23_data.csv', class_index=-1)
    k = 8
    k_folds = generate_k_folds(datasets, k)
    for fold in k_folds:
        training_set, test_set = fold
        features, classes = training_set
        forest = ChallengeClassifier()
        forest.fit(features, classes)
        y_pred1 = forest.classify(test_set[0])
        acc1 = accuracy(y_pred1, test_set[1])
        print(acc1, end=' ')
        # tree = DecisionTree(10)
        # tree.fit(features, classes)
        # y_pred2 = tree.classify(test_set[0])
        # acc2 = accuracy(y_pred2, test_set[1])
        # print(acc2, end='')
        print()
