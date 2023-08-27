import random

random.seed(0)


def group_by_labels(X, y, selected_labels=None):
    """
    returns { phoneme : samples }, where the samples = [ [x[0],x[1]...], [] ]
    """
    group = {}
    for i in range(len(y)):
        if selected_labels is not None and y[i] not in selected_labels:
            continue
        if y[i] not in group.keys():
            group[y[i]] = []
        group[y[i]].append(X[i])
    return group


def filter_by_speaker(features, labels, speakers, selected_speakers):
    """
    Filters the provided features and labels so that only instances from selected speakers are included.
    :param features: List of features.
    :param labels: List of corresponding labels.
    :param speakers: List of corresponding speakers.
    :param selected_speakers: Speakers to be selected.
    :return: Filtered features and labels.
    """
    selected_features = []
    selected_labels = []

    for i in range(len(labels)):
        if speakers[i] in selected_speakers:
            selected_features.append(features[i])
            selected_labels.append(labels[i])

    return selected_features, selected_labels


def downsample(features, labels, limit=None, percent_limit=None, selected_labels=None):
    """
    Downsamples the data according to either an absolute or relative size limit.
    :param features: List of features.
    :param labels: List of corresponding labels.
    :param limit: Absolute size limit.
    :param percent_limit: Relative size limit.
    :return: Downsampled features and labels, and the remaining features and labels.
    """
    group = group_by_labels(features, labels, selected_labels=selected_labels)
    remaining_features = []
    remaining_labels = []

    if limit == "auto":
        limit = min([len(value) for value in group.values()])

    downsampled_features = []
    downsampled_labels = []

    for key in group.keys():
        random.shuffle(group[key])

        if percent_limit is not None:
            limit = int(len(group[key]) * percent_limit)

        if limit is not None and len(group[key]) > limit:
            remaining_features += group[key][limit:]
            remaining_labels += [key] * (len(group[key]) - limit)
            group[key] = group[key][:limit]

        downsampled_features += group[key]
        downsampled_labels += [key] * len(group[key])

    return downsampled_features, downsampled_labels, remaining_features, remaining_labels


def filter_and_downsample(features, labels, selected_labels=None, limit=None, percent_limit=None, speakers=None,
                          selected_speakers=None):
    """
    Filters the provided features and labels according to given criteria and then downsamples them.
    :param features: List of features.
    :param labels: List of corresponding labels.
    :param selected_labels: Labels to be selected. If None, all labels will be used.
    :param limit: Absolute size limit.
    :param percent_limit: Relative size limit.
    :param speakers: List of speakers.
    :param selected_speakers: Speakers to be selected.
    :param test: Whether to return the remaining features and labels.
    :return: Classes, filtered features, filtered labels, and optionally remaining features and labels.
    """
    if speakers:
        features, labels = filter_by_speaker(features, labels, speakers, selected_speakers)

    downsampled_features, downsampled_labels, remaining_features, remaining_labels \
        = downsample(features, labels, limit, percent_limit, selected_labels=selected_labels)

    print(f"Filtered to {len(downsampled_features)} train and {len(remaining_features)} test samples.")
    print(f"...of shape: {downsampled_features[0].shape}")

    classes = list(set(downsampled_labels))

    if limit is not None or percent_limit is not None:
        return classes, downsampled_features, downsampled_labels, remaining_features, remaining_labels
    else:
        return classes, downsampled_features, downsampled_labels


def train_test_split(features, labels, proportion=0.8):
    """
    Splits the features and labels into train and test sets.
    :param features: List of features.
    :param labels: List of corresponding labels.
    :param proportion: Proportion of data to be used for the training set.
    :return: Train features, train labels, test features, test labels.
    """
    combined = list(zip(features, labels))
    random.shuffle(combined)

    split_index = int(len(features) * proportion)
    train_data, test_data = combined[:split_index], combined[split_index:]

    train_features, train_labels = zip(*train_data)
    test_features, test_labels = zip(*test_data)

    return list(train_features), list(train_labels), list(test_features), list(test_labels)
