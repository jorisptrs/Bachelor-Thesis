import random
random.seed(0)

def filter_data(unfiltered_features, unfiltered_labels, limit=None, selected_labels=None):
    group = group_by_labels(unfiltered_features, unfiltered_labels)
    group = downsample(group, limit)

    classes = list(group.keys())
    filtered_labels = []
    filtered_features = []
    for label, features in group.items():
        for feature in features:
            filtered_features.append(feature)
            filtered_labels.append(label)

    print(f"Filtered to {len(filtered_features)} samples of shape {filtered_features[0].shape}")
    return classes, filtered_features, filtered_labels

def group_by_labels(X, y, selected_labels=None):
    """
    returns { phoneme : samples }, where the samples = [ [x[0],x[1]...], [] ]
    """
    group = {}
    for i in range(len(y)):
        if selected_labels != None and y[i] not in selected_labels:
            continue
        if y[i] not in group.keys():
            group[y[i]] = []
        group[y[i]].append(X[i])
    return group

def downsample(group, limit):
    for key in group.keys():
        if limit is not None and len(group[key]) > limit:
            group[key] = random.sample(group[key], limit)
    return group