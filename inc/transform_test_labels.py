def transform_test_labels_to_indexes(class_names, test_labels_string):
    """Change string labels to integers that are indexes of the class labels"""

    test_labels_final = []
    for label in test_labels_string:
        index = class_names.index(label)
        test_labels_final.append(index)

    return test_labels_final
