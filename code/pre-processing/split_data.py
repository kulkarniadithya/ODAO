import pickle

if __name__ == "__main__":

    read_path_14 = "../../data/processed_data/SemEval14-Restaurant/pseudo_labeled_data_train.pickle"
    with open(read_path_14, 'rb') as handle:
        read_pseudo_labeled_data_14 = pickle.load(handle)

    read_path_16 = "../../data/processed_data/SemEval16-Restaurant/pseudo_labeled_data_train.pickle"
    with open(read_path_16, 'rb') as handle:
        read_pseudo_labeled_data_16 = pickle.load(handle)

    pseudo_train_dataset = {}
    pseudo_test_dataset = {}

    pseudo_train_counter = 0
    pseudo_test_counter = 0

    for j in range(0, len(read_pseudo_labeled_data_14)):
        labels = read_pseudo_labeled_data_14[j]['labels']
        if ('AT' in labels) and ('OP' in labels):
            pseudo_train_dataset[pseudo_train_counter] = read_pseudo_labeled_data_14[j]
            pseudo_train_counter = pseudo_train_counter + 1
        else:
            pseudo_test_dataset[pseudo_test_counter] = read_pseudo_labeled_data_14[j]
            pseudo_test_counter = pseudo_test_counter + 1

    for i in range(0, len(read_pseudo_labeled_data_16)):
        labels = read_pseudo_labeled_data_16[i]['labels']
        if ('AT' in labels) and ('OP' in labels):
            pseudo_train_dataset[pseudo_train_counter] = read_pseudo_labeled_data_16[i]
            pseudo_train_counter = pseudo_train_counter + 1
        else:
            pseudo_test_dataset[pseudo_test_counter] = read_pseudo_labeled_data_16[i]
            pseudo_test_counter = pseudo_test_counter + 1

    pseudo_train_write_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_train_dataset.pickle"
    pseudo_test_write_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_test_dataset.pickle"

    with open(pseudo_train_write_path, 'wb') as handle:
        pickle.dump(pseudo_train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pseudo_train_write_path, 'rb') as handle:
        read_pseudo_train_dataset = pickle.load(handle)
    assert pseudo_train_dataset == read_pseudo_train_dataset

    with open(pseudo_test_write_path, 'wb') as handle:
        pickle.dump(pseudo_test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pseudo_test_write_path, 'rb') as handle:
        read_pseudo_test_dataset = pickle.load(handle)
    assert pseudo_test_dataset == read_pseudo_test_dataset
