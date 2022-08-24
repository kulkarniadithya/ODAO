import pickle


def get_aspect_opinion_pairs(dictionary):
    processed_dictionary = {}
    for a in range(0, len(dictionary)):
        review = dictionary[a]['review']
        tokens = dictionary[a]['tokens']
        labels = dictionary[a]['labels']
        aspect_terms = []
        opinion_terms = []
        aspect_opinion_pairs = []
        temp_at = []
        temp_op = []
        pair_at = []
        pair_op = []
        boolean_at = False
        boolean_op = False
        for i in range(0, len(labels)):
            if labels[i] == 'AT':
                temp_at.append(tokens[i])
                if boolean_at == False and boolean_op == False:
                    boolean_at = True
                elif boolean_at == True and len(pair_op) > 0:
                    for x in range(0, len(pair_at)):
                        for y in range(0, len(pair_op)):
                            temp = []
                            temp.append(pair_at[x])
                            temp.append(pair_op[y])
                            if temp not in aspect_opinion_pairs:
                                aspect_opinion_pairs.append(temp)
                    pair_at = []
                    pair_op = []
            elif labels[i] == 'OP':
                temp_op.append(tokens[i])
                if boolean_at == False and boolean_op == False:
                    boolean_op = True
                elif boolean_op == True and len(pair_at) > 0:
                    for x in range(0, len(pair_at)):
                        for y in range(0, len(pair_op)):
                            temp = []
                            temp.append(pair_at[x])
                            temp.append(pair_op[y])
                            if temp not in aspect_opinion_pairs:
                                aspect_opinion_pairs.append(temp)
                    pair_at = []
                    pair_op = []
            else:
                if len(temp_at) > 0:
                    aspect_terms.append(" ".join(temp_at))
                    pair_at.append(" ".join(temp_at))
                    temp_at = []
                elif len(temp_op) > 0:
                    opinion_terms.append(" ".join(temp_op))
                    pair_op.append(" ".join(temp_op))
                    temp_op = []
        if len(temp_at) > 0:
            aspect_terms.append(" ".join(temp_at))
            pair_at.append(" ".join(temp_at))
        if len(temp_op) > 0:
            opinion_terms.append(" ".join(temp_op))
            pair_op.append(" ".join(temp_op))

        if len(pair_at) > 0 and len(pair_op) > 0:
            for x in range(0, len(pair_at)):
                for y in range(0, len(pair_op)):
                    temp = []
                    temp.append(pair_at[x])
                    temp.append(pair_op[y])
                    if temp not in aspect_opinion_pairs:
                        aspect_opinion_pairs.append(temp)
        processed_dictionary[a] = {}
        processed_dictionary[a]['review'] = review
        processed_dictionary[a]['tokens'] = tokens
        processed_dictionary[a]['labels'] = labels
        processed_dictionary[a]['aspect_terms'] = aspect_terms
        processed_dictionary[a]['opinion_terms'] = opinion_terms
        processed_dictionary[a]['aspect_opinion_pairs'] = aspect_opinion_pairs

    return processed_dictionary


if __name__ == "__main__":
    pseudo_train_read_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_train_dataset.pickle"
    with open(pseudo_train_read_path, 'rb') as handle:
        read_pseudo_train_dataset = pickle.load(handle)
    print(read_pseudo_train_dataset[0])
    formatted_dictionary = get_aspect_opinion_pairs(read_pseudo_train_dataset)

    write_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_train_dataset_formatted.pickle"
    with open(write_path, 'wb') as handle:
        pickle.dump(formatted_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(write_path, 'rb') as handle:
        read_formatted_dictionary = pickle.load(handle)
    assert formatted_dictionary == read_formatted_dictionary
    print(formatted_dictionary[0])
