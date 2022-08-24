from nltk.parse.corenlp import CoreNLPDependencyParser
import pickle


def get_pseudo_labels(parser, sentence):
    result = parser.raw_parse(sentence)
    dep = next(result)
    address_list = []
    head_list = []
    relation_list = []
    tag_list = []
    word_list = []

    for i in range(1, len(dep.nodes)):
        address_list.append(int(dep.nodes[i]['address'])-1)
        head_list.append(int(dep.nodes[i]['head'])-1)
        relation_list.append(str(dep.nodes[i]['rel']).split(":")[0])
        tag_list.append(str(dep.nodes[i]['tag']))
        word_list.append(str(dep.nodes[i]['word']))
    labels = ['O'] * len(word_list)

    for j in range(0, 3):
        if 'nsubj' in relation_list:
            for i in range(0, len(relation_list)):
                if relation_list[i] == 'nsubj':
                    if tag_list[i] in ['NN', 'NNS', 'NNP', 'NNPS']:
                        if tag_list[int(head_list[i])] in ['JJ', 'JJR', 'JJS']:
                            try:
                                labels[int(head_list[i])] = 'OP'
                                labels[i] = 'AT'
                            except:
                                print(head_list[i])

        for i in range(0, len(relation_list)):
            if relation_list[i] in ['compound', 'conj']:
                if tag_list[i] in ['JJ', 'JJR', 'JJS']:
                    if labels[int(head_list[i])] == 'OP':
                        labels[i] = 'OP'
                elif tag_list[i] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    if labels[int(head_list[i])] == 'AT':
                        labels[i] = 'AT'
    return word_list, labels


if __name__ == "__main__":
    dep_parser = CoreNLPDependencyParser('http://localhost:9015')
    read_path = "../../data/processed_data/SemEval14-Restaurant/sentence_dictionary_train.pickle"
    with open(read_path, 'rb') as handle:
        read_data = pickle.load(handle)
    print(read_data[0])
    pseudo_labeled_data = {}
    for i in range(0, len(read_data)):
        words, pseudo_labels = get_pseudo_labels(dep_parser, read_data[i]['review'])
        pseudo_labeled_data[i] = {}
        pseudo_labeled_data[i]['review'] = read_data[i]['review']
        pseudo_labeled_data[i]['tokens'] = words
        pseudo_labeled_data[i]['labels'] = pseudo_labels

    write_path = "../../data/processed_data/SemEval14-Restaurant/pseudo_labeled_data_train.pickle"
    with open(write_path, 'wb') as handle:
        pickle.dump(pseudo_labeled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(write_path, 'rb') as handle:
        read_pseudo_labeled_data = pickle.load(handle)
    assert pseudo_labeled_data == read_pseudo_labeled_data
    print(len(read_data))
    print(len(pseudo_labeled_data))
