import pickle
import xml.etree.ElementTree as Et


def format_data(file_path):
    tree = Et.parse(file_path)
    root = tree.getroot()
    sentence_dictionary = {}
    counter = 0
    for i in range(0, len(root)):
        for j in range(0, len(root[i])):
            for k in range(0, len(root[i][j])):
                sentence_dictionary[counter] = {}
                sentence_dictionary[counter]['review'] = str(root[i][j][k][0].text)
                target_array = []
                category_array = []
                polarity_array = []
                from_array = []
                to_array = []
                try:
                    for l in range(0, len(root[i][j][k][1])):
                        target_array.append(root[i][j][k][1][l].attrib['target'])
                        category_array.append(root[i][j][k][1][l].attrib['category'])
                        polarity_array.append(root[i][j][k][1][l].attrib['polarity'])
                        from_array.append(root[i][j][k][1][l].attrib['from'])
                        to_array.append(root[i][j][k][1][l].attrib['to'])
                except:
                    print(root[i][j][k][0].text)
                    print(counter)

                sentence_dictionary[counter]['target'] = target_array
                sentence_dictionary[counter]['category'] = category_array
                sentence_dictionary[counter]['polarity'] = polarity_array
                sentence_dictionary[counter]['from'] = from_array
                sentence_dictionary[counter]['to'] = to_array
                counter = counter + 1

    return sentence_dictionary


if __name__ == "__main__":
    read_path = "../../data/original_data/SemEval16-Restaurant/test.xml"
    write_path = "../../data/processed_data/SemEval16-Restaurant/sentence_dictionary_test.pickle"
    formatted_sentence_dictionary = format_data(file_path=read_path)
    with open(write_path, 'wb') as handle:
        pickle.dump(formatted_sentence_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(write_path, 'rb') as handle:
        read_data = pickle.load(handle)
    assert formatted_sentence_dictionary == read_data
    print(len(read_data))
    print(read_data[0])
