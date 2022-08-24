from transformers import BertTokenizer


def labels_to_span_index(input_ids, output, tokenizer):
    start_index = [0]*len(input_ids)
    end_index = [0]*len(input_ids)
    boolean_array = [False]*len(output)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for i in range(0, len(output)):
        start_boolean = False
        end_boolean = False
        for j in range(0, len(tokens)):
            if tokens[j] == output[i][0]:
                start_index[j] = 1
                start_boolean = True
            if tokens[j] == output[i][-1]:
                end_index[j] = 1
                end_boolean = True
        if start_boolean==True and end_boolean==True:
            boolean_array[i] = True

    for i in range(0, len(output)):
        start_boolean = False
        end_boolean = False
        if not boolean_array[i]:
            for j in range(0, len(tokens)):
                if tokens[j].replace('##', '') in output[i][0]:
                    start_index[j] = 1
                    start_boolean = True
                if tokens[j].replace('##', '') in output[i][-1]:
                    end_index[j] = 1
                    end_boolean = True
            if start_boolean == True and end_boolean == True:
                boolean_array[i] = True

    return start_index, end_index

