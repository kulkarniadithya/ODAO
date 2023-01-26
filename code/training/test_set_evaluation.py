import pickle
from transformers import BertTokenizer
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from construct_dataset import ConstructDatasetSelfTrainingLayer1, ConstructDatasetSelfTrainingLayer2
from statistics import mean
from nltk.corpus import stopwords
import random


def filter_span_tokens(span_tokens, span_confidence, threshold):
    stop_words = set(stopwords.words('english'))
    span_strings = []
    updated_span_confidence = []
    for i in range(0, len(span_tokens)):
        temp = []
        for j in range(0, len(span_tokens[i])):
            if span_tokens[i][j][0].lower() not in stop_words:
                temp.append(span_tokens[i][j][0])
        if len(temp) > 0:
            if " ".join(temp) not in span_strings:
                span_strings.append(" ".join(temp))
            updated_span_confidence.append(span_confidence[i])

    if len(updated_span_confidence) > 0:
        max_confidence = max(updated_span_confidence)
        updated_span_strings = []
        for i in range(0, len(span_strings)):
            if updated_span_confidence[i] >= max_confidence*threshold:
                updated_span_strings.append(span_strings[i])
        return updated_span_strings
    else:
        return []


def complete_spans(spans, tokens):
    updated_tokens = []
    for i in range(0, len(tokens)):
        updated_tokens.append(tokens[i][0])
    updated_spans = []
    for i in range(0, len(spans)):
        split_spans = spans[i].split(" ")
        for k in range(0, len(split_spans)):
            index = updated_tokens.index(split_spans[k])
            temp = []
            if '##' not in split_spans[k]:
                temp.append(split_spans[k])
                for j in range(index+1, len(updated_tokens)):
                    if '##' in updated_tokens[j]:
                        temp.append(updated_tokens[j].replace("##", ""))
                    else:
                        break
            else:
                start_index = 0
                for j in range(index, -1, -1):
                    if '##' in updated_tokens[j]:
                        continue
                    else:
                        start_index = j
                        break
                for j in range(start_index, len(updated_tokens)):
                    if '##' in updated_tokens[j]:
                        temp.append(updated_tokens[j].replace("##", ""))
                    else:
                        break
            if len(temp) > 0:
                updated_spans.append("".join(temp))
    return updated_spans


def get_label_spans(start_logits, end_logits, tokens):
    start_logits = start_logits.tolist()[0]
    end_logits = end_logits.tolist()[0]
    start_max_value = max(start_logits)
    end_max_value = max(end_logits)
    start_indices = []
    end_indices = []
    for i in range(0, len(start_logits)):
        if start_logits[i] >= start_max_value/2 and start_logits[i] > 0:
            start_indices.append(i)
        if end_logits[i] >= end_max_value/2 and end_logits[i] > 0:
            end_indices.append(i)
    if len(start_indices) > 0:
        update_end_indices = []
        for i in range(0, len(end_indices)):
            if end_indices[i] < start_indices[0]:
                continue
            else:
                update_end_indices.append(end_indices[i])

        spans = []
        span_tokens = []
        span_confidence = []
        for i in range(0, len(start_indices)):
            temp = []
            temp_confidence = []
            temp.append(start_indices[i])
            temp_confidence.append(start_logits[start_indices[i]])
            for j in range(0, len(update_end_indices)):
                if update_end_indices[j] >= start_indices[i]:
                    temp.append(update_end_indices[j])
                    temp_confidence.append(end_logits[update_end_indices[j]])
                    break
            if len(temp) == 2:
                spans.append(temp)
                if temp[1] < len(tokens)-1:
                    span_tokens.append(tokens[temp[0]:temp[1]+1])
                else:
                    span_tokens.append(tokens[temp[0]])
                span_confidence.append(mean(temp_confidence))

        inter_spans = filter_span_tokens(span_tokens, span_confidence, 0.2)
        output_spans = complete_spans(inter_spans, tokens)
        return output_spans
    else:
        return []


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    test_read_path = "../../data/processed_data/SemEval16-Restaurant/sentence_dictionary_test.pickle"
    with open(test_read_path, 'rb') as handle:
        read_test_dataset = pickle.load(handle)
    print(read_test_dataset[0])

    bert_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    review = []
    for i in range(0, len(read_test_dataset)):
        review.append(read_test_dataset[i]['review'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = torch.load('../../saved_models/at_epoch_' + str(8))
    model2 = torch.load('../../saved_models/op_epoch_' + str(8))
    at_dataset = ConstructDatasetSelfTrainingLayer1(review=review, tokenizer=tokenizer)
    op_dataset = ConstructDatasetSelfTrainingLayer1(review=review, tokenizer=tokenizer)
    at_train_data = DataLoader(at_dataset, batch_size=1)
    op_train_data = DataLoader(op_dataset, batch_size=1)

    aspect_term_prediction = {}
    opinion_term_prediction = {}

    for i, batch in enumerate(tqdm(at_train_data)):
        at_input_ids = batch['input_ids'].to(device)
        at_segment_ids = batch['segment_ids'].to(device)
        tokens = batch['tokens']
        at_bert_output, at_start_logits, at_end_logits = model1(input_ids=at_input_ids, token_type_ids=at_segment_ids, attention_mask=None,
                                                                module_start_positions=None, module_end_positions=None)
        at_predictions = get_label_spans(at_start_logits, at_end_logits, tokens)
        aspect_term_prediction[i] = {}
        aspect_term_prediction[i]['review'] = review[i]
        aspect_term_prediction[i]['tokens'] = tokens
        aspect_term_prediction[i]['predictions'] = at_predictions

    for i, batch in enumerate(tqdm(op_train_data)):
        op_input_ids = batch['input_ids'].to(device)
        op_segment_ids = batch['segment_ids'].to(device)
        tokens = batch['tokens']
        op_bert_output, op_start_logits, op_end_logits = model2(input_ids=op_input_ids, token_type_ids=op_segment_ids,
                                                                attention_mask=None,
                                                                module_start_positions=None, module_end_positions=None)

        op_predictions = get_label_spans(op_start_logits, op_end_logits, tokens)
        opinion_term_prediction[i] = {}
        opinion_term_prediction[i]['review'] = review[i]
        opinion_term_prediction[i]['tokens'] = tokens
        opinion_term_prediction[i]['predictions'] = op_predictions

    layer1 = {'aspect_term': aspect_term_prediction, 'opinion_term': opinion_term_prediction}

    write_path = "../../data/processed_data/SemEval16-Restaurant/test_self_learning_layer1_iteration1_0_2.pickle"
    with open(write_path, 'wb') as handle:
        pickle.dump(layer1, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # write_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_test_self_learning_layer1_iteration1.pickle"
    # with open(write_path, 'rb') as handle:
    #     layer1 = pickle.load(handle)
    #
    # aspect_term_prediction = layer1["aspect_term"]
    # opinion_term_prediction = layer1["opinion_term"]

    asoe_input = []
    asoe_review = []
    osae_input = []
    osae_review = []

    for i in range(0, len(review)):
        if len(opinion_term_prediction[i]['predictions']) > 0:
            op_predictions = opinion_term_prediction[i]['predictions']
            for j in range(0, len(op_predictions)):
                osae_review.append(review[i])
                osae_input.append(op_predictions[j])
        if len(aspect_term_prediction[i]['predictions']) > 0:
            at_predictions = aspect_term_prediction[i]['predictions']
            for j in range(0, len(at_predictions)):
                asoe_review.append(review[i])
                asoe_input.append(at_predictions[j])

    model3 = torch.load('../../saved_models/asoe_epoch_' + str(8))
    model4 = torch.load('../../saved_models/osae_epoch_' + str(8))

    asoe_dataset = ConstructDatasetSelfTrainingLayer2(review=asoe_review, additional_input=asoe_input,
                                                      tokenizer=tokenizer)
    osae_dataset = ConstructDatasetSelfTrainingLayer2(review=osae_review, additional_input=osae_input,
                                                      tokenizer=tokenizer)
    asoe_train_data = DataLoader(asoe_dataset, batch_size=1)
    osae_train_data = DataLoader(osae_dataset, batch_size=1)

    asoe_model_prediction = {}
    osae_model_prediction = {}

    for i, batch in enumerate(tqdm(asoe_train_data)):
        asoe_input_ids = batch['input_ids'].to(device)
        asoe_segment_ids = batch['segment_ids'].to(device)
        tokens = batch['tokens']
        asoe_bert_output, asoe_start_logits, asoe_end_logits = model3(input_ids=asoe_input_ids, token_type_ids=asoe_segment_ids, attention_mask=None,
                                                                      module_start_positions=None, module_end_positions=None)
        asoe_predictions = get_label_spans(asoe_start_logits, asoe_end_logits, tokens)
        asoe_model_prediction[i] = {}
        asoe_model_prediction[i]['review'] = asoe_review[i]
        asoe_model_prediction[i]['tokens'] = tokens
        asoe_model_prediction[i]['input'] = asoe_input[i]
        asoe_model_prediction[i]['predictions'] = asoe_predictions

    for i, batch in enumerate(tqdm(osae_train_data)):
        osae_input_ids = batch['input_ids'].to(device)
        osae_segment_ids = batch['segment_ids'].to(device)
        tokens = batch['tokens']
        osae_bert_output, osae_start_logits, osae_end_logits = model4(input_ids=osae_input_ids, token_type_ids=osae_segment_ids, attention_mask=None,
                                                                      module_start_positions=None, module_end_positions=None)
        osae_predictions = get_label_spans(osae_start_logits, osae_end_logits, tokens)
        osae_model_prediction[i] = {}
        osae_model_prediction[i]['review'] = osae_review[i]
        osae_model_prediction[i]['tokens'] = tokens
        osae_model_prediction[i]['input'] = osae_input[i]
        osae_model_prediction[i]['predictions'] = osae_predictions

    layer2 = {'asoe_model_prediction': asoe_model_prediction, 'osae_model_prediction': osae_model_prediction}

    write_path = "../../data/processed_data/SemEval16-Restaurant/test_self_learning_layer2_iteration1_0_2.pickle"
    with open(write_path, 'wb') as handle:
        pickle.dump(layer2, handle, protocol=pickle.HIGHEST_PROTOCOL)

