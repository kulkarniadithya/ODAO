import pickle
import numpy as np
from transformers import BertTokenizer
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cross_decomposition import CCA
from construct_dataset import ConstructDatasetLayer1, ConstructDatasetLayer2
from build_model import Model

if __name__ == "__main__":
    pseudo_train_read_path = "../../data/processed_data/SemEval16-Restaurant/pseudo_train_dataset_formatted.pickle"
    with open(pseudo_train_read_path, 'rb') as handle:
        read_pseudo_train_dataset = pickle.load(handle)
    print(read_pseudo_train_dataset[0])
    bert_version = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    review = []
    review_layer2 = []
    at_output = []
    ot_output = []
    osae_output = []
    osae_input = []
    add_output = []
    add_input = []

    for i in range(0, len(read_pseudo_train_dataset)):
        review.append(read_pseudo_train_dataset[i]['review'])
        at_output.append(read_pseudo_train_dataset[i]['aspect_terms'])
        ot_output.append(read_pseudo_train_dataset[i]['opinion_terms'])
        pairs = read_pseudo_train_dataset[i]['aspect_opinion_pairs']
        if len(pairs) > 1:
            osae_input.append(pairs[0][1])
            osae_output.append(pairs[0][0])
            for k in range(1, len(pairs)):
                add_input.append(pairs[k][1])
                add_output.append(pairs[k][0])
                review_layer2.append(read_pseudo_train_dataset[i]['review'])
        else:
            osae_input.append(pairs[0][1])
            osae_output.append(pairs[0][0])

    asoe_input = []
    asoe_output = []
    asoe_review = []
    osae_input_formatted = []
    osae_output_formatted = []
    for a in range(0, len(osae_input)):
        asoe_input.append(osae_output[a])
        asoe_output.append([osae_input[a]])
        osae_input_formatted.append(osae_input[a])
        osae_output_formatted.append([osae_output[a]])
        asoe_review.append(review[a])
    for b in range(0, len(add_input)):
        asoe_input.append(add_output[b])
        asoe_output.append([add_input[b]])
        asoe_review.append(review_layer2[b])
        osae_input_formatted.append(add_input[b])
        osae_output_formatted.append([add_output[b]])

    at_dataset = ConstructDatasetLayer1(review=review, output=at_output, tokenizer=tokenizer)
    op_dataset = ConstructDatasetLayer1(review=review, output=ot_output, tokenizer=tokenizer)
    asoe_dataset = ConstructDatasetLayer2(review=asoe_review, additional_input=asoe_input, output=asoe_output, tokenizer=tokenizer)
    osae_dataset = ConstructDatasetLayer2(review=asoe_review, additional_input=osae_input_formatted, output=osae_output_formatted,
                                          tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = Model()
    model1.train().to(device)
    model2 = Model()
    model2.train().to(device)
    model3 = Model()
    model3.train().to(device)
    model4 = Model()
    model4.train().to(device)
    optimizer1 = optim.AdamW(params=model1.parameters(), lr=1e-5)
    optimizer2 = optim.AdamW(params=model2.parameters(), lr=1e-5)
    optimizer3 = optim.AdamW(params=model3.parameters(), lr=1e-5)
    optimizer4 = optim.AdamW(params=model4.parameters(), lr=1e-5)
    n_epochs = 1
    at_train_data = DataLoader(at_dataset, batch_size=1)
    op_train_data = DataLoader(op_dataset, batch_size=1)
    asoe_train_data = DataLoader(asoe_dataset, batch_size=1)
    osae_train_data = DataLoader(osae_dataset, batch_size=1)
    correlation_result = []
    for epochs in range(n_epochs):
        at_train_loss = []
        at_current_loss = 0
        op_train_loss = []
        op_current_loss = 0
        asoe_train_loss = []
        asoe_current_loss = 0
        osae_train_loss = []
        osae_current_loss = 0
        epoch_at_bert_output = []
        epoch_op_bert_output = []
        epoch_asoe_bert_output = []
        epoch_osae_bert_output = []
        for i, batch in enumerate(tqdm(at_train_data)):
            at_input_ids = batch['input_ids'].to(device)
            at_segment_ids = batch['segment_ids'].to(device)
            at_start_index = batch['start_index'].to(device)
            at_end_index = batch['end_index'].to(device)
            at_loss, at_bert_output = model1(input_ids=at_input_ids, token_type_ids=at_segment_ids, attention_mask=None,
                module_start_positions=at_start_index, module_end_positions=at_end_index)
            epoch_at_bert_output.append(at_bert_output[0].squeeze().detach().cpu())
            at_loss.backward()
            at_current_loss += at_loss.item()
            if i % 8 == 0 and i > 0:
                optimizer1.step()
                optimizer1.zero_grad()
                at_train_loss.append(at_current_loss / 8)
                at_current_loss = 0

        for i, batch in enumerate(tqdm(op_train_data)):
            op_input_ids = batch['input_ids'].to(device)
            op_segment_ids = batch['segment_ids'].to(device)
            op_start_index = batch['start_index'].to(device)
            op_end_index = batch['end_index'].to(device)
            op_loss, op_bert_output = model2(input_ids=op_input_ids, token_type_ids=op_segment_ids, attention_mask=None,
                module_start_positions=op_start_index, module_end_positions=op_end_index)
            epoch_op_bert_output.append(op_bert_output[0].squeeze().detach().cpu())
            op_loss.backward()
            op_current_loss += op_loss.item()
            if i % 8 == 0 and i > 0:
                optimizer2.step()
                optimizer2.zero_grad()
                op_train_loss.append(op_current_loss / 8)
                op_current_loss = 0

        for i, batch in enumerate(tqdm(asoe_train_data)):
            asoe_input_ids = batch['input_ids'].to(device)
            asoe_segment_ids = batch['segment_ids'].to(device)
            asoe_start_index = batch['start_index'].to(device)
            asoe_end_index = batch['end_index'].to(device)
            asoe_loss, asoe_bert_output = model3(input_ids=asoe_input_ids, token_type_ids=asoe_segment_ids, attention_mask=None,
                module_start_positions=asoe_start_index, module_end_positions=asoe_end_index)
            asoe_input_ids = np.array(asoe_input_ids.squeeze().detach().cpu().tolist())
            asoe_indexes = np.where(asoe_input_ids == 102)[0]
            epoch_asoe_bert_output.append(asoe_bert_output[0].squeeze()[asoe_indexes[0]:asoe_indexes[1]+1].detach().cpu())
            asoe_loss.backward()
            asoe_current_loss += asoe_loss.item()
            if i % 8 == 0 and i > 0:
                optimizer3.step()
                optimizer3.zero_grad()
                asoe_train_loss.append(asoe_current_loss / 8)
                asoe_current_loss = 0

        for i, batch in enumerate(tqdm(osae_train_data)):
            osae_input_ids = batch['input_ids'].to(device)
            osae_segment_ids = batch['segment_ids'].to(device)
            osae_start_index = batch['start_index'].to(device)
            osae_end_index = batch['end_index'].to(device)
            osae_loss, osae_bert_output = model4(input_ids=osae_input_ids, token_type_ids=osae_segment_ids, attention_mask=None,
                module_start_positions=osae_start_index, module_end_positions=osae_end_index)
            osae_input_ids = np.array(osae_input_ids.squeeze().detach().cpu().tolist())
            osae_indexes = np.where(osae_input_ids == 102)[0]
            epoch_osae_bert_output.append(osae_bert_output[0].squeeze()[osae_indexes[0]:osae_indexes[1]+1].detach().cpu())
            osae_loss.backward()
            osae_current_loss += osae_loss.item()
            if i % 8 == 0 and i > 0:
                optimizer4.step()
                optimizer4.zero_grad()
                osae_train_loss.append(osae_current_loss / 8)
                osae_current_loss = 0

        temp_correlation_result = []
        for z in range(0, len(epoch_at_bert_output)):
            X = epoch_at_bert_output[z]
            Y = epoch_osae_bert_output[z][:len(X)]
            X1 = epoch_op_bert_output[z]
            Y1 = epoch_asoe_bert_output[z][:len(X1)]
            cca = CCA(n_components=4)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            result1 = np.corrcoef(X_c.T, Y_c.T)[0, 1]
            cca1 = CCA(n_components=4)
            cca1.fit(X1, Y1)
            X1_c, Y1_c = cca1.transform(X1, Y1)
            result2 = np.corrcoef(X1_c.T, Y1_c.T)[0, 1]
            correlation = result1 + result2
            temp_correlation_result.append(correlation)
        mean_correlation = np.mean(np.array(temp_correlation_result))
        correlation_result.append(mean_correlation)

        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()
        optimizer3.step()
        optimizer3.zero_grad()
        optimizer4.step()
        optimizer4.zero_grad()
        torch.save(model1, '../../saved_models/at_epoch_' + str(epochs))
        torch.save(model2, '../../saved_models/op_epoch_' + str(epochs))
        torch.save(model3, '../../saved_models/asoe_epoch_' + str(epochs))
        torch.save(model4, '../../saved_models/osae_epoch_' + str(epochs))
        print("Epoch: " + str(epochs) + " Correlation score: " + str(mean_correlation))
    print(correlation_result)
