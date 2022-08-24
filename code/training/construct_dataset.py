import torch
from torch.utils.data import Dataset
from convert_labels_to_span_index import labels_to_span_index


class ConstructDatasetLayer1(Dataset):
    def __init__(self, review, output, tokenizer):
        self.review = review
        self.output = output
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = self.review[idx].lower()
        input_ids = self.tokenizer.encode(review)
        sep_idx = input_ids.index(self.tokenizer.sep_token_id)
        num_seg_a = sep_idx + 1
        num_seq_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seq_b

        output = []
        for i in range(0, len(self.output[idx])):
            if self.output[idx][i] not in ['[CLS]']:
                temp = self.output[idx][i].lower().split(" ")
                output.append(temp)

        # tokens_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        start_index, end_index = labels_to_span_index(input_ids=input_ids, output=output, tokenizer=self.tokenizer)

        sample = {"input_ids": torch.tensor(input_ids), "segment_ids": torch.tensor(segment_ids),
                  "start_index": torch.tensor(start_index), "end_index": torch.tensor(end_index)}
        return sample


class ConstructDatasetLayer2(Dataset):
    def __init__(self, review, additional_input, output, tokenizer):
        self.review = review
        self.output = output
        self.tokenizer = tokenizer
        self.additional_input = additional_input

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = self.review[idx].lower()
        additional_input = self.additional_input[idx].lower()
        input_ids = self.tokenizer.encode(additional_input, review)
        sep_idx = input_ids.index(self.tokenizer.sep_token_id)
        num_seg_a = sep_idx + 1
        num_seq_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seq_b

        output = []
        for i in range(0, len(self.output[idx])):
            if self.output[idx][i] not in ['[CLS]']:
                temp = self.output[idx][i].lower().split(" ")
                output.append(temp)

        # tokens_list = self.tokenizer.convert_ids_to_tokens(input_ids)
        start_index, end_index = labels_to_span_index(input_ids=input_ids, output=output, tokenizer=self.tokenizer)

        sample = {"input_ids": torch.tensor(input_ids), "segment_ids": torch.tensor(segment_ids),
                  "start_index": torch.tensor(start_index), "end_index": torch.tensor(end_index)}
        return sample
