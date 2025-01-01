from datasets import load_dataset, concatenate_datasets
from custom_logging import logging
from exception import customexception
import sys
from datasets import Audio



class processing:
    def __init__(self, data_path,feature_extractor,tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.data_path = data_path

    def _prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["text"]).input_ids
        return batch
        

    def load_data(self):
        try:
            if isinstance(self.data_path, list):
                train_dataset = []
                valid_dataset = []
                test_dataset = []
                for data in self.data_path:
                    dataset = load_dataset(data)
                    train_data = dataset['train']
                    valid_data = dataset['validation']
                    test_data = dataset['test']
                    train_dataset.append(train_data)
                    valid_dataset.append(valid_data)
                    test_dataset.append(test_data)
                    logging.info(f"{data} has been loaded successfully")
                

                concatenate_train = concatenate_datasets(train_dataset)
                concatenate_valid = concatenate_datasets(valid_dataset)
                concatenate_test = concatenate_datasets(test_dataset)

                logging.info("concatenation completed")
                
                train_dataset_casting = concatenate_train.cast_column("audio", Audio(sampling_rate=16000))
                valid_dataset_casting = concatenate_valid.cast_column("audio", Audio(sampling_rate=16000))
                test_dataset_casting = concatenate_test.cast_column("audio", Audio(sampling_rate=16000))

                logging.info("casting completed")

                prepare_train = train_dataset_casting.map(self._prepare_dataset, remove_columns=train_dataset_casting.column_names, num_proc=4)
                prepare_valid = valid_dataset_casting.map(self._prepare_dataset, remove_columns=valid_dataset_casting.column_names, num_proc=4)
                prepare_test = test_dataset_casting.map(self._prepare_dataset, remove_columns=test_dataset_casting.column_names, num_proc=4)


                logging.info("data loading complete")
                return prepare_train, prepare_valid, prepare_test
            else:
                dataset = load_dataset(self.data_path)
                train_dataset = dataset['train']
                valid_dataset = dataset['validation']
                test_dataset = dataset['test']

                train_dataset_casting = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
                valid_dataset_casting = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
                test_dataset_casting = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

                logging.info("casting completed")

                prepare_train = train_dataset_casting.map(self._prepare_dataset, remove_columns=train_dataset_casting.column_names, num_proc=4)
                prepare_valid = valid_dataset_casting.map(self._prepare_dataset, remove_columns=valid_dataset_casting.column_names, num_proc=4)
                prepare_test = test_dataset_casting.map(self._prepare_dataset, remove_columns=test_dataset_casting.column_names, num_proc=4)


                logging.info("data loading complete")
                return prepare_train, prepare_valid, prepare_test
                


        except Exception as e:
            logging.info("Problem arise in the data processing")
            customexception(e, sys)


    
# if __name__=="__main__":
#     print("Starting...")
#     from transformers import WhisperFeatureExtractor
#     from transformers import WhisperTokenizer
#     from transformers import WhisperProcessor
#     from transformers import WhisperForConditionalGeneration
#     import evaluate
#     from transformers import Seq2SeqTrainingArguments
#     from transformers import Seq2SeqTrainer


#     feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
#     tokenizer = WhisperTokenizer.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
#     processor = WhisperProcessor.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000", language="Arabic", task="transcribe")
#     model = WhisperForConditionalGeneration.from_pretrained("whisper-large-v3-turbo_2/checkpoint-5000")
#     p1 = processor(["crtvai/jordan-dataset-v21","crtvai/jordan-dataset-v21"])
#     train, valid, test = p1.load_data()
#     print(train, valid, test)


# stt_data_21 = load_dataset('crtvai/jordan-dataset-v21')
# stt_data_22 = load_dataset('crtvai/jordan-dataset-v22')
# stt_data_23 = load_dataset('crtvai/jordan-dataset-v23')
# stt_data_24 = load_dataset('crtvai/jordan-dataset-v24')
# stt_data_25 = load_dataset('crtvai/jordan-dataset-v25')
# stt_data_26 = load_dataset('crtvai/jordan-dataset-v26')
# stt_data_27 = load_dataset('crtvai/jordan-dataset-v27')
# stt_data_28 = load_dataset('crtvai/jordan-dataset-v28')
# stt_data_29 = load_dataset('crtvai/jordan-dataset-v29')
# stt_data_30 = load_dataset('crtvai/jordan-dataset-v30')


# atc_dataset_train = concatenate_datasets([stt_data_21['train'], stt_data_22['train'],stt_data_23['train'],
#                                           stt_data_24['train'],stt_data_25['train'],
#                                          stt_data_26['train'], stt_data_27['train'],stt_data_28['train'],
#                                           stt_data_29['train'],stt_data_30['train']])

# atc_dataset_valid = concatenate_datasets([stt_data_21['validation'], stt_data_22['validation'],stt_data_23['validation'],
#                                           stt_data_24['validation'],stt_data_25['validation'],
#                                          stt_data_26['validation'], stt_data_27['validation'],stt_data_28['validation'],
#                                           stt_data_29['validation'],stt_data_30['validation']])

# atc_dataset_test = concatenate_datasets([stt_data_21['test'], stt_data_22['test'],stt_data_23['test'],stt_data_24['test'],
#                                          stt_data_25['test'],
#                                         stt_data_26['test'], stt_data_27['test'],stt_data_28['test'],stt_data_29['test'],
#                                          stt_data_30['test']])

# print(atc_dataset_train, atc_dataset_valid, atc_dataset_test)
# print(atc_dataset_train[0])

# from datasets import Audio

# atc_dataset_train = atc_dataset_train.cast_column("audio", Audio(sampling_rate=16000))
# atc_dataset_valid = atc_dataset_valid.cast_column("audio", Audio(sampling_rate=16000))
# atc_dataset_test = atc_dataset_test.cast_column("audio", Audio(sampling_rate=16000))

