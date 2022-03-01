from django.apps import AppConfig
from django.core.files.storage import FileSystemStorage
import os
import html
import pathlib
from django.apps import AppConfig
import torchaudio
from datasets import load_dataset
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import nltk
#from models import File





directory2 = "/home/kineubuntu/Models/modelFromWav2vec"
directory ="fkHug/modelFromWav2vec"
#clips_directory = '/home/kineubuntu/Models/clips'
#audio_mp = 'AUDIO-2021-07-21-01-08-52.mp3'
#csv_files = '/home/kineubuntu/Models/FileCSV'

class GestioncontactConfig(AppConfig):
    name = 'gestionContact'
    #test_dataset = load_dataset('csv',data_files=[f'{csv_files}/SampleSubmission2.csv'])
    #test_dataset = load_dataset('csv',data_files=[f'{csv_files}/SampleSubmission.csv'])
    #tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
    #model = Wav2Vec2ForCTC.from_pretrained(directory)
   


    # def load_model():
        
    #     #tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
    #     processor = Wav2Vec2Processor.from_pretrained(directory)
    #     model = Wav2Vec2ForCTC.from_pretrained(directory)
    #     #model = AutoModel.from_pretrained(directory)
    #     #model.to("cuda")
    #     return model, processor
    #     #return tokenizer,model

    # def speech_file_to_array_fn(batch):
    #      speech_array, sampling_rate = torchaudio.load(f"clips_directory/{batch['ID']}.mp3")
    #      resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    #      batch["speech"] = resampler(speech_array).squeeze().numpy()
    #      return batch
      
    # test_dataset = test_dataset.map(speech_file_to_array_fn)

    # def evaluate(batch):
    #     processor,model=load_model()
    #     inputs = processor(batch["speech"], sampling_rate=16000, return_tensors="pt", padding=True)
    #     with torch.no_grad():
    #         logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    #     pred_ids = torch.argmax(logits, dim=-1)
    #     batch["transcription"] = processor.batch_decode(pred_ids)
    #     return batch
    # result = test_dataset.map(evaluate, batched=True, batch_size=8)
    # print(result)

    # def asr_transcription():
    #     data = pd.read_csv(f'{csv_files}/Train.csv')
    #     final_pred = [ 'None' if pred=='' else pred for pred in result['data']['transcription']]
    #     #final_pred[200]
    #     return final_pred
    # def correct_sentence(input_text):
    #     sentences = nltk.sent_tokenize(input_text)
    #     return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))


   
    # def load_model():
        
    #     tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
    #     processor = Wav2Vec2Processor.from_pretrained(directory)
    #     model = Wav2Vec2ForCTC.from_pretrained(directory)
    #     #model = AutoModel.from_pretrained(directory)
    #     #model.to("cuda")
    #     #return model, processor
    #     return tokenizer,model


    # def asr_transcript(input_file):

    #     #tokenizer, model = load_model()
    #     tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
    #     processor = Wav2Vec2Processor.from_pretrained(directory)
    #     model = Wav2Vec2ForCTC.from_pretrained(directory)

        
    #     speech, fs = sf.read(input_file)

    #     if len(speech.shape) > 1: 
    #         speech = speech[:,0] + speech[:,1]

    #     if fs != 16000:
    #         speech = librosa.resample(speech, fs, 16000)

    #     input_values = tokenizer(speech, return_tensors="pt").input_values
    #     logits = model(input_values).logits
        
    #     predicted_ids = torch.argmax(logits, dim=-1)
        
    #     transcription = tokenizer.decode(predicted_ids[0])
    #     print(transcription)

    #     return transcription.lower()

    # def load_processor():
    #     processor = Wav2Vec2Processor.from_pretrained(directory)
    #     print("processor loading")
    #     return processor

    # def load_model():
    #     model = Wav2Vec2ForCTC.from_pretrained(directory)
    #     print("model loading")
    #     return model

 
    # def asr_transcript(processor,model,input_file):

    #     #tokenizer, model = load_model()
    #     processor = load_processor()
    #     model = load_model()
    #     #tokenizer = Wav2Vec2Tokenizer.from_pretrained(directory)
    #     #processor = Wav2Vec2Processor.from_pretrained(directory)
    #     #model = Wav2Vec2ForCTC.from_pretrained(directory)

    #     speech_array, sampling_rate = torchaudio.load(input_file)
    #     resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    #     speech = resampler(speech_array).squeeze().numpy()

    #     inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    #     with torch.no_grad():
    #         logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
        
    #         predicted_ids = torch.argmax(logits, dim=-1)
        
    #         transcription = processor.decode(predicted_ids[0])
    #         print(transcription)

    #     return transcription.lower()


    #load_model()
    #asr_transcript('/home/kineubuntu/AUDIO-2021-07-21-01-08-52.wav')   




