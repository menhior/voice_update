from django.shortcuts import render, redirect
from .forms import AudioFileForm


from .models import AudioFile, TextFile


import librosa
from scipy.io import wavfile
import numpy as np
import os
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from huggingface_hub import login


def convert_to_mono_and_resample(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=False)
    y_mono = librosa.to_mono(y)
    '''wavfile.write(output_file, sample_rate, segment.astype(np.float32))
    sf.write(audio_path, y, sr)'''
    #wavfile.write(output_file, sample_rate, segment.astype(np.float32))
    wavfile.write(audio_path, sr, y_mono)
    return y_mono, sr

'''def downsample_audio(y, sr, target_sample_rate=16000):
    y_downsampled = librosa.resample(y, orig_sr = sr, target_sr =target_sample_rate)
    return y_downsampled'''


def split_wav_into_segments(input_path, segment_duration=10):
    # Create a subfolder with the same name as the audio file
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(os.path.dirname(input_path), file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio file
    audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)

    # Check if the file is stereo (2 channels)
    if len(audio_data.shape) > 1 and audio_data.shape[0] == 2:
        # Convert stereo to mono by taking the average of the two channels
        audio_data = librosa.to_mono(audio_data)

    # Calculate the number of samples corresponding to the desired segment duration
    segment_samples = int(segment_duration * sample_rate)

    # Calculate the total number of segments
    total_segments = len(audio_data) // segment_samples

    # Split the audio data into segments
    segments = np.array_split(audio_data, total_segments)

    output_files_list = []

    # Save each segment as a separate WAV file in the subfolder
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_dir, f"segment_{i+1}.wav")
        wavfile.write(output_file, sample_rate, segment.astype(np.float32))
        output_files_list.append(output_file)

    return output_dir, output_files_list

def load_wav_files_as_numpy_arrays(folder_path):
    # List all files in the selected folder
    files = os.listdir(folder_path)

    def get_integer_value(element):
        split_data = element.split('\\')[-1].split('_')[1]
        try:
            split_data = split_data.split('.')[0]
        except:
            pass
        return int(split_data)  # Assuming the integer value is at the beginning of each element

    # Sort the list based on the extracted integer value
    files = sorted(files, key=get_integer_value)

    audio_arrays_list = []

    # Filter and process only the WAV files
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

        # Append the NumPy array to the list
        audio_arrays_list.append(audio_data)

    return audio_arrays_list


def upload_file(request):
    if request.method == 'POST':
        form = AudioFileForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                audio_file = form.save()
                # Get the path of the uploaded audio file
                audio_path = audio_file.file.path
                if audio_path.lower().endswith(".wav"):
                    # Process the audio file (convert to mono and downsample)
                    y, sr = convert_to_mono_and_resample(audio_path)
                    
                    # Example usage
                    segment_duration = 10  # seconds
                    segments_dir, output_files_list = split_wav_into_segments(audio_path, segment_duration)
                    #files = os.listdir(segments_dir)

                    numpy_arrays_list = load_wav_files_as_numpy_arrays(segments_dir)

                    access_token = "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"
                    login(access_token)

                    processor = AutoProcessor.from_pretrained("menhior/wav2vec2-large-xls-r-300m-azeri-colab")
                    model = AutoModelForCTC.from_pretrained("menhior/wav2vec2-large-xls-r-300m-azeri-colab")

                    list_of_predictions = []

                    print('Making predictions.')

                    for numpy_array in numpy_arrays_list:
                        input_dict = processor(numpy_array, return_tensors="pt", padding=True, sampling_rate = 16000 )

                        logits = model(input_dict.input_values).logits

                        pred_ids = torch.argmax(logits, dim=-1)[0]

                        decoded_preds =  processor.decode(pred_ids)
                        list_of_predictions.append(decoded_preds)
                        #time.sleep(3)
                        
                    text = ' '.join(list_of_predictions)
                    

                    BASE_DIR = Path(__file__).resolve().parent.parent
                    txt_files_dir = os.path.join(BASE_DIR, 'media/txt_files')

                    if not os.path.exists(txt_files_dir):
                        os.makedirs(txt_files_dir)

                    print('Saving predictiong into a txt file.')

                    file_name = str(audio_path).split('\\')[-1].split('.')[0] + "_preds.txt"
                    full_file_path = os.path.join(txt_files_dir, file_name)
                    with open(full_file_path, 'w', encoding="utf-8") as file:
                        file.write(text)

                    TextFile.objects.create(
                        name=str(audio_path).split('\\')[-1],
                        text=text,
                        )

                    return redirect('upload_success')
            except:
                audio_file = form.save()
                # Get the path of the uploaded audio file
                audio_path = audio_file.file.path
                if audio_path.lower().endswith(".wav"):
                    # Process the audio file (convert to mono and downsample)
                    y, sr = convert_to_mono_and_resample(audio_path)
                    
                    # Example usage
                    segment_duration = 10  # seconds
                    segments_dir, output_files_list = split_wav_into_segments(audio_path, segment_duration)
                    #files = os.listdir(segments_dir)

                    numpy_arrays_list = load_wav_files_as_numpy_arrays(segments_dir)

                    processor = AutoProcessor.from_pretrained("menhior/wav2vec2-large-xls-r-300m-azeri-colab")
                    model = AutoModelForCTC.from_pretrained("menhior/wav2vec2-large-xls-r-300m-azeri-colab")

                    list_of_predictions = []

                    print('Making predictions.')

                    for numpy_array in numpy_arrays_list:
                        input_dict = processor(numpy_array, return_tensors="pt", padding=True, sampling_rate = 16000 )

                        logits = model(input_dict.input_values).logits

                        pred_ids = torch.argmax(logits, dim=-1)[0]

                        decoded_preds =  processor.decode(pred_ids)
                        list_of_predictions.append(decoded_preds)
                        #time.sleep(3)
                        
                    text = ' '.join(list_of_predictions)
                    

                    BASE_DIR = Path(__file__).resolve().parent.parent
                    txt_files_dir = os.path.join(BASE_DIR, 'media/txt_files')

                    if not os.path.exists(txt_files_dir):
                        os.makedirs(txt_files_dir)

                    print('Saving predictiong into a txt file.')
                    file_name = str(audio_path).split('\\')[-1].split('.')[0] 
                    '''file_name = str(audio_path).split('\\')[-1].split('.')[0]
                    if '_audio' in file_name:
                        file_name = file_name[:-len('_audio')] + "_preds.txt":
                    else:
                        file_name = file_name + "_preds.txt"'''

                    full_file_path = os.path.join(txt_files_dir, file_name)
                    with open(full_file_path, 'w', encoding="utf-8") as file:
                        file.write(text)

                    TextFile.objects.create(
                        name=str(audio_path).split('\\')[-1],
                        text=text,
                        )

                    return redirect('upload_success')
            else:
                return redirect('upload_failed')
    else:
        form = AudioFileForm()
    return render(request, 'upload.html', {'form': form})

def upload_success(request):
    return render(request, 'upload_success.html')