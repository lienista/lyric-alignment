import json
import os
import torchaudio
import sys
import torch
from transformers import AutoTokenizer, AutoFeatureExtractor
from model_handling import Wav2Vec2ForCTC

def handle_sample(audio_tensor, lyric_data):
    # Aligns lyrics with audio tensor using the pre-trained model.
    inputs = feature_extractor(audio_tensor.squeeze(0).numpy(), return_tensors="pt", sampling_rate=16000)

    # Perform inference using the model
    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted token ids from logits
    predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    # Convert predicted ids to tokens (words)
    predicted_words = [tokenizer.convert_ids_to_tokens(i) for i in predicted_ids if i < len(tokenizer)]

    # Initialize aligned lyrics output
    aligned_lyrics = []
    audio_length = audio_tensor.shape[1]  # Length of audio in samples
    word_duration = audio_length / len(predicted_words)  # Average duration per word
    
    # Align words with lyric data
    for i, segment in enumerate(lyric_data):
        if i < len(predicted_words):
            start_time = int(i * word_duration)
            end_time = int((i + 1) * word_duration)
            aligned_lyrics.append({
                "s": start_time,
                "e": end_time,
                "d": predicted_words[i]
            })
    
    return aligned_lyrics

def format_input_txt_to_json(input_txt, output_json):
    # Formats an input text file into a JSON format for lyric alignment.
    try:
        formatted_data = []
        
        with open(input_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            words = line.strip().split()  # Split the line into words
            segment = {
                "s": 0,
                "e": 0,
                "l": []
            }
            for word in words:
                formatted_word = {
                    "s": 0,
                    "e": 0,
                    "d": word
                }
                segment["l"].append(formatted_word)
            formatted_data.append(segment)

        # Check if the directory exists, create it if not
        output_dir = os.path.dirname(output_json)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)

        print(f"Formatted JSON saved to: {output_json}")

    except Exception as e:
        print(f"Error processing the file: {str(e)}")
        sys.exit(1)

def check_audio_properties(input_path):
    """Check if the audio file is 16 kHz and single channel."""
    audio, sample_rate = torchaudio.load(input_path)
    num_channels = audio.size(0)

    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Number of Channels: {num_channels}")

    return sample_rate == 16000 and num_channels == 1

def preprocess_mp3(input_path, output_path):
    """Convert MP3 file to 16kHz mono format."""
    audio, sample_rate = torchaudio.load(input_path)
    
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    
    torchaudio.save(output_path, audio, 16000)

def perform_lyric_alignment(mp3_path, path_lyric):
    """Perform lyric alignment using the audio file and lyric data."""
    temp_mp3_path = 'temp.mp3'
    
    print(f"Preprocessing {mp3_path} to {temp_mp3_path}...")
    preprocess_mp3(mp3_path, temp_mp3_path)

    if not check_audio_properties(temp_mp3_path):
        print("Audio file does not meet the required specifications of 16 kHz and single channel. Aborting alignment.")
        sys.exit(1)

    model_path = 'nguyenvulebinh/lyric-alignment'
    global model, tokenizer, feature_extractor
    model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    wav, _ = torchaudio.load(temp_mp3_path)

    with open(path_lyric, 'r', encoding='utf-8') as file:
        lyric_data = json.load(file)

    lyric_alignment = handle_sample(wav, lyric_data)

    os.remove(temp_mp3_path)
    
    return lyric_alignment

if __name__ == "__main__":
    input_txt = 'data/songs_of_the_sea/vi-no/2.txt'  # Path to the input text file
    output_json = 'data/songs_of_the_sea/vi-no/formatted/2.json'  # Path to the output formatted JSON file
    mp3_file = 'data/songs_of_the_sea/vi-no/2.mp3'  # Path to the MP3 file

    # Format the input text file into JSON
    format_input_txt_to_json(input_txt, output_json)

    # Perform lyric alignment
    alignment_output = perform_lyric_alignment(mp3_file, output_json)

    # Print the output alignment (or save it to a file)
    print(json.dumps(alignment_output, ensure_ascii=False, indent=2))
