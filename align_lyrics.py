import torchaudio
import json
from model_handling import Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoFeatureExtractor

# Load the model and tokenizer
model_path = 'nguyenvulebinh/lyric-alignment'
model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

def convert_to_spoken_form(lyric_data):
    # Convert lyrics to their spoken forms
    spoken_data = []
    for word in lyric_data:
        spoken_word = word  # You may implement specific conversion logic here
        spoken_data.append(spoken_word)
    return spoken_data

def ctc_segment(wav, spoken_data):
    # Implement CTC segmentation logic here using the model
    input_values = feature_extractor(wav.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return tokenizer.batch_decode(predicted_ids)  # Adjust as needed for your segmentation

def re_align_to_original_words(adjusted_segments, lyric_data):
    # Re-align the adjusted segments back to original words
    # Implement this logic based on your specific requirements
    final_alignment = []
    for segment in adjusted_segments:
        # Match segments to original words here
        final_alignment.append(segment)  # Example implementation
    return final_alignment

def align_lyrics(wav_path, path_lyric):
    # Load the audio file
    wav, _ = torchaudio.load(wav_path)
    
    # Load the lyric data
    with open(path_lyric, 'r', encoding='utf-8') as file:
        lyric_data = json.load(file)
    
    # Step 1: Convert lyrics to spoken form
    spoken_data = convert_to_spoken_form([word['d'] for segment in lyric_data for word in segment['l']])
    
    # Step 2: Use CTC-Segmentation for alignment
    word_segments = ctc_segment(wav, spoken_data)

    # Step 3: Apply heuristic rules for timestamp adjustments
    adjusted_segments = apply_heuristic_adjustments(word_segments)

    # Step 4: Re-align the spoken form back to original words
    final_alignment = re_align_to_original_words(adjusted_segments, lyric_data)

    return final_alignment

def apply_heuristic_adjustments(word_segments):
    # Implement timestamp adjustment logic here
    for i in range(len(word_segments) - 1):
        word = word_segments[i]
        next_word = word_segments[i + 1]
        # Example adjustment
        word.end = next_word.start  # Adjust to next word's start time
    return word_segments

# Example usage
wav_path = 'data/vi-no/6.mp3'
path_lyric = 'data/vi-no/6.json'
aligned_lyrics = align_lyrics(wav_path, path_lyric)
print(json.dumps(aligned_lyrics, ensure_ascii=False, indent=2))
