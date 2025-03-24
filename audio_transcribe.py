#!/usr/bin/env python3

import speech_recognition as sr

# obtain path to "english.wav" in the same folder as this script
from os import path

import csv
import re

def compare_strings(real_string, transcribed_string, output_file="comparison_results.csv"):
    """
    Compare words in a real string with words in a transcribed string.
    For each word in the real string, output whether it appears in the transcribed string.
    
    Args:
        real_string (str): The original/real string
        transcribed_string (str): The transcribed string to compare against
        output_file (str): Name of the output CSV file
    
    Returns:
        list: List of tuples containing (word, presence_flag)
    """
    # Normalize strings: convert to lowercase and split into words
    real_words = re.findall(r'\w+', real_string.lower())
    transcribed_words = set(re.findall(r'\w+', transcribed_string.lower()))
    
    # Check if each real word appears in the transcribed string
    results = []
    for word in real_words:
        is_present = 1 if word in transcribed_words else 0
        results.append((word, is_present))
    
    # Write results to CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Present'])  # Header
        writer.writerows(results)
    
    return results

def read_file_to_string(file_path):
    """
    Read the contents of a text file into a string
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Contents of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def main():
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "french.aiff")
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "chinese.flac")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google Speech Recognition
    trancription = ""
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        transcription = r.recognize_google(audio)
    #    print("Google Speech Recognition thinks you said " + transcription)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    #except sr.RequestError as e:
    #    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    real = read_file_to_string("at.txt")
    compare_strings(real, transcription)

main()
    

