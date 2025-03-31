#!/usr/bin/env python3

import speech_recognition as sr

# obtain path to "english.wav" in the same folder as this script
from os import path

import csv
import re

import os
import subprocess

import pandas as pd
import matplotlib.pyplot as plt

# Collect data
def plot(results_dir="results"):
    percentages = {}
    for filename in os.listdir(results_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(results_dir, filename)
            df = pd.read_csv(filepath)  # Read CSV

            # Count ones and calculate percentage
            ones_count = df.iloc[:, 1].sum()  # Sum the second column (0s and 1s)
            total_count = len(df)
            percentage = (ones_count / total_count) * 100 if total_count > 0 else 0

            percentages[filename] = percentage

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(percentages.keys(), percentages.values(), color='skyblue')
    plt.xticks(rotation=45, ha="right")  # Rotate x labels for readability
    plt.ylabel("Percentage of Ones (%)")
    plt.title("Percentage of Ones in CSV Files")
    plt.tight_layout()

    # Save or show plot
    plt.savefig("results_percentage_plot.png")  # Save the plot as an image
    plt.show()

def concat(csv_file):
        # Read the CSV and concatenate the first column
    text = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            text.append(row[0])  # First column (text)

    # Convert list to a single string
    result = " ".join(text)

    # Print or save to a file
    return result


def transform():
    input_dir = "testcases"
    output_dir = "temp"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".webm"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".wav"
            output_path = os.path.join(output_dir, output_filename)

            subprocess.run(["ffmpeg", "-i", input_path, "-q:a", "0", "-map", "a", output_path], check=True)

    print("Conversion complete.")

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
    transform()
    d = "temp"
    for filename in os.listdir(d):
        AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "temp/"+filename)
        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio = r.record(source)  # read the entire audio file
        trancription = ""
        try:
            transcription = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        #real = read_file_to_string("at.txt")
        real = concat("transcriptions/"+filename[:-4] + "_words.csv")
        compare_strings(real, transcription, "results/" + filename[:-4]+"_results.csv")
    plot()

main()
    

