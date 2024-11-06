# includes
import torch
import torchaudio
#from beam import DecoderRNN
from decoder import GreedyCTCDecoder
from spellchecker import SpellChecker
import IPython
import matplotlib.pyplot as plt
import argparse


# tutorial from
# https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html#overview


# a dataset https://huggingface.co/datasets/Nexdata/American_Children_Speech_Data_by_Microphone/blob/main/T0003G0036S0001.wav

def spell_check_and_correct(input_string):
    spell = SpellChecker()

    words = input_string.split()

    corrected_words = [spell.correction(word) if spell.correction(word) is not None else word for word in words]


    #corrected_string = ' '.join(corrected_words)

    return corrected_words

def check_percentage(sample_file, test_file):
    n = 0
    total_correct = 0
    sample_file = sample_file.split()
    test_file = test_file.split()
    print(sample_file, test_file)
    for i in range(len(sample_file)):
        n += 1
        if i < len(test_file) and  test_file[i] == sample_file[i]:
            total_correct += 1


    return float(float(total_correct) / float(n))
def get_labels():
    # Standard character set (modify based on the task/language)
    return ["<pad>",  # Padding token
    "|",      # Word separator (space)
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", 
    "r", "s", "t", "u", "v", "w", "x", "y", "z", "'",
    "<unk>",  # Unknown token
    "<ctc_blank>"  # CTC blank token
    ]


# helper function to plot features
def plot_features(features):
    fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
    for i, feats in enumerate(features):
        ax[i].imshow(feats[0].cpu(), interpolation="nearest")
        ax[i].set_title(f"Feature from transformer layer {i+1}")
        ax[i].set_xlabel("Feature dimension")
        ax[i].set_ylabel("Frame (time-axis)")
    fig.tight_layout()
    fig.savefig("figures/feature_visualization.png")


# helper function to plot emissions
def plot_emissions(emission, bundle):
    plt.imshow(emission[0].cpu().T, interpolation="nearest")
    plt.title("Classification result")
    plt.xlabel("Frame (time-axis)")
    plt.ylabel("Class")
    plt.tight_layout()
    # print("Class labels:", bundle.get_labels())
    plt.savefig("figures/classification_results.png")


def main():
    # argument management
    parser = argparse.ArgumentParser(
        prog="Convert speech to text.",
        description="This program uses the WAV2VEC and ASR_BASE_960H model to convert speech to text.",
        epilog="Note: Most code at this point comes from PyTorch Wave2Vec Tutorial.",
    )
    parser.add_argument(
        "filename", type=str, help="The path to the .wav file wanting to be decoded"
    )
    args = parser.parse_args()
    SPEECH_FILE = args.filename

    # prints version of out python
    print(torch.__version__)
    print(torchaudio.__version__)

    # seed to reproducability, and device selection
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # what we are decoding
    # SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

    # blundle and sample rates
    # bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    # print(f"mode: {bundle.get_model()}")
    print("Sample Rate:", bundle.sample_rate)
    # print("Labels:", bundle.get_labels())

    # model we are using
    model = bundle.get_model().to(device)
    print(model.__class__)

    # load in our audio into the wave form
    waveform, sample_rate = torchaudio.load(SPEECH_FILE)

    # put the wave form onto our GPU for speed
    waveform = waveform.to(device)

    """
    Pretrained models expect a specific sample rate. The sample rate is how quickly the amplitiude of the 
    sound wave is measured at a given time. Higher sample rates can capture more detail in the code. In this case
    the model is trained on a specific rate, if the audio we are using is NOT the same rate we must resample it.
    """

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate
        )

    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    plot_features(features=features)

    with torch.inference_mode():
        emission, _ = model(waveform)

    plot_emissions(emission=emission, bundle=bundle)


    #embedding_size = 256   
    #hidden_size = 512      
    #output_size = len(bundle.get_labels)  
    #cell_type = 'LSTM'     

    #decoder = DecoderRNN()

    # this is when the model actually decodes our sample. We are using a GREEDY alogirthm for this approach.
    decoder = GreedyCTCDecoder(bundle.get_labels())
    transcript = decoder(emission[0])

    #print(transcript)
    #IPython.display.Audio(SPEECH_FILE)

    transcript = str(transcript).replace("|", " ")
    print((' '.join(spell_check_and_correct(transcript))).lower())




if __name__ == "__main__":
    main()
