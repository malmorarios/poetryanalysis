import numpy as np
import tensorflow as tf
from tensorflow import keras

batch_size = 64  # Batch size for training. --> ???
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space // i.e. # of hidden nodes // i.e. "NUMBER OF LSTM UNITS"
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "fra.txt"

""" 
Data is a .txt file with lines of the form:

I'm back.   Je suis revenu.	    CC-BY 2.0 (France) Attribution: tatoeba.org #564159 (darinmex) & #591724 (qdii)
I'm lazy.	Je suis paresseux.	CC-BY 2.0 (France) Attribution: tatoeba.org #2203056 (CK) & #1133774 (sacredceltic)
I'm sure.	J'en suis sûr.	    CC-BY 2.0 (France) Attribution: tatoeba.org #2283730 (CK) & #2285478 (sacredceltic)
Etc.

"""

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")

    # We use "tab" as the "start sequence" character for the targets, 
    #   and "\n" as the "end sequence" character.

    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)

    # Based on the collection of input texts (target texts) from data, 
    #   generate the set of all inputted characters (target characters)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0 # here, since they have an official "no character" character in their character list, they fill the REST of the ith array (translation) with indicators of "no more characters here"
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be "ahead" by one timestep --> TODO (10/31/2020): why?? Don't really understand the difference between target data and input data
            # and will not include the start character. 
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0



# Define an input sequence and process it. TODO (10/31/2020): START HERE NEXT WORK SESH (https://keras.io/examples/nlp/lstm_seq2seq/). 
#   Use the info you gained about the #return_sequence and #return_state parameters to more clearly decipher what is going on below and continue from there!


encoder_inputs = keras.Input(shape=(None, num_encoder_tokens)) # "Number of unique input tokens" to the model (this is what dictates what should act as the Keras input, I believe)
encoder = keras.layers.LSTM(latent_dim, return_state=True) # "Number of hidden nodes" in the LSTM // "NUMBER OF LSTM UNITS" // Also remember that 'return_state=True' indicates our LSTM model to return the last states (cell + hidden) in addition to our output
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states. --> this is how the encoder-decoder LSTM model works (we don't care about encoder outputs until our LAST time step)
encoder_states = [state_h, state_c] # it APPEARS like THIS is the context vector that we want the encoder LSTM components to spit out [viewing it as such because this is what is being fed as input to the decoder]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True) # recall that return_sequences=True returns the last output in the output sequence.
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
