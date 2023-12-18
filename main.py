import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def one_hot_encoder(encoded_text, num_uni_chars):
    '''
    encoded_text : batch of encoded text
    num_uni_chars = number of unique characters (len(set(text)))
    # METHOD FROM: https://stackoverflow.com/questions/29831489/convert-encoded_textay-of-indices-to-1-hot-encoded-numpy-encoded_textay
    '''
    one_hot = np.zeros((encoded_text.size, num_uni_chars)) # Create a placeholder for zeros.
    one_hot = one_hot.astype(np.float32) # Convert data type for later use with pytorch (errors if we dont!)
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0 # Using fancy indexing fill in the 1s at the correct index locations
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars)) # Reshape it so it matches the batch sahe
    return one_hot

def generate_batches(encoded_text, samp_per_batch=10, seq_len=50):
    '''
    Generate (using yield) batches for training.
    X: Encoded Text of length seq_len
    Y: Encoded Text shifted by one
    Example:
    X:
    [[1 2 3]]
    Y:
    [[ 2 3 4]]
    encoded_text : Complete Encoded Text to make batches from
    batch_size : Number of samples per batch
    seq_len : Length of character sequence
    '''

    char_per_batch = samp_per_batch * seq_len # Total number of characters per batch, # Example: If samp_per_batch is 2 and seq_len is 50, then 100, # characters come out per batch.
    num_batches_avail = int(len(encoded_text)/char_per_batch)  # Number of batches available to make,# Use int() to roun to nearest integer
    encoded_text = encoded_text[:num_batches_avail * char_per_batch]  # Cut off end of encoded_text that,# won't fit evenly into a batch
    encoded_text = encoded_text.reshape((samp_per_batch, -1)) # Reshape text into rows the size of a batch
    for n in range(0, encoded_text.shape[1], seq_len):         # Go through each row in array.       
        x = encoded_text[:, n:n+seq_len] # Grab feature characters
        y = np.zeros_like(x) # y is the target shifted over by 1
        try:
            y[:, :-1] = x[:, 1:]
            y[:, -1]  = encoded_text[:, n+seq_len]
        except: # FOR POTENTIAL INDEXING ERROR AT THE END
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
        yield x, y


class CharModel(nn.Module):
    def __init__(self, all_chars, num_hidden=256, num_layers=4,drop_prob=0.5,use_gpu=False):
        # SET UP ATTRIBUTES
        super().__init__()
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.use_gpu = use_gpu
        self.all_chars = all_chars    #CHARACTER SET, ENCODER, and DECODER
        self.decoder = dict(enumerate(all_chars))
        self.encoder = {char: ind for ind,char in decoder.items()}
        self.lstm = nn.LSTM(len(self.all_chars), num_hidden, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc_linear = nn.Linear(num_hidden, len(self.all_chars))
    
    def forward(self, x, hidden):
        lstm_output, hidden = self.lstm(x, hidden)
        drop_output = self.dropout(lstm_output)
        drop_output = drop_output.contiguous().view(-1, self.num_hidden)
        final_out = self.fc_linear(drop_output)
        return final_out, hidden
    
    def hidden_state(self, batch_size):
        '''
        Used as separate method to account for both GPU and CPU users.
        '''
        if self.use_gpu: 
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda(),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden).cuda())
        else:
            hidden = (torch.zeros(self.num_layers,batch_size,self.num_hidden),
                     torch.zeros(self.num_layers,batch_size,self.num_hidden))
        
        return hidden

# Generating Predictions
def predict_next_char(model, char, hidden=None, k=1):
        encoded_text = model.encoder[char] # Encode raw letters with model
        encoded_text = np.array([[encoded_text]]) # NOTE THE [[ ]] dimensions!! # set as numpy array for one hot encoding
        encoded_text = one_hot_encoder(encoded_text, len(model.all_chars)) # One hot encoding
        inputs = torch.from_numpy(encoded_text) # Convert to Tensor
        if(model.use_gpu): # Check for CPU
            inputs = inputs.cuda()
        hidden = tuple([state.data for state in hidden]) # Grab hidden states
        lstm_out, hidden = model(inputs, hidden) # Run model and get predicted output
        probs = F.softmax(lstm_out, dim=1).data # Convert lstm_out to probabilities
        if(model.use_gpu):
            probs = probs.cpu() # move back to CPU to use with numpy
        probs, index_positions = probs.topk(k) # Return k largest probabilities in tensor    # k determines how many characters to consider # for our probability choice.   # https://pytorch.org/docs/stable/torch.html#torch.topk
        index_positions = index_positions.numpy().squeeze()
        probs = probs.numpy().flatten() # Create array of probabilities
        probs = probs/probs.sum() # Convert to probabilities per index
        char = np.random.choice(index_positions, p=probs) # randomly choose a character based on probabilities
        return model.decoder[char], hidden # return the encoded value of the predicted char and the hidden state

def generate_text(model, size, seed='The', k=1):
    if(model.use_gpu):  # CHECK FOR GPU
        model.cuda()
    else:
        model.cpu()
    model.eval()  # Evaluation mode
    output_chars = [c for c in seed]  # begin output from initial seed
    hidden = model.hidden_state(1)   # intiate hidden state
    for char in seed:  # predict the next character for every character in seed
        char, hidden = predict_next_char(model, char, hidden, k=k)
    output_chars.append(char)   # add initial characters to output
    for i in range(size):    # Now generate for size requested
        char, hidden = predict_next_char(model, output_chars[-1], hidden, k=k)     # predict based off very last letter in output_chars
        output_chars.append(char)     # add predicted character
    return ''.join(output_chars)     # return string of predicted text


with open('Data/shakespeare.txt','r',encoding='utf8') as f:
    text = f.read()
    all_characters = set(text)
    decoder = dict(enumerate(all_characters))
    encoder = {char: ind for ind,char in decoder.items()}
    encoded_text = np.array([encoder[char] for char in text])

print(text[:1000])
print(len(text))
text[:1000]
one_hot_encoder(np.array([1,2,0]),3)
"""
# Creating Training Batches
We need to create a function that will generate batches of characters along with the next character in the sequence as a label.
"""
example_text = np.arange(10)
example_text.reshape((5,-1))
"""
### Example of generating a batch
"""
sample_text = encoded_text[:20]
batch_generator = generate_batches(sample_text,samp_per_batch=2,seq_len=5)
# Grab first batch
x, y = next(batch_generator)
"""
## GPU Check
Remember this will take a lot longer on CPU!

"""
torch.cuda.is_available()
"""
# Creating the LSTM Model

**Note! We will have options for GPU users and CPU users. 
CPU will take MUCH LONGER to train and you may encounter 
RAM issues depending on your hardware. 
If that is the case, consider using cloud services like AWS, GCP, or Azure. Note, these may cost you money to use!**
"""

"""
## Instance of the Model
"""
model = CharModel(
    all_chars=all_characters,
    num_hidden=512,
    num_layers=3,
    drop_prob=0.5,
    use_gpu=True,
)
# print(generate_text(model, 1000, seed='The ', k=3))
total_param  = []
for p in model.parameters():
    total_param.append(int(p.numel()))
"""
Try to make the total_parameters be roughly the same magnitude as the number of characters in the text.
"""
sum(total_param)
len(encoded_text)

### Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

## Training Data and Validation Data
# percentage of data to be used for training
train_percent = 0.1
len(encoded_text)
int(len(encoded_text) * (train_percent))
train_ind = int(len(encoded_text) * (train_percent))
train_data = encoded_text[:train_ind]
val_data = encoded_text[train_ind:]

# Training the Network
## VARIABLES # Feel free to play around with these values
epochs = 50 # Epochs to train for
batch_size = 128 # batch size 
seq_len = 100 # Length of sequence
tracker = 0 # for printing report purposes # always start at 0
num_char = max(encoded_text)+1 # number of characters in text


model.train() # Set model to train
if model.use_gpu: # Check to see if using GPU
    model.cuda()
    pass

for i in range(epochs):
    hidden = model.hidden_state(batch_size)
    for x,y in generate_batches(train_data,batch_size,seq_len):
        tracker += 1 
        x = one_hot_encoder(x,num_char) # One Hot Encode incoming data
        inputs = torch.from_numpy(x) # Convert Numpy Arrays to Tensor
        targets = torch.from_numpy(y)
        if model.use_gpu: # Adjust for GPU if necessary
            inputs = inputs.cuda()
            targets = targets.cuda()
        hidden = tuple([state.data for state in hidden]) # Reset Hidden State     # If we dont' reset we would backpropagate through all training history
        model.zero_grad()
        lstm_output, hidden = model.forward(inputs,hidden)
        loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
        loss.backward()
        # POSSIBLE EXPLODING GRADIENT PROBLEM!
        # LET"S CLIP JUST IN CASE
        nn.utils.clip_grad_norm_(model.parameters(),max_norm=5)
        optimizer.step()
        ###################################
        ### CHECK ON VALIDATION SET ######
        #################################
        if tracker % 25 == 0:
            val_hidden = model.hidden_state(batch_size)
            val_losses = []
            model.eval()
            for x,y in generate_batches(val_data,batch_size,seq_len):
                x = one_hot_encoder(x,num_char) # One Hot Encode incoming data
                inputs = torch.from_numpy(x) # Convert Numpy Arrays to Tensor
                targets = torch.from_numpy(y)
                # Adjust for GPU if necessary
                if model.use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                val_hidden = tuple([state.data for state in val_hidden]) # If we dont' reset we would backpropagate through     # all training history    # Reset Hidden State
                lstm_output, val_hidden = model.forward(inputs,val_hidden)
                val_loss = criterion(lstm_output,targets.view(batch_size*seq_len).long())
                val_losses.append(val_loss.item())
            # Reset to training model after val for loop
            model.train()
            print(f"Epoch: {i} Step: {tracker} Val Loss: {val_loss.item()}")

"""

"""


