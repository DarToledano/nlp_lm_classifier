import torch

# Model and training configuration
SEQ_LEN = 30
BATCH_SIZE = 64
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
