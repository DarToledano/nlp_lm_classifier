import torch

# Model and training configuration
SEQ_LEN = 150
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.3
EPOCHS = 10
LEARNING_RATE = 0.0005

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
