import pandas as pd  # pip install pandas
import torch  # pip install torch torchvision torchaudio
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np  # pip install numpy
import random
import os
from tqdm import tqdm  # pip install tqdm
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")