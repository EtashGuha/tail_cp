from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np

def get_data(args):
    name = args.dataset_name

    if name == "diabetes":
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target
    elif name == "bio":
        df = pd.read_csv("datasets/CASP.csv")        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values 
    elif name == "concrete":
        dataset = np.loadtxt(open('datasets/Concrete_Data.csv', "rb"), delimiter=",", skiprows=1)
        X = dataset[:, :-1]
        y = dataset[:, -1:]
    elif name == "bimodal":
        with open("datasets/bimodal.pkl", "rb") as f:
            X, y = pickle.load(f)



    

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()  # Reshape y for scaler

    return torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(y_normalized, dtype=torch.float32)

def get_loaders(args):
    name = args.dataset_name
    X_normalized, y_normalized = get_data(args)
    # Split the normalized data and labels into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_normalized, test_size=args.test_size, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Create a DataLoader for training and validation data
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, val_loader

def get_input_and_range(args):
    name = args.dataset_name

    X_train, y_train = get_data(args)
    input_size = X_train.shape[1]
    range_vals = torch.linspace(torch.min(y_train), torch.max(y_train), args.range_size)
    return input_size, range_vals

def get_val_data(args):
    name = args.dataset_name
    X_normalized, y_normalized = get_data(args)
    # Split the normalized data and labels into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_normalized, test_size=args.test_size, random_state=42)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    return X_val, y_val