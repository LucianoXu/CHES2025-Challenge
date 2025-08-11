from typing import Callable, Literal, Iterable, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Dataset
from src.utils import calculate_HW_single, load_ctf_2025
import torch
from .config import Config
import pickle

# denoising function
def notch_many_fft(
    x: Union[torch.Tensor, "np.ndarray"],
    fs: float,
    freqs: Iterable[float],
    Q: float = 40.0,                 # Q ≈ f0 / (-3 dB bandwidth)
    taper_ratio: float = 0.5,        # 0..1, fraction of half-band used for cosine skirts
    atten_db: float = 60.0,          # depth of the notch (in dB)
    mix: float = 1.0,                # 1.0 = full notch, <1.0 = wet/dry mix
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Narrow-band noise removal using smooth frequency-domain notches.

    Args:
        x: 1D (T,) or 2D (N, T) real signal. torch tensor or numpy array.
        fs: Sample rate in Hz.
        freqs: Iterable of center frequencies to notch (Hz).
        Q: Quality factor; -3 dB bandwidth BW ≈ f0 / Q. Larger Q => narrower notch.
        taper_ratio: Cosine taper fraction of the half-band. 0 = brick-wall, 0.5 = pure cosine.
        atten_db: Notch depth in dB (amplitude domain). 60 dB means ~1e-3 gain at center.
        mix: Wet/dry mix. y = mix * filtered + (1-mix) * original.
        device: Torch device. If None, inferred from x or defaults to CPU.

    Returns:
        y : filtered signal with same shape/type as input.
    """
    # --- Normalize input to torch tensor (float32) and batch shape (B, T) ---
    is_numpy = False
    try:
        import numpy as _np
        if not torch.is_tensor(x):
            is_numpy = True
            x = torch.from_numpy(_np.asarray(x))
    except Exception:
        pass

    if device is None:
        device = x.device if torch.is_tensor(x) else torch.device("cpu")

    assert isinstance(x, torch.Tensor)
    
    x = x.to(device).float()
    if x.ndim == 1:
        x = x[None, :]  # (1, T)

    B, T = x.shape
    n_r = T // 2 + 1

    # --- rFFT and frequency bins ---
    X = torch.fft.rfft(x, dim=-1)  # (B, n_r)
    # rFFT bin frequencies: f_k = k * fs / T
    f_bins = torch.arange(n_r, device=device, dtype=x.dtype) * (fs / T)

    # --- Build smooth multiplicative mask: start from ones (pass-through) ---
    mask = torch.ones(n_r, device=device, dtype=x.dtype)

    # Precompute center attenuation (amplitude gain)
    att = 10.0 ** (-atten_db / 20.0)

    for f0 in freqs:
        # Skip invalid or out-of-band centers
        if not (0.0 < f0 < fs * 0.5):
            continue

        # Half-bandwidth by Q: H = BW/2 ≈ f0 / (2Q)
        H = float(f0) / (2.0 * float(Q))
        if H <= 0:
            continue

        # Split half-band into a flat "core" and cosine "skirts"
        taper = max(taper_ratio * H, 0.0)
        core  = max(H - taper, 0.0)

        # Distance from the center frequency
        delta = (f_bins - f0).abs()

        if taper > 0:
            # Smooth gain profile:
            #   delta <= core: gain = att
            #   core < delta < core+taper: cosine ramp from att -> 1
            #   delta >= core+taper: gain = 1
            t = ((delta - core) / taper).clamp(0.0, 1.0)          # 0..1
            r = 0.5 - 0.5 * torch.cos(torch.pi * t)               # 0..1 (cosine ease)
            gain = att + (1.0 - att) * r                          # att..1
        else:
            # Brick-wall notch within |delta| <= H
            gain = torch.where(delta <= H, torch.tensor(att, device=device, dtype=x.dtype),
                                           torch.tensor(1.0, device=device, dtype=x.dtype))

        mask = mask * gain  # combine multiple notches multiplicatively

    # --- Apply mask and inverse rFFT ---
    Xf = X * mask[None, :]
    y = torch.fft.irfft(Xf, n=T, dim=-1)

    # Wet/dry mix
    if mix != 1.0:
        y = mix * y + (1.0 - mix) * x

    # Restore original shape/type
    y = y.squeeze(0) if B == 1 else y
    if is_numpy:
        y = y.cpu().numpy()

    return y


def chunked_notch_many_fft(
    x: np.ndarray,
    fs: float,
    notch_freqs: Iterable[float],
    Q: float = 40.0,
    taper_ratio: float = 0.5,
    atten_db: float = 60.0,
    mix: float = 1.0,
    device: Optional[torch.device] = None,
    chunk_size: int = 100_000
) -> np.ndarray:
    """Chunked version of notch_many_fft for large inputs."""
    # Split input into chunks to avoid memory issues
    num_chunks = (x.shape[-1] + chunk_size - 1) // chunk_size

    y_chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, x.shape[-1])
        y_chunk = notch_many_fft(x[..., start:end], fs, notch_freqs, Q, taper_ratio, atten_db, mix, device)
        y_chunks.append(y_chunk)

    if len(y_chunks) == 1:
        return y_chunks[0]
    else:
        return torch.cat(y_chunks, dim=-1).cpu().numpy()
    
def calculating_data_drift_fixing_matrices(X_profiling: np.ndarray, X_attack: np.ndarray, K_profiling: np.ndarray):
    '''
    Calculate the matrices that will be used to fix the data drift of the test set.
    '''
    X_profiling_tensor = torch.from_numpy(X_profiling).float().to('cuda')
    X_attack_tensor = torch.from_numpy(X_attack).float().to('cuda')

    mean_profiling = X_profiling_tensor.mean(dim=0, keepdim=False).cpu().numpy()
    std_profiling = X_profiling_tensor.std(dim=0, keepdim=False, unbiased=False).cpu().numpy()

    mean_attack = X_attack_tensor.mean(dim=0, keepdim=False).cpu().numpy()
    std_attack = X_attack_tensor.std(dim=0, keepdim=False, unbiased=False).cpu().numpy()

    # collect all traces with key 127 from profiling data
    X_profiling_raw_key_127 = []
    for i in range(len(X_profiling)):
        if K_profiling[i] == 127:
            X_profiling_raw_key_127.append(X_profiling[i])

    X_profiling_raw_key_127 = np.array(X_profiling_raw_key_127)

    mean_profiling_raw_key_127 = X_profiling_raw_key_127.mean(axis=0)
    std_profiling_raw_key_127 = X_profiling_raw_key_127.std(axis=0, ddof=0)

    matrices = {
        'mean_profiling': mean_profiling,
        'std_profiling': std_profiling,
        'mean_attack': mean_attack,
        'std_attack': std_attack,
        'mean_profiling_raw_key_127': mean_profiling_raw_key_127,
        'std_profiling_raw_key_127': std_profiling_raw_key_127
    }

    return matrices

def standardize(X_train: np.ndarray, X_test: np.ndarray, device: str = 'cuda'):
    """
    GPU-accelerated standardization using PyTorch.
    
    Args:
        X_train: Training data to fit the scaler on
        X_test: Test data to transform
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Tuple of (standardized_X_train, standardized_X_test) as numpy arrays
    """
    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    # Calculate mean and std on training data
    mean = X_train_tensor.mean(dim=0, keepdim=True)
    std = X_train_tensor.std(dim=0, keepdim=True, unbiased=False)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    # Standardize both datasets
    X_train_normalized = (X_train_tensor - mean) / std
    X_test_normalized = (X_test_tensor - mean) / std
    
    # Convert back to numpy arrays on CPU
    return X_train_normalized.cpu().numpy(), X_test_normalized.cpu().numpy()

def standardize_by_matrixfile(X_train: np.ndarray, X_test: np.ndarray, std_m: dict[str, np.ndarray], device: str = 'cuda'):
    """
    Standardize the data using pre-computed mean and std matrices from a file, and fix data drift of the test set.

    Args:
        X_train: Training data to fit the scaler on, shape (N, T, 1) or (N, T)
        X_test: Test data to transform, shape (N, T, 1) or (N, T)
        std_m: Dictionary containing pre-computed mean and std matrices
        device: Device to use for computation ('cuda' or 'cpu')
    
    Returns:
        Tuple of (standardized_X_train, standardized_X_test) as numpy arrays
    """

    mean_profiling = torch.from_numpy(std_m['mean_profiling']).float().to(device)
    std_profiling = torch.from_numpy(std_m['std_profiling']).float().to(device)
    mean_attack = torch.from_numpy(std_m['mean_attack']).float().to(device)
    std_attack = torch.from_numpy(std_m['std_attack']).float().to(device)
    mean_profiling_raw_key_127 = torch.from_numpy(std_m['mean_profiling_raw_key_127']).float().to(device)
    std_profiling_raw_key_127 = torch.from_numpy(std_m['std_profiling_raw_key_127']).float().to(device)

    # Convert to PyTorch tensors and move to GPU
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    # this step transform the drifted test set to the same distribution as the profiling set
    # it seems that only fixing the mean is enough
    X_test_tensor = (X_test_tensor - mean_attack) + mean_profiling_raw_key_127
    # X_test_tensor = (X_test_tensor - mean_attack) * (std_profiling_raw_key_127 / std_attack) + mean_profiling_raw_key_127


    # standardize the data
    X_test_normalized = (X_test_tensor - mean_profiling) / std_profiling
    X_train_normalized = (X_train_tensor - mean_profiling) / std_profiling

    # Convert back to numpy arrays on CPU
    return X_train_normalized.cpu().numpy(), X_test_normalized.cpu().numpy()

def load_data(config: Config, device: str = 'cuda'):
    '''
    Load the data from the dataset specified in the configuration.

    Args:
        config: Configuration object containing dataset parameters
        device: Device to use for data processing ('cuda' or 'cpu')

    Returns:
        Tuple of training, validation, and test datasets as numpy arrays
        X_train, X_val, X_test: shape (N, T, 1)
        other arrays: shape (N, )
    '''

    profiling_in_train = config["profiling_in_train"]
    attack_in_train = config["attack_in_train"]
    profiling_in_val = config["profiling_in_val"]
    attack_in_val = config["attack_in_val"]
    attack_in_test = config["attack_in_test"]

    (X_profiling, X_attack), \
    (Y_profiling, Y_attack), \
    (P_profiling, P_attack), \
    (K_profiling, K_attack) = load_ctf_2025(
        config["dataset"],
        byte=0, 
        train_begin=0, train_end=profiling_in_train + profiling_in_val, 
        test_begin=0, test_end=attack_in_train + attack_in_val + attack_in_test,)
    
    # denoising (the arguments are hardcoded)
    if config["denoising"]:
        print("Denoising traces ...")

        X_profiling = chunked_notch_many_fft(X_profiling, **config["denoising_args"], device=torch.device('cuda'))
        X_attack = chunked_notch_many_fft(X_attack, **config["denoising_args"], device=torch.device('cuda'))


    # standardize the data according to the matrix file (fix data drift)
    print("Standardizing traces ...")
    if config["fixing_data_drift"]:
        # calculate the standardization matrices
        std_m = calculating_data_drift_fixing_matrices(X_profiling, X_attack, K_profiling)
        # apply the standardization matrices
        X_profiling, X_attack = standardize_by_matrixfile(
            X_profiling, X_attack,
            std_m=std_m,
            device=device
        )
    else:    
        # standardize the data according to profiling set
        X_profiling, X_attack = standardize(X_profiling, X_attack, device=device)

    # split train into train and validation
    X_profiling_in_train = X_profiling[:profiling_in_train]
    Y_profiling_in_train = Y_profiling[:profiling_in_train]
    P_profiling_in_train = P_profiling[:profiling_in_train]
    K_profiling_in_train = K_profiling[:profiling_in_train]

    X_profiling_in_val = X_profiling[profiling_in_train:profiling_in_train + profiling_in_val]
    Y_profiling_in_val = Y_profiling[profiling_in_train:profiling_in_train + profiling_in_val]
    P_profiling_in_val = P_profiling[profiling_in_train:profiling_in_train + profiling_in_val]
    K_profiling_in_val = K_profiling[profiling_in_train:profiling_in_train + profiling_in_val]

    X_attack_in_train = X_attack[:attack_in_train]
    Y_attack_in_train = Y_attack[:attack_in_train]
    P_attack_in_train = P_attack[:attack_in_train]
    K_attack_in_train = K_attack[:attack_in_train]

    X_attack_in_val = X_attack[attack_in_train:attack_in_train + attack_in_val]
    Y_attack_in_val = Y_attack[attack_in_train:attack_in_train + attack_in_val]
    P_attack_in_val = P_attack[attack_in_train:attack_in_train + attack_in_val]
    K_attack_in_val = K_attack[attack_in_train:attack_in_train + attack_in_val]

    X_attack_in_test = X_attack[attack_in_train + attack_in_val:attack_in_train + attack_in_val + attack_in_test]
    Y_attack_in_test = Y_attack[attack_in_train + attack_in_val:attack_in_train + attack_in_val + attack_in_test]
    P_attack_in_test = P_attack[attack_in_train + attack_in_val:attack_in_train + attack_in_val + attack_in_test]
    K_attack_in_test = K_attack[attack_in_train + attack_in_val:attack_in_train + attack_in_val + attack_in_test]

    # concatenate to create training, validation, and test sets
    X_train = np.concatenate((X_profiling_in_train, X_attack_in_train), axis=0)
    X_train = np.expand_dims(X_train, -1)  # add channel dimension
    Y_train = np.concatenate((Y_profiling_in_train, Y_attack_in_train), axis=0)
    P_train = np.concatenate((P_profiling_in_train, P_attack_in_train), axis=0)
    K_train = np.concatenate((K_profiling_in_train, K_attack_in_train), axis=0)

    X_val = np.concatenate((X_profiling_in_val, X_attack_in_val), axis=0)
    X_val = np.expand_dims(X_val, -1)  # add channel dimension
    Y_val = np.concatenate((Y_profiling_in_val, Y_attack_in_val), axis=0)
    P_val = np.concatenate((P_profiling_in_val, P_attack_in_val), axis=0)
    K_val = np.concatenate((K_profiling_in_val, K_attack_in_val), axis=0)

    X_test = np.expand_dims(X_attack_in_test, -1)  # add channel dimension
    Y_test = Y_attack_in_test
    P_test = P_attack_in_test
    K_test = K_attack_in_test   
    
    torch.cuda.empty_cache()  # clear GPU memory

    return (X_train, Y_train, P_train, K_train), (X_val, Y_val, P_val, K_val), (X_test, Y_test, P_test, K_test)

def data_augmentation(config: Config, X: np.ndarray, device: str) -> np.ndarray:
    """
    Apply data augmentation to the input traces, according to the configuration.

    If data augmentation is disabled, return the original traces.
    Otherwise, a new array with augmented traces is returned.
    
    Args:
        config: Configuration object containing augmentation parameters
        X: Input traces as a numpy array, shape (N, T, 1) or (N, T). Should be already standardized (zero mean, unit variance).
    
    Returns:
        Augmented traces as a numpy array
    """
    
    if not config["data_augmentation"]:
        return X
    
    # transform to torch
    if len(X.shape) == 3:
        squeeze_last = True
        X_tensor = torch.from_numpy(X).squeeze(-1).to(device)
    else:
        squeeze_last = False
        X_tensor = torch.from_numpy(X).to(device)

    N, T = X_tensor.shape

    # apply Gaussian noise
    sigma_ratio = float(config["aug_gaussian_noise"])
    if sigma_ratio > 0:
        noise = torch.randn_like(X_tensor) * sigma_ratio
        X_tensor += noise

    # apply random shift
    K = int(config["aug_random_shift"])
    if K > 0:
        # Sample per-trace shifts
        shifts = torch.randint(-K, K + 1, (N,), device=device)
        # Vectorized roll via gather
        idx = torch.arange(T, device=device).unsqueeze(0).expand(N, T)     # (N, T)
        idx = (idx - shifts.unsqueeze(1)) % T               # (N, T)
        X_tensor = X_tensor.gather(1, idx)     
    
    # ---- return with original shape ----
    if squeeze_last:
        X_tensor = X_tensor.unsqueeze(-1)  # (N, T, 1)
    return X_tensor.to('cpu').numpy()  # convert back to numpy and return

class SCA_Dataset(Dataset):
    def __init__(self, config: Config, X: np.ndarray, Y: np.ndarray, P: np.ndarray, K: np.ndarray):
        self.config = config
        self.X = X
        self.Y = Y
        self.P = P
        self.K = K

        self.leakage_fun : Callable

        if self.config['leakage'] == 'HW':
            self.leakage_fun = calculate_HW_single

        elif self.config['leakage'] == 'ID':
            self.leakage_fun = lambda x: x

        else:
            raise ValueError("Unsupported leakage model.")
        
    def augment(self, device: str):
        '''
        According to the configuration, return a new augmented dataset or the original dataset.
        '''
        if not self.config["data_augmentation"]:
            return self
        
        else:
            X_augmented = data_augmentation(self.config, self.X, device)
            return SCA_Dataset(
                config=self.config,
                X=X_augmented,
                Y=self.Y,
                P=self.P,
                K=self.K
            )


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        trace = self.X[idx]
        sensitive = self.leakage_fun(self.Y[idx])

        trace = torch.from_numpy(trace).float()
        sensitive = torch.from_numpy(np.array(sensitive)).long()

        return trace, sensitive