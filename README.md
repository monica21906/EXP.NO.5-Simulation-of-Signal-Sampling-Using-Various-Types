# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM: 
```
To simulate and verify the signal sampling using various types.

# SOFTWARE REQUIRED
```
colab
```

# ALGORITHMS
```
Create the Continuous Signal:

Set the sampling frequency fs = 100 Hz and signal frequency f = 5 Hz.

Generate time values from 0 to 1 second using fs.

Create a sine wave using the time values.

Plot the Continuous Signal:

Plot the sine wave over time.

Sample the Signal:

Use the same time values (t_sampled) as the continuous signal.

Sample the sine wave at these time points.

Plot the Continuous and Sampled Signals:

Plot both the continuous signal and sampled signal together.

Highlight the sampled points with red circles.

Reconstruct the Sampled Signal:

Use a resampling function to reconstruct the signal from the sampled points.

Plot the Continuous and Reconstructed Signals:

Plot both the continuous signal and the reconstructed signal.

Show the reconstructed signal as a dashed red line.
```

# PROGRAM
```
Program :
Impulse Sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs)

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)  # Define width of each pulse
for i in range(0, len(t), int(fs / pulse_rate)):
    pulse_train[i:i + pulse_width] = 1
nat_signal = message_signal * pulse_train
sampled_signal = nat_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))  # Find the closest index to the sample time
    reconstructed_signal[index] = sampled_signal[i]
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal, 10, fs)
plt.figure(figsize=(14, 10))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
def platop_sampling(probabilities, platop=0.9):
    """
    Platop Sampling: A modified nucleus sampling approach.
    :param probabilities: List or numpy array of probabilities for each token.
    :param platop: The cumulative probability threshold for nucleus sampling.
    :return: Index of the sampled token.
    """
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort indices by probability (descending order)
    sorted_probs = probabilities[sorted_indices]  # Sort probabilities accordingly
    
    cumulative_probs = np.cumsum(sorted_probs)  # Compute cumulative probabilities
    cutoff_index = np.searchsorted(cumulative_probs, platop) + 1  # Find the cutoff index
    
    # Restrict to the nucleus of tokens
    nucleus_indices = sorted_indices[:cutoff_index]
    nucleus_probs = sorted_probs[:cutoff_index]
    nucleus_probs /= nucleus_probs.sum()  # Normalize probabilities
    
    # Sample from the nucleus
    sampled_index = np.random.choice(nucleus_indices, p=nucleus_probs)
    return sampled_index
fs = 100  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
f = 5  # Frequency of the sine wave
signal = np.sin(2 * np.pi * f * t)  # Generate sine wave

plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
probs = np.abs(signal) / np.sum(np.abs(signal))  # Normalize probabilities
t_sampled_indices = [platop_sampling(probs) for _ in range(len(t)//2)]  # Select indices
signal_sampled = signal[t_sampled_indices]  # Sampled signal values
t_sampled = t[t_sampled_indices]  # Corresponding time values
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Platop Sampled Signal')
plt.title('Platop Sampling of Continuous Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Original Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal')
plt.title('Reconstruction of Platop Sampled Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```
## OUTPUT
```
![Screenshot 2025-04-15 122808](https://github.com/user-attachments/assets/51daaf13-79d2-47c9-a88c-f450d3a93d6f)
```
```
![Screenshot 2025-04-15 122817](https://github.com/user-attachments/assets/b383e679-eade-4709-bcfa-3b013eeac5e2)
```
```
![Screenshot 2025-04-02 100718](https://github.com/user-attachments/assets/5b5ebc22-e4b9-419c-95ee-9c2478a5efa7)
![Screenshot 2025-04-02 100727](https://github.com/user-attachments/assets/5ace2319-cd16-4398-9549-22ed9385d0f0)
```
## RESULT / CONCLUSIONS
```
Thus the signal sampling using various types were verified and simulated successfully.
```


