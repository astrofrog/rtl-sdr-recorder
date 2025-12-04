import numpy as np
from rtlsdr import RtlSdr
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import os

# The central frequency
center_freq = 1420e6  # 1420 MHz

# The number of samples per second - 2.4MS/s is the max for RTL-SDR
sample_rate = 2.4e6

# How many samples to take - we just use the sample rate so that we end up
# taking 1 second worth of samples for each FFT
chunk_samples = int(sample_rate)

# The number of frequencies to sample in the FFT
fft_len = 1024

# How many spectra to compute in each waterfall file
n_samples = 60

output_dir = "raw"
os.makedirs(output_dir, exist_ok=True)

# Initialize dongle
sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.center_freq = center_freq
sdr.gain = 40.2

# Initialize empty array to store waterfall
waterfall = np.zeros((n_samples, fft_len))

# Initialize current spectrum ID
spectrum_id = 0

# Burn in
sdr.read_samples(chunk_samples)

# Initialize plot

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line = ax.plot([], [])[0]
plt.ion()
plt.show()

try:

    while True:

        # Read in raw samples
        samples = sdr.read_samples(chunk_samples)

        # Compute FFT
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples, n=fft_len)))**2
        spectrum_db = 10 * np.log10(spectrum + 1e-12)

        print(dir(line))
        line.set_xy(spectrum_db)

        # plt.draw()
        # plt.pause(0.001)

        # # Add spectrum to waterfall
        # waterfall[spectrum_id % n_samples] = spectrum_db

        # # Update plot
        # image.set_data(waterfall)
        # plt.draw()
        # plt.pause(0.001)

        # # Write out raw data if needed
        # spectrum_id += 1
        # if spectrum_id % n_samples == 0:
        #     print(f'Writing to raw/waterfall_{spectrum_id:08d}.fits...')
        #     fits.writeto(f'raw/waterfall_{spectrum_id:08d}.fits', waterfall, overwrite=True)
        #     waterfall[:] = 0

except KeyboardInterrupt:
    print("Stopping recording data...")
finally:
    sdr.close()
