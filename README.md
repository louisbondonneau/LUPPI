# README for LUPPI (Low-frequency Ultimate Pulsar Processing Instrument)

## Overview
`luppi_daq_dedisp_GPU1` is the main executable component of LUPPI (Low-frequency Ultimate Pulsar Processing Instrument). It is specifically designed for handling UDP waveform packets from the NenuFAR instrument, which are formatted in the LOFAR style. The core functionality of `luppi_daq_dedisp_GPU1` includes loading these packets into a circular buffer, processing them through a GPU for fast Fourier transforms (FFT), and applying coherent dedispersion and Faraday rotation corrections. After inverse FFT (iFFT) to return to the time domain, the data is folded at the pulsar's period (with an option to downsample instead of fold) and written to disk in the PSRFITS format. Additionally, `luppi_daq_dedisp_GPU1` prepares scripts for uploading observation files to the databf storage server.

## Installation
Before installing `luppi_daq_dedisp_GPU1`, ensure you have CUDA libraries and tools for PSRFITS format handling.

1. Clone the repository (provide the repository link).
2. In the Makefile, specify the location of Presto libraries.
3. Adjust the CUDA architecture flags (`arch=compute_xx,code=sm_xx`) in the Makefile to match your GPU. For instance, for NVIDIA Pascal architecture, use `arch=compute_61,code=sm_61`.
4. Compile the source with a C compiler, ensuring dependency paths are correctly set.
5. Install the executable in your desired directory.

## Usage
Execute `luppi_daq_dedisp_GPU1` with the following command:

```
./luppi_daq_dedisp_GPU1 [options]
```

### Options
- `-h, --help`: Show help message.
- `-n, --null`: No disk output.
- `-D, --ds`: Downsample instead of fold.
- `-g, --gpu <GPUid>`: GPU ID (default 0).
- `-j, --databfdirname <dirname>`: Subdirectory on databf (optional).

### Example
```
./luppi_daq_dedisp_GPU1 --gpu 1 --databfdirname pulsar_data
```

## Threads and Data Flow

### Main Threads:

1. **Network Thread (`net_thread`):** Handles incoming UDP packets, loading data into the circular buffer.

2. **Dedispersion Thread (`dedisp_thread`/`dedisp_ds_thread`):** Transfers data to the GPU, applies FFT, coherent dedispersion, and inverse FFT. Optionally, it can downsample the data.

3. **PSRFITS Thread (`psrfits_thread`):** Folds the data at the pulsar's period and writes it to disk in PSRFITS format. Alternatively, the `null_thread` can be used for testing purposes without disk output.

### Data Flow:

- Data enters through the network thread via UDP, is processed in the GPU by the dedispersion thread, and finally written to disk according to the specified mode.


## Contributing
To contribute:

1. Fork and branch the repository.
2. Make and test your changes.
3. Submit a pull request with detailed changes.