# Estimating the cognitive load in physical spatial navigation

[T. -T. N. Do, A. K. Singh, C. A. T. Cortes and C. -T. Lin, "Estimating the cognitive load in physical spatial navigation," 2020 IEEE Symposium Series on Computational Intelligence (SSCI), Canberra, Australia, 2020, pp. 568-575, doi: 10.1109/SSCI47803.2020.9308389.](https://ieeexplore.ieee.org/abstract/document/9308389)


# Requirements

- Python == 3.7 or 3.8
- tensorflow == 2.X (both for CPU and GPU)
- PyRiemann >= 0.2.5
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

# How to run

- Input Data Format: Number of EEG Channels x Number of Samples X Number of Trials for EEG data and Labels as vector. See [testData.mat](https://github.com/thinknew/BCINet/tree/main/testdata) for references with sampling rate of 400 Hz.
- Provide input data related information in 'op.py' such as path, sampling rate, number of classes, etc.
- Execute the following line of code

```
python main.py
```

# Models implemented/used
- DeepConvNet [[2]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)

DeepConvNet is based on repo [[2]](https://github.com/thinknew/BCINet)
