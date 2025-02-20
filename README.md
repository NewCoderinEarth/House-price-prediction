# House-price-prediction

## Overview
This project combines Machine Learning (Lasso Regression) with a web interface to predict house prices.

## Features
- Machine Learning model using Lasso Regression
- Interactive data visualization
- Feature importance analysis
- Model performance metrics

###Running the Code

Google Colab (Recommended)
This code is designed to run in Google Colab. To use it:

1. Open [Google Colab](https://colab.research.google.com)

Or
#### Required Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

