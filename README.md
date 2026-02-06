# PCA Implementation - Advanced Linear Algebra

## Dataset
African health and economic data (500 samples, 16 features)
- **Missing values**: Yes (75 per numeric column)
- **Non-numeric columns**: Country, Region (encoded)
- **Source**: Synthetically generated for educational purposes

## Implementation Details
- **Manual standardization**: NumPy only (no sklearn for Step 1)
- **Eigenvalue decomposition**: Using `np.linalg.eig()`
- **Sorting**: DESCENDING order by explained variance
- **Component selection**: 95% variance threshold
- **Validation**: Perfect match with sklearn (0.000000% difference)

## Files
- **`PCA_Implementation.ipynb`** - Main Colab notebook with full implementation
- **`african_health_economic_data.csv`** - Dataset
- **`generate_sample_data.py`** - Script to regenerate the dataset
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## Results Summary
- **Original dimensions**: 16 features
- **Reduced dimensions**: 15 components (for 95% variance)
- **Variance retained**: 95.17%
- **Implementation accuracy**: 100% match with sklearn

## Quick Start
1. Open the Colab notebook: [PCA_Implementation.ipynb](https://drive.google.com/file/d/1emt0c2fp7_czi9CkmaQ_jBysj1_8fmTG/view?usp=sharing)
2. Upload `african_health_economic_data.csv` when prompted
3. Run all cells sequentially

## Dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
pca-assignment/
├── .gitignore                          # Git ignore file
├── PCA_Implementation.ipynb            # Main Colab notebook with PCA implementation
├── README.md                           # Project documentation
├── african_health_economic_data.csv    # Dataset (500 samples, 16 features)
├── generate_sample_data.py             # Script to regenerate the dataset
└── requirements.txt                    # Python dependencies
```

## Assignment Tasks

### Task 1: Implement PCA from Scratch
- Load and preprocess data (handle missing values and non-numeric columns)
- Calculate the covariance matrix
- Perform eigendecomposition
- Sort eigenvalues and eigenvectors
- Project data onto principal components

### Task 2: Dynamic Component Selection
- Calculate explained variance ratio for each principal component
- Implement automatic selection based on cumulative variance threshold
- Visualize explained variance

### Task 3: Performance Optimization
- Optimize matrix operations for large datasets
- Implement efficient memory management
- Benchmark against scikit-learn's PCA implementation

## Data Requirements
- Must contain missing values (NaN)
- Must have at least 1 non-numeric column
- Must have more than 10 columns
- Should be impactful African/Africanized data
- Avoid generic datasets (house prices, wine quality, etc.)

## Key Concepts Covered

### Linear Algebra
- **Covariance Matrix**: Measures how features vary together
- **Eigenvalues**: Represent the variance explained by each principal component
- **Eigenvectors**: Define the direction of principal components
- **Matrix Decomposition**: Breaking down covariance into eigenvalues/eigenvectors

### Statistical Concepts
- **Variance**: Measure of data spread
- **Standardization**: Scaling features to have mean=0, std=1
- **Explained Variance Ratio**: Proportion of variance explained by each PC

## Usage Example

```python
# Load your data
import pandas as pd
data = pd.read_csv('your_data.csv')

# The notebook will guide you through:
# 1. Data preprocessing
# 2. PCA implementation
# 3. Visualization
# 4. Analysis
```

## Expected Outputs
1. **Data preprocessing results**: Clean dataset with encoded features
2. **Eigenvalue table**: Sorted eigenvalues with explained variance
3. **Scree plot**: Visualization of explained variance per component
4. **Before/After PCA plots**: Original vs transformed feature space
5. **Performance metrics**: Execution time and memory usage

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **Memory errors**: Use smaller dataset or implement batch processing
3. **Singular matrix**: Check for constant or highly correlated features

### Tips
- Always standardize your data before PCA
- Check for multicollinearity in your features
- Visualize data at each preprocessing step
- Compare your implementation with scikit-learn for validation

## Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [PCA Mathematical Foundation](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)


