# improving TextCNN Model for IMDB sentiment analysis

This repository contains an improved **TextCNN** model for sentiment analysis on the **IMDB dataset**. The model classifies movie reviews as **positive (0)** or **negative (1)**. The key improvements resulted in a **test accuracy of over 87%**, making the model more effective for binary sentiment classification.

## 📂 Dataset & Preprocessing

The **IMDB dataset** (`IMDB Dataset.csv`) contains **50,000** movie reviews labeled as **positive** or **negative**. The dataset was **split into 45,000 training samples and 5,000 test samples**:

```python
df = pd.read_csv('IMDB Dataset.csv')
df_train, df_test = np.split(df, [45000], axis=0)
To improve model efficiency, a custom vocabulary was built by filtering low-frequency words (min_freq=5), and pre-trained GloVe embeddings (glove-twitter-25) were used for initialization.

🏗 Model Improvements

The TextCNN architecture was enhanced with several modifications:

- Pre-trained Word Embeddings: Initialized with GloVe Twitter-25, replacing randomly initialized embeddings.
- Expanded Convolutional Features: Increased Conv2D out_channels from 16 to 64 to capture richer patterns.
- Batch Normalization: Applied after convolutional layers to stabilize learning.
- Higher Dropout (0.5): Improved generalization and reduced overfitting.
- Optimized Pooling Strategy: Applied MaxPooling across multiple kernel sizes [2,3,4,5] for better representation.

⚙️ Training Details

- Loss Function: BCEWithLogitsLoss()
- Optimizer: Adam(lr=1e-3, weight_decay=1e-4)
- Evaluation Metric: Binary Accuracy
- Training: 7 epochs on the IMDB dataset.

📊 Results

- Train Accuracy: ~90%
- Test Accuracy: 87%+
- Loss Reduction: Achieved stability with batch normalization.
- Training Loss & Accuracy Plot
