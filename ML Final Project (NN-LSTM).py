import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding, LSTM, Bidirectional, MultiHeadAttention, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from textblob import TextBlob

# Utility Functions
def preprocess_text(text):
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def preprocess_columns(df, text_cols):
    """Apply preprocessing sequentially to text columns."""
    for col in text_cols:
        df[col] = df[col].apply(preprocess_text)
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

def add_features(df):
    """Add engineered features."""
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))
    df['average_word_length'] = df['char_count'] / (df['word_count'] + 1e-5)
    df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Load your dataset
train_data = pd.read_csv("C:\\Users\\savag\\.vscode\\424_F2024_Final_PC_large_train_v1.csv")
test_data = pd.read_csv("C:\\Users\\savag\\.vscode\\424_F2024_Final_PC_test_without_response_v1.csv")

# Preprocess text columns
text_columns = ['headline', 'pros', 'cons']
preprocess_columns(train_data, text_columns)
preprocess_columns(test_data, text_columns)

# Add features
add_features(train_data)
add_features(test_data)

# Extract target and text for modeling
X_train_text = train_data['text']
y_train = train_data['rating']
X_test_text = test_data['text']

# --- TF-IDF Implementation ---
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# --- LDA Topic Modeling ---
lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_train_lda = lda.fit_transform(X_train_tfidf)
X_test_lda = lda.transform(X_test_tfidf)

# --- GloVe Embedding Implementation ---
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

glove_file_path = "C:\\Users\\savag\\Downloads\\glove.6B\\glove.6B.300d.txt"  
embeddings_index = load_glove_embeddings(glove_file_path)

# Tokenize and pad sequences for GloVe
MAX_LEN = 120
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_text)
X_train_sequences = tokenizer.texts_to_sequences(X_train_text)
X_test_sequences = tokenizer.texts_to_sequences(X_test_text)

X_train_padded = pad_sequences(X_train_sequences, maxlen=MAX_LEN)
X_test_padded = pad_sequences(X_test_sequences, maxlen=MAX_LEN)

# Create embedding matrix
word_index = tokenizer.word_index
embedding_dim = 300  
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if i >= len(word_index) + 1:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# --- Hybrid Neural Network Model with Multi-Head Attention ---
def create_hybrid_model(input_length, embedding_matrix, lda_features, dropout_rate=0.3):
    glove_input = Input(shape=(input_length,))
    lda_input = Input(shape=(lda_features,))
    
    # GloVe Embedding and Attention
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False
    )(glove_input)
    
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
    attention_layer = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_layer, lstm_layer)
    pooling_layer = GlobalAveragePooling1D()(attention_layer)
    glove_output = Dense(64, activation='relu')(pooling_layer)  # Match LDA dimensions
    glove_output = Dropout(dropout_rate)(glove_output)
    
    # LDA Feature Processing
    lda_dense = Dense(64, activation='relu')(lda_input)  # Match GloVe dimensions
    lda_dense = Dropout(dropout_rate)(lda_dense)
    
    # Combine outputs
    combined = Concatenate()([glove_output, lda_dense])
    x = Dense(128, activation='relu')(combined)
    x = Dropout(dropout_rate)(x)
    
    output_layer = Dense(1, activation='linear')(x)

    model = Model(inputs=[glove_input, lda_input], outputs=output_layer)
    return model

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
KFOLDS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
validation_scores = []

fold_no = 1
for train_index, val_index in kf.split(X_train_padded, y_train):
    print(f"\nTraining Fold {fold_no}/{KFOLDS}")

    # Split data
    X_train_fold = [X_train_padded[train_index], X_train_lda[train_index]]
    X_val_fold = [X_train_padded[val_index], X_train_lda[val_index]]
    y_train_fold = y_train.iloc[train_index]
    y_val_fold = y_train.iloc[val_index]

    # Create and compile the hybrid model
    hybrid_model = create_hybrid_model(MAX_LEN, embedding_matrix, X_train_lda.shape[1])
    
    # Reinitialize optimizer
    adamw_optimizer = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    hybrid_model.compile(optimizer=adamw_optimizer, loss='mse', metrics=['mae'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

    # Train the model
    start_time = time.time()
    history = hybrid_model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    print(f"Training for Fold {fold_no} completed in {time.time() - start_time:.2f} seconds.")

    # Evaluate
    val_loss, val_mae = hybrid_model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
    validation_scores.append(val_loss)
    fold_no += 1

# Final evaluation
print(f"\nAverage Validation Loss: {np.mean(validation_scores):.4f}, Std Dev: {np.std(validation_scores):.4f}")

# Predict on test data
predictions = hybrid_model.predict([X_test_padded, X_test_lda])
test_data['rating_prediction'] = predictions
test_data.to_csv('test_predictions_hybrid.csv', index=False)
print("Predictions saved to 'test_predictions_hybrid.csv'")
