import json
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# 1. Define high-quality dummy training dataset
training_data = [
    # Positive reviews/sentences
    ("This is an absolutely amazing product, highly recommended!", "positive"),
    ("I love this so much! It works perfectly and is very fast.", "positive"),
    ("Superb quality and excellent customer service. Five stars!", "positive"),
    ("Incredible performance and very easy to set up and use.", "positive"),
    ("Great design, extremely useful, and worth every single penny.", "positive"),
    ("Best purchase I have made this year. Absolutely fantastic.", "positive"),
    ("Truly outstanding work, very pleased with the fast execution.", "positive"),
    ("The user interface is beautiful, clean, and very responsive.", "positive"),
    ("Perfect solution for my needs. Fast, reliable, and lightweight.", "positive"),
    ("Very happy with this product. Exceeded my high expectations.", "positive"),
    
    # Negative reviews/sentences
    ("This is terrible. It broke after just one day of usage.", "negative"),
    ("I hate it. It is extremely slow, laggy, and useless.", "negative"),
    ("Horrible customer support and poor product build quality.", "negative"),
    ("Do not buy this! It is a complete waste of time and money.", "negative"),
    ("Very disappointed. It does not work as advertised at all.", "negative"),
    ("Extremely complicated to set up and constantly crashes.", "negative"),
    ("Bad design, low performance, and very frustrating experience.", "negative"),
    ("The worst purchase ever. It is full of bugs and errors.", "negative"),
    ("Slow customer response and defective hardware. Very unhappy.", "negative"),
    ("Fails to load, crashes on startup, and is totally broken.", "negative"),
]

def main():
    print("Training Sentiment Analysis MLP Model...")
    
    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]
    
    # Convert labels to integers: negative=0, positive=1
    label_map = {"negative": 0, "positive": 1}
    y = np.array([label_map[l] for l in labels])
    
    # 2. Vectorize texts using TF-IDF (sublinear_tf matches our custom preprocessor)
    vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2')
    X = vectorizer.fit_transform(texts).toarray()
    
    # Get vocab and idf values
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_.tolist()
    
    # Ensure vocab keys are standard python strings (not numpy objects)
    vocab = {str(k): int(v) for k, v in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # 3. Train MLP Classifier
    # 1 hidden layer with 8 units, relu activation, L-BFGS solver for small datasets
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,), 
        activation='relu', 
        solver='lbfgs', 
        random_state=42, 
        max_iter=500
    )
    mlp.fit(X, y)
    
    # Evaluate training accuracy
    train_acc = mlp.score(X, y)
    print(f"Training accuracy: {train_acc * 100:.1f}%")
    
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    # 4. Save scikit-learn model as a pickle file (for SklearnBackend demo)
    sklearn_model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
    with open(sklearn_model_path, "wb") as f:
        pickle.dump(mlp, f)
    print(f"Saved Scikit-learn model to: {sklearn_model_path}")
    
    # 5. Extract weights & biases and dump to JSON for NumPy-only backend
    # coefs_[0] is W1 (input_dim -> hidden_dim)
    # intercepts_[0] is b1 (hidden_dim)
    # coefs_[1] is W2 (hidden_dim -> output_dim)
    # intercepts_[1] is b2 (output_dim)
    #
    # Note: Binary classification in MLPClassifier still outputs output layer sizes matching classes
    # if we have 2 classes, outputs shape might be (hidden_dim, 1) or (hidden_dim, 2).
    # Since we want to use softmax, we can expand W2 and b2 to represent both class 0 and class 1 probabilities
    # if mlp.n_outputs_ is 1. Let's handle both.
    
    W1 = mlp.coefs_[0].tolist()
    b1 = mlp.intercepts_[0].tolist()
    
    if mlp.n_outputs_ == 1:
        # If single output node, it outputs sigmoid logit for class 1.
        # To turn it into 2 classes for softmax, we can formulate:
        # class_0_logit = 0, class_1_logit = z
        # which yields probabilities matching sigmoid.
        # W2 shape: (8, 2), b2 shape: (2,)
        w2_coef = mlp.coefs_[1] # shape (8, 1)
        b2_intercept = mlp.intercepts_[1] # shape (1,)
        
        W2 = np.zeros((w2_coef.shape[0], 2))
        W2[:, 1] = w2_coef[:, 0]
        
        b2 = np.zeros(2)
        b2[1] = b2_intercept[0]
        
        W2 = W2.tolist()
        b2 = b2.tolist()
    else:
        W2 = mlp.coefs_[1].tolist()
        b2 = mlp.intercepts_[1].tolist()
        
    model_json = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "vocab": vocab,
        "idf": idf,
        "labels": ["negative", "positive"]
    }
    
    json_model_path = os.path.join(os.path.dirname(__file__), "sentiment_mlp.json")
    with open(json_model_path, "w") as f:
        json.dump(model_json, f, indent=2)
        
    print(f"Saved NumPy-compatible model parameters to: {json_model_path}")
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
