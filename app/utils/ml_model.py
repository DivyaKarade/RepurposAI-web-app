import tensorflow as tf
import numpy as np
import pickle
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import AllChem

# Load model and label map once when module is imported
MODEL_PATH = 'app/target_model.h5'  # or .keras
LABEL_MAP_PATH = 'app/label_map.pkl'

MAX_COMPOUNDS = 5000  # keep this modest for hackathon; increase if you have time
TOP_N_TARGETS = 50  # keep to top N targets by counts (multi-class)
FINGERPRINT_RADIUS = 2
FINGERPRINT_NBITS = 2048
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

try:
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, 'rb') as f:
        _label_map = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    _model = None
    _label_map = {}

# converting SMILES string into a numerical fingerprint
def smiles_to_fingerprint(smiles: str, nbits:int=FINGERPRINT_NBITS, radius:int=FINGERPRINT_RADIUS) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((int(nbits),), dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def build_model(input_dim: int, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_acc")
        ]
    )
    return model


def predict_targets_from_features(model: tf.keras.Model, compound_features: np.ndarray, label_map: Dict[int, str]) -> List[Dict[str, float]]:
    # Ensure compound_features has correct shape
    if compound_features.ndim == 1:
        compound_features = np.expand_dims(compound_features, axis=0)

    probs = model.predict(compound_features)  # shape: (n_samples, n_classes)
    all_predictions = []
    for prob_vec in probs:
        pred_dict = {label_map[i]: float(prob) for i, prob in enumerate(prob_vec)}
        all_predictions.append(pred_dict)
    return all_predictions

def predict_targets(smiles: str) -> List[Dict[str, float]]:
    # Convert SMILES to fingerprint
    try:
        features = smiles_to_fingerprint(smiles)
    except ValueError as e:
        raise ValueError(f"Could not process SMILES: {e}")

    # Get predictions
    predictions = predict_targets_from_features(_model, features, _label_map)
    
    # Format output
    results = []
    for pred_dict in predictions:
        for gene, confidence in pred_dict.items():
            results.append({
                "gene": gene,
                "confidence": float(confidence),
            })
    
    # Sort by confidence
    results.sort(key=lambda x: x['confidence'], reverse=True)
    return results
