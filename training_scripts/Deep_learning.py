# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 02:38:23 2025

@author: Pradnya

"""

# ----------------------------Imports -----------------------------------------------
import os
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler #StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import  Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

output_dir = 'path_to_output'

mordred_data = pd.read_excel('path_to_data', engine='openpyxl')
mordred_data = mordred_data.drop(columns=mordred_data.columns[0], axis=1)
X = mordred_data.drop(columns=['Label'], axis=1).apply(pd.to_numeric, errors='coerce')
Y = mordred_data[['Label']].replace({'Inhibitor':1, 'NonInhibitor':0})




# ------------------------ GPU Configuration ------------------------
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print("GPUs available:", physical_devices)


# ------------------------ Train-Test Split & Normalization ------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y) #, random_state=54
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------ Prepare Data for Training ------------------------
X = X_train_scaled
y = Y_train.values.ravel()
X_test = X_test_scaled
y_test = Y_test.values.ravel()

# ------------------------ Hyperparameters ------------------------
learning_rate = 0.00002
epochs = 500
batch_size = 64
k_folds = 10

# ---------------- K-Fold Cross Validation ----------------
kf = StratifiedKFold(n_splits=k_folds, shuffle=True) #, random_state=28

best_val_acc = 0
best_model = None
best_history = None
best_cv_metrics = None
fold_metrics = []

# ---------------- Callbacks ----------------
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, min_lr=1e-6)
# ---------------- Model Training ----------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n---------Training fold {fold + 1}---------")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    model = Sequential([
        Dense(250, kernel_initializer='he_normal', kernel_regularizer=l2(0.001), input_shape=(X.shape[1],)),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.4),

        Dense(50, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.3),

        Dense(5, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.01),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=epochs, batch_size=batch_size,
        verbose=1
        # callbacks=[early_stop, reduce_lr]
    )

    y_val_pred_prob = model.predict(X_val_fold).ravel()
    y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val_fold, y_val_pred)
    prec = precision_score(y_val_fold, y_val_pred, zero_division=0)
    rec = recall_score(y_val_fold, y_val_pred, zero_division=0)
    f1 = f1_score(y_val_fold, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val_fold, y_val_pred_prob)
    pr_auc = average_precision_score(y_val_fold, y_val_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_val_fold, y_val_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    fold_metrics.append([acc, prec, rec, f1, auc, pr_auc, sensitivity, specificity])

    # Store validation predictions of best fold
    if acc > best_val_acc:
        y_val_class_best = y_val_fold  # ground truth
        val_pred_class_prob = y_val_pred_prob  # predicted probabilities
        
    # Save the model with the best validation accuracy
    if acc > best_val_acc:
        best_val_acc = acc
        best_model = model
        best_history = history
        best_cv_metrics = [acc, prec, rec, f1, auc, pr_auc, sensitivity, specificity]

# ---------------- Cross-Validation Metrics ----------------
fold_metrics = np.array(fold_metrics)
accuracy_list = fold_metrics[:, 0]
avg_metrics = np.mean(fold_metrics, axis=0)
cv_ci_low, cv_ci_up = stats.t.interval(
    0.95, len(accuracy_list) - 1,
    loc=np.mean(accuracy_list),
    scale=stats.sem(accuracy_list)
)

# ---------------- Save Model History ----------------
history_df = pd.DataFrame(best_history.history)


# ---------------- Plot Accuracy and Loss ----------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(best_history.history['accuracy'], label='Train Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plots", "mordred_ann_Accuracy_Loss_Plot.png"))
plt.close()
























