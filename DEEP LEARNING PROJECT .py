#import libraries 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

df=pd.read_csv('Prodigy University Dataset.csv')
df

X = df[['sat_sum', 'hs_gpa']]
y = df['fy_gpa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)
print(f"Linear Regression - MSE: {lr_mse:.4f}, R²: {lr_r2:.4f}")
print(f"Coefficients: SAT_sum = {lr_model.coef_[0]:.4f}, HS_GPA = {lr_model.coef_[1]:.4f}")
print(f"Intercept: {lr_model.intercept_:.4f}")

X_train_scaled = (X_train - X_train.mean()) / X_train.std()
X_test_scaled = (X_test - X_train.mean()) / X_train.std()

def build_ann(optimizer):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),  # Hidden layer with 16 neurons
        Dense(8, activation='relu'),                    # Hidden layer with 8 neurons
        Dense(1)                                        # Output layer (no activation for regression)
    ])
    model.compile(optimizer=optimizer, loss='mse')      # Mean squared error as loss
    return model

gradient_descents = {
    'Batch': {'batch_size': len(X_train_scaled), 'optimizer': SGD(learning_rate=0.01), 'desc': 'Batch Gradient Descent'},
    'SGD': {'batch_size': 1, 'optimizer': SGD(learning_rate=0.01), 'desc': 'Stochastic Gradient Descent'},
    'Mini-Batch': {'batch_size': 32, 'optimizer': SGD(learning_rate=0.01), 'desc': 'Mini-Batch Gradient Descent'},
    'Adam': {'batch_size': 32, 'optimizer': Adam(learning_rate=0.001), 'desc': 'Adam Optimizer'}
}

results = {}
histories = {}

for gd_type, config in gradient_descents.items():
    print(f"\nTraining with {config['desc']}...")

model = build_ann(config['optimizer'])

history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=config['batch_size'],
        validation_split=0.2,
        verbose=0  # Silent training
    )


y_pred_ann = model.predict(X_test_scaled, verbose=0)
    
mse = mean_squared_error(y_test, y_pred_ann)
r2 = r2_score(y_test, y_pred_ann)
results[gd_type] = {'MSE': mse, 'R²': r2}
histories[gd_type] = history.history
    
print(f"{config['desc']} - MSE: {mse:.4f}, R²: {r2:.4f}")

ann_model = build_ann(Adam(learning_rate=0.001))
history = ann_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                       validation_split=0.2, verbose=0)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ANN Loss Over Epochs (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()





