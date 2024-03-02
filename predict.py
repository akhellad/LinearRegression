import numpy as np
from train import model


theta_final = np.load('finalTheta.npy')
mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')

x_new = float(input("Entrez le kilométrage de la voiture: "))
x_new_normalized = (x_new - mean_x) / std_x  # Utiliser mean_x et std_x du dataset original
x_new_array = np.array([x_new_normalized])
X_new = np.hstack([x_new_array, np.ones(x_new_array.shape)])

X_new = X_new.reshape(1, -1)

# Prédiction
prediction = model(X_new, theta_final)
print(f"Prédiction du prix: {prediction[0][0]}")