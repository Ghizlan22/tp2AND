import numpy as np
import pandas as pd

# Création du DataFrame avec des valeurs manquantes
data = {
    'A': [80.0, 44.0, np.nan, 50.0, 29.0],
    'B': [30.0, np.nan, 85.0, 70.0, 54.0],
    'C': [7, 10, 25, 74, 49],
    'D': [14, 0, 5, 9, 20],
    'E': [27.0, 29.0, 88.0, 49.0, np.nan]
}
df = pd.DataFrame(data)

print("DataFrame initial :")
print(df)

# Fonction pour calculer la distance entre deux lignes (en ignorant les NaN)
def calculate_distance(row1, row2, columns_with_values):
    valid_columns = [col for col in columns_with_values if not np.isnan(row1[col]) and not np.isnan(row2[col])]
    if len(valid_columns) == 0:  # Si aucune colonne n'est valide, retourner une distance infinie
        return np.inf
    num_features = len(valid_columns)
    scaling_factor = (len(columns_with_values) - 1) / num_features
    distance = np.sqrt(scaling_factor * sum((row1[col] - row2[col]) ** 2 for col in valid_columns))
    return distance

# Fonction pour imputer les valeurs manquantes en utilisant la méthode kNN
def impute_missing_values_knn(df, k=2):
    columns = df.columns
    for index, row in df.iterrows():
        for col in columns:
            if np.isnan(row[col]):  # Si la valeur est manquante
                # Calculer la distance avec les autres lignes
                distances = []
                for other_index, other_row in df.iterrows():
                    if other_index != index:  # Éviter de calculer la distance avec soi-même
                        distance = calculate_distance(row, other_row, columns)
                        distances.append((distance, other_index))

                # Trier les distances et choisir les K plus proches voisins
                distances.sort()
                nearest_neighbors = distances[:k]

                # Calculer la moyenne des valeurs des voisins pour cette colonne
                neighbor_values = [df.loc[neighbor[1], col] for neighbor in nearest_neighbors if not np.isnan(df.loc[neighbor[1], col])]
                if neighbor_values:
                    df.loc[index, col] = np.mean(neighbor_values)

    return df

# Appliquer la méthode kNN au DataFrame
df_imputed = impute_missing_values_knn(df.copy())

print("\nDataFrame après imputation avec kNN :")
print(df_imputed)
