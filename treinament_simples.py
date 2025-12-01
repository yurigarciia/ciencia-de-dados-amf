import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset tratado
df = pd.read_csv('dataset_tratado.csv')

# Colunas numéricas do dataset tratado (já estão limpas)
num_cols = ['cc', 'hp', 'max_speed', '0_100_sec', 'seats', 'torque']

# Separar features e target
X = df[num_cols].copy()
y = df['price']

# Remover linhas com valores nulos
X = X.dropna()
y = y[X.index]

# Normalizar as features (escalar valores entre 0 e 1)
scaler = MinMaxScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Dividir dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo Random Forest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o desempenho do modelo
print("MSE:", mean_squared_error(y_test, y_pred))  # Erro quadrático médio
print("R2:", r2_score(y_test, y_pred))  # Coeficiente de determinação (quanto melhor, mais próximo de 1)

# Extrair importância das features
importances = model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]  # Ordenar do maior para o menor

# Plotar gráfico de importância das features
plt.figure(figsize=(12,6))
plt.title("Importância das Features")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.show()