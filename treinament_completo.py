import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import optuna
from optuna.integration import OptunaSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================
print("\n1. Carregando e preparando dados...")

df = pd.read_csv('dataset_tratado.csv')
print(f"   Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

# Separar features e target
alvo = 'price'
X = df.drop(columns=[alvo, 'model', 'engine'])  # Remover colunas muito específicas
y = df[alvo]

# Features numéricas e categóricas
numeric_features = ['cc', 'hp', 'max_speed', '0_100_sec', 'seats', 'torque']
categorical_features = ['company', 'fuel_type']

print(f"   Features numéricas: {numeric_features}")
print(f"   Features categóricas: {categorical_features}")

# =============================================================================
# 2. PIPELINE DE PRÉ-PROCESSAMENTO
# =============================================================================
print("\n2. Criando pipeline de pré-processamento...")

# Pipeline para features numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para features categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("   Pipeline criado com sucesso!")

# Divisão inicial para exploração
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]} amostras | Test: {X_test.shape[0]} amostras")

# =============================================================================
# 3. SELEÇÃO DO MELHOR MODELO COM OPTUNA
# =============================================================================
print("\n3. Seleção do melhor modelo entre 3 opções usando Optuna...")
print("   Modelos: Random Forest, XGBoost, LightGBM")

# Definir modelos e espaços de busca
modelos = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "param_distributions": {
            "regressor__n_estimators": optuna.distributions.IntDistribution(100, 500),
            "regressor__max_depth": optuna.distributions.IntDistribution(5, 30),
            "regressor__min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "regressor__min_samples_leaf": optuna.distributions.IntDistribution(1, 5),
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "param_distributions": {
            "regressor__n_estimators": optuna.distributions.IntDistribution(100, 1000),
            "regressor__max_depth": optuna.distributions.IntDistribution(3, 10),
            "regressor__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "regressor__subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
            "regressor__colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
            "regressor__reg_alpha": optuna.distributions.FloatDistribution(0.0, 5.0),
            "regressor__reg_lambda": optuna.distributions.FloatDistribution(0.0, 5.0),
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
        "param_distributions": {
            "regressor__n_estimators": optuna.distributions.IntDistribution(100, 1000),
            "regressor__max_depth": optuna.distributions.IntDistribution(3, 10),
            "regressor__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "regressor__num_leaves": optuna.distributions.IntDistribution(20, 150),
            "regressor__min_child_samples": optuna.distributions.IntDistribution(5, 50),
            "regressor__subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
        }
    }
}

# Treinar e comparar modelos
resultados = []
melhores_modelos = {}

for nome, config in modelos.items():
    print(f"\n   Otimizando {nome}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', config["model"])
    ])
    
    # OptunaSearchCV
    optuna_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=config["param_distributions"],
        cv=5,
        n_trials=20,  # Número de tentativas (reduzido para velocidade)
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    optuna_search.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    score_train = optuna_search.best_score_
    score_test = optuna_search.score(X_test, y_test)
    
    melhores_modelos[nome] = optuna_search.best_estimator_
    
    resultados.append({
        "Modelo": nome,
        "R² (CV Train)": score_train,
        "R² (Test)": score_test,
        "Melhores Params": optuna_search.best_params_
    })
    
    print(f"   ✓ {nome} - R² CV: {score_train:.4f} | R² Test: {score_test:.4f}")

# Mostrar comparação
df_resultados = pd.DataFrame(resultados).sort_values(by="R² (CV Train)", ascending=False)
print("\n" + "="*70)
print("COMPARAÇÃO DOS MODELOS:")
print("="*70)
for idx, row in df_resultados.iterrows():
    print(f"{row['Modelo']:15s} | R² CV: {row['R² (CV Train)']:.4f} | R² Test: {row['R² (Test)']:.4f}")

melhor_nome = df_resultados.iloc[0]["Modelo"]
melhor_modelo = melhores_modelos[melhor_nome]

print("\n" + "="*70)
print(f"🏆 MODELO VENCEDOR: {melhor_nome}")
print("="*70)

# =============================================================================
# 4. AVALIAÇÃO FINAL COM NESTED CV
# =============================================================================
print("\n4. Avaliação final com Nested Cross-Validation...")

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

outer_scores = []
all_predictions = []
all_true_values = []

print("   Executando Nested CV (5-fold externo, 3-fold interno)...")

# Obter configuração do melhor modelo
best_config = modelos[melhor_nome]

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    
    # Criar pipeline para este fold
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', best_config["model"])
    ])
    
    # Otimização interna com Optuna
    search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=best_config["param_distributions"],
        cv=inner_cv,
        n_trials=15,  # Reduzido para velocidade
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train_fold, y_train_fold)
    
    # Avaliar no fold de teste
    y_pred_fold = search.predict(X_test_fold)
    score = r2_score(y_test_fold, y_pred_fold)
    
    outer_scores.append(score)
    all_predictions.extend(y_pred_fold)
    all_true_values.extend(y_test_fold)
    
    print(f"   Fold {fold}/5 - R²: {score:.4f}")

# Calcular estatísticas finais
mean_r2 = np.mean(outer_scores)
std_r2 = np.std(outer_scores)
ci95 = 1.96 * std_r2 / np.sqrt(5)

all_predictions = np.array(all_predictions)
all_true_values = np.array(all_true_values)

mae = mean_absolute_error(all_true_values, all_predictions)
rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
r2_final = r2_score(all_true_values, all_predictions)

print("\n" + "="*70)
print("RESULTADOS FINAIS - NESTED CROSS-VALIDATION")
print("="*70)
print(f"Modelo selecionado: {melhor_nome}")
print(f"\nR² médio (5-fold):     {mean_r2:.4f} ± {ci95:.4f} (IC 95%)")
print(f"Intervalo de confiança: [{mean_r2-ci95:.4f}, {mean_r2+ci95:.4f}]")
print(f"\nMétricas agregadas:")
print(f"  R²:   {r2_final:.4f}")
print(f"  MAE:  R$ {mae:,.2f}")
print(f"  RMSE: R$ {rmse:,.2f}")
print("="*70)

# =============================================================================
# 5. TREINO FINAL E SALVAMENTO DO MODELO
# =============================================================================
print("\n5. Treinando modelo final com 100% dos dados...")

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', best_config["model"])
])

final_search = OptunaSearchCV(
    estimator=final_pipeline,
    param_distributions=best_config["param_distributions"],
    cv=5,
    n_trials=30,  # Reduzido para velocidade
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

final_search.fit(X, y)

# Salvar modelo
model_path = os.path.join(OUTPUT_DIR, "modelo_final_carros.pkl")
joblib.dump(final_search.best_estimator_, model_path)
print(f"   ✓ Modelo salvo: {model_path}")

# =============================================================================
# 6. VISUALIZAÇÕES
# =============================================================================
print("\n6. Gerando visualizações...")

# Gráfico 1: Comparação de modelos
plt.figure(figsize=(10, 6))
models = df_resultados['Modelo'].values
scores = df_resultados['R² (CV Train)'].values
colors = ['gold' if m == melhor_nome else 'skyblue' for m in models]

plt.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('R² Score', fontsize=12, fontweight='bold')
plt.title('Comparação de Modelos - Validação Cruzada', fontsize=14, fontweight='bold')
plt.ylim([min(scores) - 0.05, 1.0])
plt.grid(axis='y', alpha=0.3)

for i, (model, score) in enumerate(zip(models, scores)):
    plt.text(i, score + 0.01, f'{score:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, 'comparacao_modelos.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Gráfico salvo: {plot1_path}")

# Gráfico 2: Predições vs Valores Reais (Nested CV)
plt.figure(figsize=(10, 6))
plt.scatter(all_true_values, all_predictions, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([all_true_values.min(), all_true_values.max()], 
         [all_true_values.min(), all_true_values.max()], 
         'r--', lw=2, label='Predição Perfeita')
plt.xlabel('Preço Real (R$)', fontsize=12, fontweight='bold')
plt.ylabel('Preço Predito (R$)', fontsize=12, fontweight='bold')
plt.title(f'Predições vs Valores Reais - {melhor_nome}\nR² = {r2_final:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, 'predicoes_vs_reais.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Gráfico salvo: {plot2_path}")

# Gráfico 3: Importância das Features (se disponível)
if hasattr(final_search.best_estimator_.named_steps['regressor'], 'feature_importances_'):
    importances = final_search.best_estimator_.named_steps['regressor'].feature_importances_
    
    # Obter nomes das features
    feature_names = numeric_features.copy()
    cat_features = final_search.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names.extend(cat_features)
    
    # Ordenar
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), importances[indices], align="center", color='teal', edgecolor='black')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importância', fontsize=12, fontweight='bold')
    plt.title(f'Top 15 Features Mais Importantes - {melhor_nome}', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, 'importancia_features.png')
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Gráfico salvo: {plot3_path}")

print("\n" + "="*70)
print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
print("="*70)
print(f"\nArquivos gerados na pasta '{OUTPUT_DIR}':")
print("  - modelo_final_carros.pkl")
print("  - comparacao_modelos.png")
print("  - predicoes_vs_reais.png")
print("  - importancia_features.png")
print("\nPara usar o modelo:")
print(f"  modelo = joblib.load('{os.path.join(OUTPUT_DIR, 'modelo_final_carros.pkl')}')")
print("  predicao = modelo.predict(X_novo)")
print("="*70)