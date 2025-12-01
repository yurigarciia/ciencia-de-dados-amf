# 📚 Explicação Detalhada do Código - Análise de Preços de Carros

## 🎯 Visão Geral do Projeto

Este código implementa um pipeline completo de Machine Learning para **prever preços de carros** usando três modelos diferentes e selecionando o melhor através de otimização de hiperparâmetros com **Optuna**.

---

## 📦 Importações e Configurações Iniciais

### Bibliotecas Básicas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

- **pandas**: Manipulação de dados em formato tabular (DataFrames)
- **numpy**: Operações matemáticas e arrays numéricos
- **matplotlib.pyplot**: Criação de gráficos e visualizações
- **warnings.filterwarnings("ignore")**: Desativa avisos para deixar a saída mais limpa

### Bibliotecas de Machine Learning (sklearn)
```python
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
```

- **train_test_split**: Divide dados em treino e teste
- **KFold**: Implementa validação cruzada (divide dados em K partes)
- **RandomForestRegressor**: Modelo de árvores de decisão aleatórias
- **StandardScaler**: Normaliza dados (média 0, desvio padrão 1)
- **OneHotEncoder**: Converte variáveis categóricas em números binários
- **ColumnTransformer**: Aplica transformações diferentes para colunas diferentes
- **Pipeline**: Encadeia transformações e modelo em sequência
- **SimpleImputer**: Preenche valores faltantes (missing values)
- **Métricas**: MSE (erro quadrático), R² (qualidade do modelo), MAE (erro absoluto)

### Otimização e Modelos Avançados
```python
import optuna
from optuna.integration import OptunaSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
```

- **optuna**: Biblioteca de otimização bayesiana de hiperparâmetros
- **OptunaSearchCV**: Similar ao GridSearchCV, mas usa Optuna para buscar hiperparâmetros
- **XGBRegressor**: Gradient Boosting extremamente eficiente
- **LGBMRegressor**: Light Gradient Boosting, rápido e preciso

### Utilidades
```python
import joblib
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

- **joblib**: Salva e carrega modelos treinados de forma eficiente
- **os**: Operações com sistema de arquivos
- **os.makedirs()**: Cria a pasta "output" se não existir

---

## 1️⃣ CARREGAMENTO E PREPARAÇÃO DOS DADOS

### Carregar Dataset
```python
df = pd.read_csv('dataset_tratado.csv')
print(f"   Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
```
- Lê o arquivo CSV já tratado
- `.shape[0]` = número de linhas (carros)
- `.shape[1]` = número de colunas (features)

### Separar Features e Target
```python
alvo = 'price'
X = df.drop(columns=[alvo, 'model', 'engine'])
y = df[alvo]
```
- **X**: Features (características dos carros) - remove 'price', 'model' e 'engine'
  - 'price': é o que queremos prever (target)
  - 'model' e 'engine': muito específicos, podem causar overfitting
- **y**: Target (variável alvo) - os preços que queremos prever

### Definir Tipos de Features
```python
numeric_features = ['cc', 'hp', 'max_speed', '0_100_sec', 'seats', 'torque']
categorical_features = ['company', 'fuel_type']
```
- **Numéricas**: valores contínuos (cilindradas, potência, velocidade, etc.)
- **Categóricas**: valores discretos (marca do carro, tipo de combustível)

**Por que separar?** Cada tipo precisa de pré-processamento diferente!

---

## 2️⃣ PIPELINE DE PRÉ-PROCESSAMENTO

### Pipeline para Features Numéricas
```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

**Passo a passo:**
1. **SimpleImputer(strategy='median')**: Se houver valores faltantes, preenche com a mediana
2. **StandardScaler()**: Normaliza os dados
   - Fórmula: `(x - média) / desvio_padrão`
   - Resultado: média = 0, desvio = 1
   - **Por que?** Alguns algoritmos são sensíveis à escala dos dados

### Pipeline para Features Categóricas
```python
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

**Passo a passo:**
1. **SimpleImputer(strategy='most_frequent')**: Preenche faltantes com o valor mais comum
2. **OneHotEncoder(handle_unknown='ignore')**: Converte categorias em números binários
   - Exemplo: `fuel_type = ['Petrol', 'Diesel']` → `[1,0]` ou `[0,1]`
   - `handle_unknown='ignore'`: Se aparecer categoria nova no teste, ignora

### ColumnTransformer - Juntando Tudo
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```
- Aplica `numeric_transformer` nas colunas numéricas
- Aplica `categorical_transformer` nas colunas categóricas
- Combina tudo em uma única matriz processada

### Divisão Treino/Teste Inicial
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **80%** para treino, **20%** para teste
- `random_state=42`: garante que a divisão seja sempre a mesma (reprodutibilidade)

---

## 3️⃣ SELEÇÃO DO MELHOR MODELO COM OPTUNA

### Definir Modelos e Espaços de Busca

#### Random Forest
```python
"Random Forest": {
    "model": RandomForestRegressor(random_state=42, n_jobs=-1),
    "param_distributions": {
        "regressor__n_estimators": optuna.distributions.IntDistribution(100, 500),
        "regressor__max_depth": optuna.distributions.IntDistribution(5, 30),
        "regressor__min_samples_split": optuna.distributions.IntDistribution(2, 10),
        "regressor__min_samples_leaf": optuna.distributions.IntDistribution(1, 5),
    }
}
```

**Hiperparâmetros explicados:**
- **n_estimators** (100-500): Número de árvores na floresta
  - Mais árvores = mais estável, mas mais lento
- **max_depth** (5-30): Profundidade máxima de cada árvore
  - Maior = captura mais detalhes, mas pode overfitar
- **min_samples_split** (2-10): Mínimo de amostras para dividir um nó
  - Maior = árvore mais simples
- **min_samples_leaf** (1-5): Mínimo de amostras em cada folha
  - Maior = previne overfitting

#### XGBoost
```python
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
}
```

**Hiperparâmetros adicionais do XGBoost:**
- **learning_rate** (0.01-0.3): Taxa de aprendizado
  - Menor = aprende devagar, mas mais preciso
- **subsample** (0.6-1.0): Fração de amostras usadas em cada árvore
  - Menor = previne overfitting
- **colsample_bytree** (0.6-1.0): Fração de features usadas em cada árvore
- **reg_alpha** e **reg_lambda**: Regularização L1 e L2
  - Penaliza modelos muito complexos

#### LightGBM
```python
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
```

**Hiperparâmetro específico do LightGBM:**
- **num_leaves** (20-150): Número máximo de folhas por árvore
  - Controla a complexidade da árvore
- **min_child_samples** (5-50): Mínimo de dados necessários em uma folha

### Loop de Treinamento e Comparação
```python
for nome, config in modelos.items():
    print(f"\n   Otimizando {nome}...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', config["model"])
    ])
```
- Itera sobre os 3 modelos
- Cria um **Pipeline** completo: preprocessamento → modelo

### OptunaSearchCV - Otimização Inteligente
```python
optuna_search = OptunaSearchCV(
    estimator=pipeline,
    param_distributions=config["param_distributions"],
    cv=5,
    n_trials=20,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
```

**Parâmetros explicados:**
- **cv=5**: Validação cruzada com 5 folds
  - Divide treino em 5 partes, treina em 4 e valida em 1 (5 vezes)
- **n_trials=20**: Testa 20 combinações diferentes de hiperparâmetros
  - Optuna usa **otimização bayesiana**: aprende com tentativas anteriores
- **scoring='r2'**: Métrica de otimização (R² - quanto maior, melhor)
- **n_jobs=-1**: Usa todos os núcleos do processador

### Avaliar e Guardar Resultados
```python
optuna_search.fit(X_train, y_train)

score_train = optuna_search.best_score_
score_test = optuna_search.score(X_test, y_test)

melhores_modelos[nome] = optuna_search.best_estimator_

resultados.append({
    "Modelo": nome,
    "R² (CV Train)": score_train,
    "R² (Test)": score_test,
    "Melhores Params": optuna_search.best_params_
})
```

- **best_score_**: Melhor R² obtido na validação cruzada
- **score(X_test, y_test)**: R² no conjunto de teste (dados nunca vistos)
- **best_estimator_**: Pipeline completo com os melhores hiperparâmetros

### Selecionar o Vencedor
```python
df_resultados = pd.DataFrame(resultados).sort_values(by="R² (CV Train)", ascending=False)
melhor_nome = df_resultados.iloc[0]["Modelo"]
melhor_modelo = melhores_modelos[melhor_nome]
```
- Ordena por R² de validação cruzada
- Pega o primeiro (melhor modelo)

---

## 4️⃣ AVALIAÇÃO FINAL COM NESTED CROSS-VALIDATION

### Por que Nested CV?

**Problema**: Se otimizamos hiperparâmetros no mesmo conjunto onde avaliamos, podemos ter **overfitting** na escolha dos hiperparâmetros.

**Solução**: Nested CV (CV aninhado) tem dois loops:
- **Loop externo** (5-fold): Avaliação "honesta" do modelo
- **Loop interno** (3-fold): Otimização dos hiperparâmetros

### Configuração dos Folds
```python
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
```

### Loop Externo - Avaliação
```python
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
    X_train_fold = X.iloc[train_idx]
    X_test_fold = X.iloc[test_idx]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
```
- Divide os dados em 5 partes
- Em cada iteração: treina em 4/5 e testa em 1/5

### Loop Interno - Otimização
```python
search = OptunaSearchCV(
    estimator=pipeline,
    param_distributions=best_config["param_distributions"],
    cv=inner_cv,
    n_trials=15,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search.fit(X_train_fold, y_train_fold)
```
- **Dentro de cada fold externo**, otimiza hiperparâmetros com 3-fold CV
- Usa apenas os dados de treino do fold externo

### Avaliação no Fold de Teste
```python
y_pred_fold = search.predict(X_test_fold)
score = r2_score(y_test_fold, y_pred_fold)

outer_scores.append(score)
all_predictions.extend(y_pred_fold)
all_true_values.extend(y_test_fold)
```
- Faz predições no fold de teste
- Guarda os resultados de todos os folds

### Cálculo de Estatísticas Finais
```python
mean_r2 = np.mean(outer_scores)
std_r2 = np.std(outer_scores)
ci95 = 1.96 * std_r2 / np.sqrt(5)
```

**Intervalo de Confiança 95%:**
- Fórmula: `média ± 1.96 * (desvio_padrão / √n)`
- 1.96 vem da distribuição normal (95% dos valores)
- Indica: "temos 95% de confiança que o R² verdadeiro está neste intervalo"

### Métricas Agregadas
```python
mae = mean_absolute_error(all_true_values, all_predictions)
rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
r2_final = r2_score(all_true_values, all_predictions)
```

- **MAE** (Mean Absolute Error): Erro médio em Reais
  - Se MAE = 50.000, erra em média ±R$ 50.000
- **RMSE** (Root Mean Squared Error): Penaliza erros grandes
  - Mais sensível a outliers que MAE
- **R²**: Percentual de variância explicada (0 a 1)
  - R² = 0.92 → modelo explica 92% da variação nos preços

---

## 5️⃣ TREINO FINAL E SALVAMENTO DO MODELO

### Por que treinar novamente?

Até agora, treinamos múltiplas vezes em subconjuntos dos dados (folds). Agora vamos treinar **uma última vez com 100% dos dados** para ter o modelo mais robusto possível.

### Treino com Todos os Dados
```python
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', best_config["model"])
])

final_search = OptunaSearchCV(
    estimator=final_pipeline,
    param_distributions=best_config["param_distributions"],
    cv=5,
    n_trials=30,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

final_search.fit(X, y)
```
- Usa **todo o dataset** (X, y)
- Otimiza com 30 trials (mais tentativas = melhor resultado)

### Salvar Modelo
```python
model_path = os.path.join(OUTPUT_DIR, "modelo_final_carros.pkl")
joblib.dump(final_search.best_estimator_, model_path)
```
- Salva o pipeline completo (preprocessamento + modelo) em arquivo
- `.pkl` = formato pickle/joblib (binário comprimido)

---

## 6️⃣ VISUALIZAÇÕES

### Gráfico 1: Comparação de Modelos
```python
plt.figure(figsize=(10, 6))
models = df_resultados['Modelo'].values
scores = df_resultados['R² (CV Train)'].values
colors = ['gold' if m == melhor_nome else 'skyblue' for m in models]

plt.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
```

**O que mostra:**
- Gráfico de barras com R² de cada modelo
- Modelo vencedor em dourado, outros em azul

### Gráfico 2: Predições vs Valores Reais
```python
plt.scatter(all_true_values, all_predictions, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([all_true_values.min(), all_true_values.max()], 
         [all_true_values.min(), all_true_values.max()], 
         'r--', lw=2, label='Predição Perfeita')
```

**O que mostra:**
- Cada ponto = um carro
- Eixo X = preço real
- Eixo Y = preço predito
- Linha vermelha = predição perfeita (se todos os pontos estivessem nela, acertaríamos 100%)
- Quanto mais próximos da linha, melhor o modelo

### Gráfico 3: Importância das Features
```python
if hasattr(final_search.best_estimator_.named_steps['regressor'], 'feature_importances_'):
    importances = final_search.best_estimator_.named_steps['regressor'].feature_importances_
```

**O que mostra:**
- Quais features mais influenciam as predições
- Valores maiores = mais importantes
- Útil para entender o modelo e o negócio

---

## 🎓 Conceitos-Chave para Explicar

### 1. **Pipeline**
- Sequência de transformações
- Garante que teste receba mesmo pré-processamento que treino
- Evita "data leakage" (vazamento de informação)

### 2. **Validação Cruzada (Cross-Validation)**
- Divide dados em K partes
- Treina K vezes, cada vez usando parte diferente como teste
- Resultado mais confiável que treino/teste simples

### 3. **Nested Cross-Validation**
- CV dentro de CV
- Loop externo: avaliação do modelo
- Loop interno: otimização de hiperparâmetros
- Evita overfitting na seleção de hiperparâmetros

### 4. **Optuna vs GridSearch**
- **GridSearch**: Testa TODAS as combinações (lento)
- **Optuna**: Usa otimização bayesiana (inteligente)
  - Aprende com tentativas anteriores
  - Foca em regiões promissoras
  - Muito mais rápido para espaços grandes

### 5. **R² (Coeficiente de Determinação)**
- Varia de 0 a 1 (ou negativo se pior que média)
- R² = 0.92 → modelo explica 92% da variação
- R² = 1.0 → predição perfeita
- R² = 0.0 → modelo não melhor que usar sempre a média

### 6. **Overfitting vs Underfitting**
- **Overfitting**: Modelo "decora" dados de treino, falha no teste
- **Underfitting**: Modelo muito simples, não captura padrões
- **Solução**: Validação cruzada, regularização, ensemble de modelos

---

## 📊 Fluxo Completo Resumido

```
1. DADOS
   ↓
2. PRÉ-PROCESSAMENTO (Pipeline)
   ├─ Numéricos: Imputer → Scaler
   └─ Categóricos: Imputer → OneHot
   ↓
3. SELEÇÃO DE MODELO (3 candidatos)
   ├─ Random Forest
   ├─ XGBoost
   └─ LightGBM
   ↓ (Optuna otimiza cada um)
   ↓
4. VENCEDOR: LightGBM
   ↓
5. NESTED CV (avaliação robusta)
   ├─ Loop externo: 5-fold
   └─ Loop interno: 3-fold + Optuna
   ↓
6. TREINO FINAL (100% dos dados)
   ↓
7. SALVAR MODELO + GRÁFICOS
```

---

## 💡 Dicas para Apresentação

### Estrutura Sugerida:

1. **Introdução** (2 min)
   - Objetivo: Prever preços de carros
   - Dataset: 954 carros com 11 features

2. **Pré-processamento** (3 min)
   - Por que Pipeline?
   - Diferença entre features numéricas e categóricas
   - StandardScaler e OneHotEncoder

3. **Seleção de Modelos** (4 min)
   - Por que 3 modelos?
   - O que é Optuna e por que é melhor que GridSearch
   - Hiperparâmetros principais de cada modelo

4. **Validação** (3 min)
   - O que é Cross-Validation
   - Por que Nested CV
   - Intervalo de confiança

5. **Resultados** (2 min)
   - R² = 0.92 → explica 92% da variação
   - MAE = R$ 50k → erro médio
   - Mostrar gráficos

6. **Conclusão** (1 min)
   - LightGBM venceu
   - Modelo salvo e pronto para uso
   - Features mais importantes

### Perguntas que Podem Fazer:

❓ **Por que usar 3 modelos?**
→ Cada um tem vantagens: RF robusto, XGBoost preciso, LightGBM rápido. Comparamos para escolher o melhor.

❓ **O que é R²?**
→ Percentual da variação nos preços que o modelo consegue explicar. 0.92 = 92%.

❓ **Por que Nested CV?**
→ Para ter certeza que não estamos "trapaceando" ao otimizar hiperparâmetros. É a forma mais honesta de avaliar.

❓ **Como usar o modelo?**
→ `modelo = joblib.load('output/modelo_final_carros.pkl')`
→ `preco = modelo.predict(dados_carro_novo)`

---

## 🔍 Análise de Cada Seção do Output

### Seção 1: Carregamento
```
Dataset carregado: 954 linhas, 11 colunas
Features numéricas: ['cc', 'hp', 'max_speed', '0_100_sec', 'seats', 'torque']
Features categóricas: ['company', 'fuel_type']
```
→ 954 carros, 6 features numéricas, 2 categóricas

### Seção 3: Comparação
```
LightGBM        | R² CV: 0.9222 | R² Test: 0.9491
XGBoost         | R² CV: 0.9171 | R² Test: 0.9597
Random Forest   | R² CV: 0.9147 | R² Test: 0.9574
```
→ LightGBM venceu na CV, mas XGBoost foi melhor no teste
→ Escolhemos pelo CV (mais confiável que teste único)

### Seção 4: Nested CV
```
R² médio (5-fold):     0.9208 ± 0.0242 (IC 95%)
Intervalo de confiança: [0.8966, 0.9450]
MAE:  R$ 49,705.55
RMSE: R$ 97,869.95
```
→ R² consistente nos 5 folds
→ Erro médio de ~50 mil reais
→ Modelo confiável!

---

Boa sorte na apresentação! 🚀
