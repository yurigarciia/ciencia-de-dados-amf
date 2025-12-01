import pandas as pd
import re
 
imdb_df = pd.read_csv('dataset_original.csv') 
imdb_df.head()
imdb_df.info()

def tratar_preco(preco):
    if pd.isna(preco):
        return None
    preco = preco.replace('$', '').replace('US$', '').strip()
    preco = re.sub(r'[^0-9\-,]', '', preco)
    if '-' in preco:
        _, max_val = preco.split('-')
        max_val = max_val.replace(',', '').strip()
        if max_val == '':
            return None
        valor = float(max_val)
    else:
        val = preco.replace(',', '').strip()
        if val == '':
            return None
        valor = float(val)
    valor_reais = valor * 5.2
    return valor_reais

def tratar_hp(valor):
    if pd.isna(valor) or str(valor).strip() in ['', '-', 'N/A', 'unknown']:
        return None
    valor = str(valor).lower().replace('hp', '').strip()
    if '-' in valor:
        partes = valor.split('-')
        numeros = [int(p.strip()) for p in partes if p.strip().isdigit()]
        if numeros:
            return max(numeros)
        else:
            return None
    valor = valor.strip()
    return int(valor) if valor.isdigit() else None

def tratar_cc(valor):
    if pd.isna(valor) or str(valor).strip() in ['', '-', 'N/A', 'unknown']:
        return None
    valor_proc = str(valor).lower().replace('"', '').strip()
    valor_proc = valor_proc.replace(',', '').replace('.', '')
    match = re.match(r'(\d+)(\s*)(cc)?', valor_proc)
    if not match:
        return None
    numero = int(match.group(1))
    return numero

def tratar_torque(valor):
    valor = str(valor).replace(' Nm', '').replace('Nm', '').strip()
    if '-' in valor:
        partes = valor.split('-')
        numeros = [float(p.strip()) for p in partes if p.strip().replace('.', '', 1).isdigit()]
        if numeros:
            return max(numeros)
        else:
            return None
    valor = valor.strip()
    return float(valor) if valor.replace('.', '', 1).isdigit() else None

def tratar_0_100_sec(valor):
    valor = str(valor).replace(' sec', '').strip()
    if '-' in valor:
        partes = valor.split('-')
        numeros = [float(p.strip()) for p in partes if p.strip().replace('.', '', 1).isdigit()]
        if numeros:
            return max(numeros)
        else:
            return None
    valor = valor.strip()
    return float(valor) if valor.replace('.', '', 1).isdigit() else None

def agrupar_tipo_combustivel(tipo):
    tipo = str(tipo).lower()
    if 'petrol' in tipo:
        return 'Petrol'
    elif 'diesel' in tipo:
        return 'Diesel'
    else:
        return 'Others'

def normalize_brand(brand):
    if pd.isna(brand):
        return None
    return str(brand).strip().lower().replace(' ', '')
    
    
# Renomear colunas principais no início do tratamento
df = pd.read_csv('dataset.csv')
df = df.rename(columns={
    'Cars Prices': 'price',
    'Total Speed': 'max_speed',
    'Fuel Types': 'fuel_type',
    'HorsePower': 'hp',
    'Cars Names': 'model',
    'Company Names': 'company',
    'Engines': 'engine',
    'Seats': 'seats',
    'Torque': 'torque',
    'CC/Battery Capacity': 'cc',
    'Performance(0 - 100 )KM/H': '0_100_sec'
})

# Filtrar apenas veículos a combustão (remover híbridos e elétricos)
df = df[~df['fuel_type'].str.contains('Electric|Hybrid|hybrid|electric', case=False, na=False)]
df.to_csv('dataset_tratado.csv', index=False)

# Etapa 1: tratar coluna Torque (remover 'Nm' e pegar maior valor em caso de intervalo)
torque_df = pd.read_csv('dataset_tratado.csv')
torque_df['torque'] = torque_df['torque'].apply(tratar_torque)
torque_df.to_csv('dataset_tratado.csv', index=False)

# Etapa 2: tratar a coluna Performance(0 - 100 )KM/H, tirar o "sec" e pegar maior valor em caso de intervalo
performance_df = pd.read_csv('dataset_tratado.csv')
performance_df['0_100_sec'] = performance_df['0_100_sec'].apply(tratar_0_100_sec)
performance_df.to_csv('dataset_tratado.csv', index=False)

# Etapa 3: tratar a coluna price, converter para float em reais
df = pd.read_csv('dataset_tratado.csv')
df['price'] = df['price'].apply(tratar_preco)
df.to_csv('dataset_tratado.csv', index=False)

# Etapa 4: tratar a coluna max_speed, remover 'km/h'
max_speed_df = pd.read_csv('dataset_tratado.csv')
max_speed_df['max_speed'] = max_speed_df['max_speed'].astype(str).str.replace(' km/h', '', regex=False).str.replace('km/h', '', regex=False).str.strip()
max_speed_df.to_csv('dataset_tratado.csv', index=False)

# Etapa 5: tratar a coluna hp, extrair o maior valor em caso de faixa
df = pd.read_csv('dataset_tratado.csv')
df['hp'] = df['hp'].apply(tratar_hp)
df.to_csv('dataset_tratado.csv', index=False)

# Etapa 6: tratar a coluna cc
df = pd.read_csv('dataset_tratado.csv')
df['cc'] = df['cc'].apply(tratar_cc)
df.to_csv('dataset_tratado.csv', index=False)

# Verificação de nullos
df = pd.read_csv('dataset_tratado.csv')
print(df.isnull().sum())

# Remover campos nullos da coluna cc, hp e torque
df = df.dropna(subset=['cc', 'hp', 'torque'])
# Remover carros com preço acima de 2 milhões
df = df[df['price'] <= 2_000_000]

# Aplicar agrupamento
df['fuel_type'] = df['fuel_type'].apply(agrupar_tipo_combustivel)

# Normalizar marcas na coluna original
df['company'] = df['company'].apply(normalize_brand)

# Agrupar marcas com menos de 1% como 'others' na coluna original
brand_counts = df['company'].value_counts(normalize=True)
brands_to_others = brand_counts[brand_counts < 0.01].index
df['company'] = df['company'].apply(lambda x: 'others' if x in brands_to_others else x)

df.to_csv('dataset_tratado.csv', index=False)

# Verificação de nullos
df = pd.read_csv('dataset_tratado.csv')
print(df.isnull().sum())
