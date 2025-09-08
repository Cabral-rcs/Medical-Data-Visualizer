
# Etapa 1: Importar bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Etapa 2: Importar os dados do csv para um dataset
df = pd.read_csv('medical_examination.csv')

# Etapa 3: Adicionar coluna 'overweight' e calcular IMC
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Etapa 4: Normalizar cholesterol e gluc
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Etapa 5: Desenhar o gráfico categórico na função draw_cat_plot
def draw_cat_plot():

    # Etapa 6: Criar DataFrame df_cat usando pd.melt 
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Etapa 7: Agrupar e reformatar df_cat para dividir por cardio e exibir contagens
    # Renomeia a coluna de contagens para 'total' 
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Etapa 8: Converter dados para formato longo e cria o gráfico 
    cat = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    
    # Etapa 9: Obtém a saída do gráfico e armazenar em fig
    fig = cat.fig
    
    # Etapa 10: salvar o gráfico
    fig.savefig('catplot.png')
    return fig

# Etapa 11: Desenhar o mapa de calor na função draw_heat_map
def draw_heat_map():

    # Etapa 12: Limpar dados na variável df_heat, filtrando dados incorretos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Pressão diastólica <= sistólica
        (df['height'] >= df['height'].quantile(0.025)) &  # Altura >= percentil 2.5
        (df['height'] <= df['height'].quantile(0.975)) &  # Altura <= percentil 97.5
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Peso >= percentil 2.5
        (df['weight'] <= df['weight'].quantile(0.975))    # Peso <= percentil 97.5
    ]
    
    # Etapa 13: Calcular a matriz de correlação e armazenar em corr
    corr = df_heat.corr()
    
    # Etapa 14: Gerar máscara para o triângulo superior e armazenar em mask
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Etapa 15: Montar a figura matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Etapa 16: Traçar a matriz de correlação com sns.heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, ax=ax)
    
    # Etapa 17: Salva o gráfico
    fig.savefig('heatmap.png')
    return fig