import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    df = parse_data_stock()
    length = len(df)
    
    # Separação temporária para calcular o máximo e mínimo APENAS do treino
    df_training_temp = df.head(int(length * 0.8))
    vol_max = df_training_temp['last_volume'].max()
    vol_min = df_training_temp['last_volume'].min()

    # Normaliza o volume no DATASET INTEIRO usando os parâmetros do treino (evita data leakage)
    df['last_volume'] = (df['last_volume'] - vol_min) / (vol_max - vol_min)

    # Divisão definitiva das partições
    df_training = df.head(int(length * 0.8)).copy()
    df_testing  = df.tail(int(length * 0.2)).copy()

    # Variáveis de Treino
    X_training = df_training[['open', 'last_high', 'last_low', 'last_close', 'last_volume']]

    step    = 0.01
    samples = len(X_training)

    weights = np.zeros(5)
    bias    = 0

    dollars_error_current  = 1
    dollars_error_expected = 0.01
    iteration      = 1
    MAX_ITERATIONS = 250_000

    # Treinando o modelo para prever a máxima (high) com dados de treino
    while dollars_error_current > dollars_error_expected and iteration <= MAX_ITERATIONS:
        ŷ_train = (X_training @ weights) + bias
        errors_train = (ŷ_train - df_training['high'])

        weights_loss_derivative = (errors_train @ X_training) * 2 / samples
        bias_loss_derivative    = (errors_train.mean() * 2)

        weights = weights - (step * weights_loss_derivative)
        bias    = bias    - (step * bias_loss_derivative)

        mse  = (errors_train ** 2).mean()
        dollars_error_current = math.sqrt(mse)

        if iteration % 1000 == 0 or dollars_error_current <= dollars_error_expected:
            print(f'Iter {iteration} | Erro médio em Dólares: ${dollars_error_current:.5f}')
        iteration += 1

    print('============== Treino Finalizado ==============\n')
    print(f'Weights:\n{weights}\n')
    print(f'Bias: {bias}\n')

    # ---- GERAÇÃO DE PREDIÇÕES PARA O DATASET INTEIRO ----
    # Isso nos permite mapear TODAS as eras históricas no gráfico final
    X_all = df[['open', 'last_high', 'last_low', 'last_close', 'last_volume']]
    df['Predicted_High'] = (X_all @ weights) + bias

    # Atualiza as fatias de treino e teste agora contendo a coluna predita
    df_training = df.head(int(length * 0.8)).copy()
    df_testing  = df.tail(int(length * 0.2)).copy()

    # ---- Avaliação apenas no conjunto de teste (métrica oficial de validação) ----
    errors_test = (df_testing['Predicted_High'] - df_testing['high'])
    df_testing['Dollar_Error'] = errors_test.abs()
    avarage_testing_error = math.sqrt((errors_test ** 2).mean())

    print(f"Erro Médio Final no Conjunto de Teste: ${avarage_testing_error:.2f}\n")
    print(df_testing[['date', 'era', 'high', 'Predicted_High', 'Dollar_Error']].tail(10))

    # ---- Geração dos Gráficos ----
    plot_daily_analysis(df_testing, avarage_testing_error)
    plot_era_analysis(df)  # Enviando o dataframe completo para capturar todas as eras desde 1999


def plot_era_analysis(df_completo):
    """
    Gera um gráfico de linhas contendo TODAS as Eras do dataset no Eixo X
    e a média do valor em Dólar no Eixo Y, utilizando escala logarítmica
    para evidenciar as discrepâncias ocultas nas eras de menor valor nominal.
    """
    # Agrupa por era mantendo a ordem cronológica original do dataset (sort=False)
    df_era = df_completo.groupby('era', sort=False)[['high', 'Predicted_High']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plota o histórico completo agregado por era
    ax.plot(df_era['era'], df_era['high'], label='Valor Real Médio', 
            color='black', marker='o', linewidth=2.5, markersize=6)
    ax.plot(df_era['era'], df_era['Predicted_High'], label='Valor Previsto Médio', 
            color='orange', marker='s', linewidth=2, linestyle='--', alpha=0.85)

    # AQUI ESTÁ O SEGREDO: Aplica escala logarítmica no Eixo Y
    ax.set_yscale('log')
    
    # Formata os ticks do eixo Y para que mostrem os valores reais de dólares de forma legível
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%1.2f'))
    
    # Opcional: Define pontos de parada específicos no Eixo Y para guiar o olho do leitor
    # Seus preços médios vão desde ~$0.03 (1999) até ~$110.00 (2026)
    ax.set_yticks([0.03, 0.10, 0.50, 1.00, 5.00, 20.00, 50.00, 120.00])

    # Customizações estéticas e científicas
    ax.set_title('Evolução e Discrepâncias do Modelo por Todas as Eras da NVIDIA (Escala Logarítmica)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Eras Históricas da Empresa', fontsize=12, labelpad=10)
    ax.set_ylabel('Preço Médio da Ação em Escala Log (USD)', fontsize=12, labelpad=10)
    
    # Rotaciona os rótulos do eixo X para melhor leitura
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    ax.legend(fontsize=11, loc='upper left')
    
    # Ativa grades principais e secundárias (comum em gráficos logarítmicos)
    ax.grid(True, which="both", linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('analise_por_era.png', dpi=150)
    print("Gráfico 'analise_por_era.png' com escala ajustada salvo com sucesso!")

def plot_daily_analysis(df_testing, avg_error):
    df_testing = df_testing.copy()
    df_testing['date'] = pd.to_datetime(df_testing['date'], dayfirst=True)
    df_testing = df_testing.sort_values('date')

    # ---- Gráfico 1: Real vs Previsto + Erro diário, período de teste completo ----
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(df_testing['date'], df_testing['high'], label='High Real',
                color='black', linewidth=1)
    axs[0].plot(df_testing['date'], df_testing['Predicted_High'], label='High Previsto',
                color='orange', linewidth=1, alpha=0.85)
    axs[0].set_ylabel('Preço (USD)')
    axs[0].set_title('Análise Diária — Máxima Real vs Prevista (todo o conjunto de teste)')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].bar(df_testing['date'], df_testing['Dollar_Error'], color='crimson', width=1)
    axs[1].axhline(avg_error, color='blue', linestyle='--',
                    label=f'Erro médio (RMSE): ${avg_error:.2f}')
    axs[1].set_ylabel('Erro Absoluto (USD)')
    axs[1].set_xlabel('Data')
    axs[1].set_title('Erro de Previsão Dia a Dia')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('analise_diaria_completa.png', dpi=150)

    # ---- Gráfico 2: Zoom nos últimos 90 dias de pregão ----
    ultimos = df_testing.tail(90)
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(ultimos['date'], ultimos['high'], label='High Real',
             marker='o', markersize=3, color='black')
    ax2.plot(ultimos['date'], ultimos['Predicted_High'], label='High Previsto',
             marker='o', markersize=3, color='orange')
    ax2.set_title('Zoom — Últimos 90 Dias de Pregão (Teste)')
    ax2.set_ylabel('Preço (USD)')
    ax2.set_xlabel('Data')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analise_diaria_zoom90.png', dpi=150)

    plt.show()


def parse_data_stock():
    df = (pd.read_csv('nvidia_stock_data_1999_2026.csv')).drop(columns=[
        'shares_outstanding_bn', 'market_cap_usd_bn', 'quarterly_revenue_usd_bn',
        'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'stock_split', 'key_event'
    ])
    
    shifted_features = df[['open', 'high', 'low', 'close', 'volume']].shift(1).add_prefix('last_')
    df = (pd.concat([df, shifted_features], axis=1)).dropna()

    return df

if __name__ == "__main__":
    main()