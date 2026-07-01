import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt


def main():
    df = parse_data_stock()
    length = len(df)
    df_training = df.head(int(length * 0.8)).copy()
    df_testing  = df.tail(int(length * 0.2)).copy()

    # Normaliza o volume usando APENAS estatísticas do treino (evita data leakage)
    vol_max = df_training['last_volume'].max()
    vol_min = df_training['last_volume'].min()

    df_training['last_volume'] = (df_training['last_volume'] - vol_min) / (vol_max - vol_min)
    df_testing ['last_volume'] = (df_testing['last_volume'] - vol_min) / (vol_max - vol_min)

    # Variáveis de Treino e Teste
    X_training = df_training[['open', 'last_high', 'last_low', 'last_close', 'last_volume']]
    X_testing  = df_testing [['open', 'last_high', 'last_low', 'last_close', 'last_volume']]

    step    = 0.01
    samples = len(X_training)

    weights = np.zeros(5)
    bias    = 0

    dollars_error_current  = 1
    dollars_error_expected = 0.01
    iteration      = 1
    MAX_ITERATIONS = 250_000  # guarda de segurança: evita loop infinito caso não convirja

    # Treinando o modelo para prever a máxima (high) de cada dia
    while dollars_error_current > dollars_error_expected and iteration <= MAX_ITERATIONS:
        ŷ = (X_training @ weights) + bias
        errors = (ŷ - df_training['high'])

        weights_loss_derivative = (errors @ X_training) * 2 / samples
        bias_loss_derivative    = (errors.mean() * 2)

        weights = weights - (step * weights_loss_derivative)
        bias    = bias    - (step * bias_loss_derivative)

        mse  = (errors ** 2).mean()
        dollars_error_current = math.sqrt(mse)

        if iteration % 1000 == 0 or dollars_error_current <= dollars_error_expected:
            print(f'Iter {iteration} | Erro médio em Dólares: ${dollars_error_current:.5f}')
        iteration += 1

    print('============== Treino Finalizado ==============\n')
    print(f'Weights:\n{weights}\n')
    print(f'Bias: {bias}\n')

    # ---- Avaliação no conjunto de teste (dados nunca vistos pelo modelo) ----
    ŷ = (X_testing @ weights + bias)
    errors = (ŷ - df_testing['high'])

    df_testing['Predicted_High'] = ŷ
    df_testing['Dollar_Error']   = errors.abs()
    avarage_testing_error = math.sqrt((errors ** 2).mean())

    print(f"Erro Médio Final no Conjunto de Teste: ${avarage_testing_error:.2f}\n")
    print(df_testing[['date', 'high', 'Predicted_High', 'Dollar_Error']].tail(10))


    # É AQUI que entra a parte nova pedida: visualizar, dia a dia, o quão
    # boa (ou ruim) foi a previsão do modelo no período de teste.
    plot_daily_analysis(df_testing, avarage_testing_error)


def plot_daily_analysis(df_testing, avg_error):
    # Converte a coluna 'date' (string dd/mm/aaaa) para datetime, pra ficar
    # bonito no eixo X dos gráficos e podermos filtrar por período.
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
    # Com ~1400 dias no teste, o gráfico acima fica "espremido". Esse aqui
    # mostra o detalhe diário de forma legível.
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
    df = (pd.read_csv('nvidia_stock_data_1999_2026.csv')).drop(columns=['shares_outstanding_bn', 'market_cap_usd_bn',
                                                                      'quarterly_revenue_usd_bn','sma_20', 'sma_50', 'sma_200',
                                                                      'rsi_14', 'stock_split', 'era', 'key_event'])
    shifted_features = df[['open', 'high', 'low', 'close', 'volume']].shift(1).add_prefix('last_')
    df = (pd.concat([df, shifted_features], axis=1)).dropna()

    return df

if __name__ == "__main__":
    main()