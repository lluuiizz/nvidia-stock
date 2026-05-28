import pandas as pd
import numpy  as np
import math
def main():
    df = parse_data_stock()
    length = len(df)
    df_training = df.head(int(length * 0.8))
    df_testing  = df.tail(int(length * 0.2))

    X_training = df_training[['open', 'last_high', 'last_low', 'last_close', 'last_volume']]
    X_testing  = df_testing [['open', 'last_high', 'last_low', 'last_close', 'last_volume']]


    iterations   = 250000
    step         = 0.01
    samples      = len(X_training)

    weights = np.zeros(5)
    bias    = 0

    # Training the Model to predict Todays High
    for i in range(iterations):
        # Vector multiplication of Features and Weights
        ŷ = (X_training @ weights) + bias
        errors = (ŷ - df_training['high'])

        weights_loss_derivative = (errors @ X_training) * 2/samples
        bias_loss_derivative    = (errors.mean() * 2)
        
        weights = weights - (step * weights_loss_derivative)
        bias    = bias    - (step * bias_loss_derivative)

        mse  = (errors ** 2).mean()
        rmse = math.sqrt(mse)
        print(f'Iter {i+1} | Avarege Error in Dollars: ${rmse}')

    print ('============== Iterations Finished ==============\n')
    print (f'Weights:\n{weights}\n')
    print (f'Bias: {bias}')



def parse_data_stock():
    df = (pd.read_csv('nvidia_stock_data_1999_2026.csv')).drop(columns=['date', 'shares_outstanding_bn', 'market_cap_usd_bn',
                                                                      'quarterly_revenue_usd_bn','sma_20', 'sma_50', 'sma_200',
                                                                      'rsi_14', 'stock_split', 'era', 'key_event'])
    shifted_features = df[['open', 'high', 'low', 'close', 'volume']].shift(1).add_prefix('last_')
    df = (pd.concat([df, shifted_features], axis=1)).dropna()
    
    # Normalizes volume into 0 - 1 interval
    vol_max = df['last_volume'].max()
    vol_min = df['last_volume'].min()
    df['last_volume'] = (df['last_volume'] - vol_min) / (vol_max - vol_min)
    return df

if __name__ == "__main__":
    main()


