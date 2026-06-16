import pandas as pd
import numpy  as np
import math
def main():
    df = parse_data_stock()
    length = len(df)
    df_training = df.head(int(length * 0.8))
    df_testing  = df.tail(int(length * 0.2))


    # Normalizes volume into 0 - 1 interval
    vol_max = df_training['last_volume'].max()
    vol_min = df_training['last_volume'].min()

    df_training['last_volume'] = (df_training['last_volume'] - vol_min) / (vol_max - vol_min)
    df_testing ['last_volume'] = (df_testing['last_volume'] - vol_min) / (vol_max - vol_min)

    # Feature Variables for Training and Testing
    X_training = df_training[['open', 'last_high', 'last_low', 'last_close', 'last_volume']]
    X_testing  = df_testing [['open', 'last_high', 'last_low', 'last_close', 'last_volume']]


    step         = 0.01
    samples      = len(X_training)

    weights = np.zeros(5)
    bias    = 0
    
    dollars_error_current  = 1
    dollars_error_expected = 0.01
    iteration = 1

    # Training the Model to predict Todays High
    while dollars_error_current > dollars_error_expected:
        # Vector multiplication of Features and Weights
        ŷ = (X_training @ weights) + bias
        errors = (ŷ - df_training['high'])

        weights_loss_derivative = (errors @ X_training) * 2/samples
        bias_loss_derivative    = (errors.mean() * 2)
        
        weights = weights - (step * weights_loss_derivative)
        bias    = bias    - (step * bias_loss_derivative)

        mse  = (errors ** 2).mean()
        dollars_error_current = math.sqrt(mse)
        print(f'Iter {iteration} | Avarege Error in Dollars: ${dollars_error_current}')
        iteration += 1

    print ('============== Iterations Finished ==============\n')
    print (f'Weights:\n{weights}\n')
    print (f'Bias: {bias}')


    ŷ = (X_testing @ weights + bias)
    errors = (ŷ - df_testing['high'])

    dollars_error_per_row = errors.__abs__()
    
    df_testing['Predicted_High'] = ŷ
    df_testing['Dollar_Error']   = dollars_error_per_row
    avarage_testing_error = math.sqrt((errors ** 2).mean())

    print(f"Final Average Error in Untouched Testing Dataframe: ${avarage_testing_error:.2f}\n")
    print(df_testing[['date', 'high', 'Predicted_High', 'Dollar_Error']].tail(10))



def parse_data_stock():
    df = (pd.read_csv('nvidia_stock_data_1999_2026.csv')).drop(columns=['shares_outstanding_bn', 'market_cap_usd_bn',
                                                                      'quarterly_revenue_usd_bn','sma_20', 'sma_50', 'sma_200',
                                                                      'rsi_14', 'stock_split', 'era', 'key_event'])
    shifted_features = df[['open', 'high', 'low', 'close', 'volume']].shift(1).add_prefix('last_')
    df = (pd.concat([df, shifted_features], axis=1)).dropna()
    
    return df

if __name__ == "__main__":
    main()


