# load the libs
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers  import Dense, Input, BatchNormalization 
from tensorflow.keras.models  import Model 
from sklearn.model_selection  import train_test_split 
import pandas as pd 
import sklearn.metrics  as sm 
import matplotlib.pyplot as plt
import os
# seting radom seeds for repeat 
np.random.seed(42) 
tf.random.set_seed(42)
# define architecture of deep neural networks
def build_model():
    input_layer = Input(shape=(4,))

    # shared feature extraction layer
    x = Dense(200, activation='tanh', kernel_initializer='he_normal')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(200, activation='tanh', kernel_initializer='he_normal')(x)
    x = Dense(200, activation='tanh', kernel_initializer='he_normal')(x)
    x = Dense(200, activation='tanh', kernel_initializer='he_normal')(x)

    # outputs three kinds of stress 
    outputs = [
        Dense(13872, name=f'{name}_output')(x)
        for name in ['Sigma11', 'Sigma22', 'Sigma12']
    ]

    return Model(inputs=input_layer, outputs = outputs)

# ploting training processes
def plot_convergence(history, output_names):
    plt.figure(figsize=(15, 10))

    # Plot the loss curve for each output
    plt.subplot(2, 1, 1)
    for name in output_names:
        plt.plot(history.history[f'{name}_output_loss'],
                 alpha=0.7,
                 linestyle='--',
                 label=f'{name} Train')
        if f'val_{name}_output_loss' in history.history:
            plt.plot(history.history[f'val_{name}_output_loss'],
                     alpha=0.9,
                     label=f'{name} Val')

    plt.title('Individual  Output Loss Curves', fontsize=14)
    plt.ylabel('MSE  Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Plot the total loss curve
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], 'k-', label='Total Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], 'r-', label='Total Val Loss')

    plt.title('Aggregated  Training')
    plt.show()
#
# data preprocessing
class DataProcessor:
    @staticmethod 
    def normalize(data, mean=None, std=None):
        if mean is None or std is None:
            mean = np.mean(data,  axis=0)
            std = np.std(data,  axis=0)
        return mean, std, (data - mean) / std 
 
# the main function 
if __name__ == "__main__":
    # setup parameters
    # input data files
    DATA_FILES = ['./Sigma11.csv',  './Sigma22.csv',  './Sigma12.csv'] 
    #output data files
    OUTPUT_NAMES = ['SM_Sigma11', 'SM_Sigma22', 'SM_Sigma12']
    
    # load data 
    print("Loading data...") 
    all_data = [pd.read_csv(f).values for f in DATA_FILES]
    X = all_data[0][:, 0:4]  # input variables 
    Y = np.stack([d[:,  4:13876] for d in all_data], axis=1)  # shape: (n_samples, 3, 13872)
    
    # spliting data 
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42 
    )


    # normalizing data
    processor = DataProcessor()
    x_mean, x_std, x_train_scaled = processor.normalize(x_train) 
    _, _, x_test_scaled = processor.normalize(x_test,  x_mean, x_std)
    print(x_train_scaled.shape)
    # normalize processing for each outputs 
    y_stats = []
    y_train_scaled = []
    for i in range(3):
        mean, std, scaled = processor.normalize(y_train[:,  i, :])
        y_stats.append((mean,  std))
        y_train_scaled.append(scaled)

    # biudling models 
    print("\nBuilding model...")
    model = build_model()
    model.compile( 
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
        loss='mse'
    )
    
    # training models 
    print("\nTraining model...")
    history = model.fit( 
        x_train_scaled,
        {f'{name}_output': y_train_scaled[i] for i, name in enumerate(OUTPUT_NAMES)},
        epochs=400,
        batch_size=128,
        verbose=1 
    )
    plot_convergence(history, OUTPUT_NAMES)
    # 6. 模型评估 
    print("\nEvaluating model...")
    predictions = model.predict(x_test_scaled) 
    
    # inverse normalization
    results = {}
    for i, name in enumerate(OUTPUT_NAMES):
        y_pred = predictions[i] * y_stats[i][1] + y_stats[i][0]
        y_true = y_test[:, i, :]
        
        r2 = sm.r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_pred  - y_true))
        rel_err = np.mean(np.abs((y_pred  - y_true) / (y_true)))
        
        results[name] = {
            'R2': r2,
            'MAE': mae,
            'Relative_Error': rel_err 
        }
    
    # print results
    print("\nEvaluation Results:")
    for name, metrics in results.items(): 
        print(f"{name}:")
        print(f"  R² = {metrics['R2']:.4f}")
        print(f"  MAE = {metrics['MAE']:.4f}")
        print(f"  Relative Error = {metrics['Relative_Error']:.4f}\n")
    
    # save results
    model.save('multi_output_model_v2.h5') 
    print("Model saved successfully.")

    os.makedirs('csv_results', exist_ok=True)


    def save_to_csv(y_true, y_pred, output_name):
        """save true values, predicted values, and relative errors to CSV"""
        # save true values,shape: n_samples × 13872
        pd.DataFrame(y_true).to_csv(
            f'csv_results/{output_name}_true.csv',
            index=False,
            float_format='%.6f'
        )

        # save predicted values
        pd.DataFrame(y_pred).to_csv(
            f'csv_results/{output_name}_pred.csv',
            index=False,
            float_format='%.6f'
        )

        # calculate and save relative errors
        rel_error = np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8))
        pd.DataFrame(rel_error).to_csv(
            f'csv_results/{output_name}_rel_error.csv',
            index=False,
            float_format='%.6f'
        )


    # save each outputs
    for i, name in enumerate(OUTPUT_NAMES):
        y_true = y_test[:, i, :]
        y_pred = predictions[i] * y_stats[i][1] + y_stats[i][0]
        save_to_csv(y_true, y_pred, name)

    print("CSV file saved successfully: ")
    print(f"├── csv_results/{name}_true.csv")
    print(f"├── csv_results/{name}_true.csv")