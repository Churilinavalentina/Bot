import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
import keras
import ml_edu.experiment
import ml_edu.results
import plotly.express as px

#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730N88/CANDLE_INTERVAL_HOUR-1765180800-1770267600.csv' # Сбер
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730ZJ9/CANDLE_INTERVAL_HOUR-1765195200-1770375600.csv' # ВТБ
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/FUTVTBR03260/CANDLE_INTERVAL_HOUR-1765198800-1770379200.csv' # Фьючерс ВТБ
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004731032/CANDLE_INTERVAL_HOUR-1765198800-1770379200.csv' # Магнит

path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730N88/CANDLE_INTERVAL_1_MIN-1765185840-1770369780.csv' # Сбер
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730ZJ9/CANDLE_INTERVAL_1_MIN-1765561680-1770745620.csv' # ВТБ


def init_df(path_df, delta_t, k):
    # 1. Загрузка данных
    # Предположим, файл называется 'data.csv'
    df = pd.read_csv(path_df)
    df = df[pd.to_datetime(df['time']).dt.time != time(23, 0)]
    df = df.sort_values('time').reset_index(drop=True)

    # 2. Преобразование времени в формат datetime
    df['second'] = pd.to_datetime(df['time']).dt.time.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    df['normal_second_sin'] = np.sin(2 * np.pi * df['second'] / 86400)
    df['normal_second_cos'] = np.cos(2 * np.pi * df['second'] / 86400)

    # 3. Функция для парсинга странной строки в число
    def convert_to_float(val):
        # Превращаем строку "{'units':...}" в реальный словарь Python
        data = ast.literal_eval(val)
        # Считаем итоговое значение: units + nano / 10^9
        return data['units'] + data['nano'] / 1e9

    # Применяем функцию к колонке open
    df['open'] = df['open'].apply(convert_to_float)
    df['normal_price'] = (df['open']-df['open'].shift(1)) / df['open'].shift(1)*100
    df['predict_bool'] = (df['open'].rolling(window=delta_t+1).max().shift(-delta_t)-df['open']) / df['open']*100 > k
    #df['predict_abs'] = (df['open'].rolling(window=delta_t+1).max().shift(-delta_t)-df['open']) / df['open']*100
    #df['predict_max'] = df['open'].rolling(window=delta_t+1).max().shift(-delta_t)
    
    label_columns_features = []
    
    for t in range(0, delta_t):
        df[f'normal_price_{t}'] = df['normal_price'].shift(t)
        label_columns_features.append(f'normal_price_{t}')
        

    return df, label_columns_features

def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple classification model."""
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # Use a Concatenate layer to assemble the different inputs into a single
  # tensor which will be given as input to the Dense layer.
  # For example: [input_1[0][0], input_2[0][0]]

  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  # Call the compile method to transform the layers into a model that
  # Keras can execute.  Notice that we're using a different loss
  # function for classification than for regression.
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model


def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Feed a dataset into the model in order to train it."""

  # The x parameter of keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )


def main(path_df, delta_t, k, learning_rate, number_epochs, batch_size, classification_threshold):
    #Делим датасет на выборки для обучения, валидации метрик и теста
    normalized_dataset, label_columns_features = init_df(path_df, delta_t, k)
    number_samples = len(normalized_dataset)
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    train_data = normalized_dataset.iloc[0:index_80th]
    validation_data = normalized_dataset.iloc[index_80th:index_90th]
    test_data = normalized_dataset.iloc[index_90th:]

    test_data.head()
    
    label_columns = ['predict_bool'] 

    train_features = train_data[label_columns_features]
    train_labels = train_data[label_columns].to_numpy()
    validation_features = validation_data[label_columns_features]
    validation_labels = validation_data[label_columns].to_numpy()
    test_features = test_data[label_columns_features]
    test_labels = test_data[label_columns].to_numpy()
    
    # Let's define our first experiment settings.
    settings = ml_edu.experiment.ExperimentSettings(
        learning_rate=learning_rate,
        number_epochs=number_epochs,
        batch_size=batch_size,
        classification_threshold=classification_threshold,
        input_features=label_columns_features,
    )

    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy', threshold=settings.classification_threshold
        ),
        keras.metrics.Precision(
            name='precision', thresholds=settings.classification_threshold
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model = create_model(settings, metrics)

    # Train the model on the training set.
    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    # Plot metrics vs. epochs
    ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
    ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

if __name__ == "__main__":
    df = init_df(path_df, 50, 0.1)
    print(df)
    main(path_df, 50, 0.1, 0.001, 30, 6000, 0.35)