import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=2,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score


def train_and_save_model(model, X, y, path_to_model='./model.pkl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)


def prepare_data(path_to_data='/clean_data/fulldata.csv'):
    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c].copy()

        # creating target
        df_temp.loc[:, 'target'] = df_temp.loc[:, 'temperature'].shift(1)

        # creating features
        for i in range(1, 5):
            df_temp.loc[:, 'temp_m-{}'.format(i)] = df_temp.loc[:, 'temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    return features, target


if __name__ == '__main__':
    # Combining json files to prepare the data
    X, y = prepare_data('./clean_data/fulldata.csv')

    # Instantiate the models
    models = {
        'Linear Regression': LinearRegression(), 
        'Decision Tree Regression': DecisionTreeRegressor(), 
        'Random Forest Regression': RandomForestRegressor()
    }

    # Instantiate the scores dict
    neg_mse_scores = {}

    # Cross-validation of each model
    for key, model in models.items():
        scores = compute_model_score(model, X, y)
        neg_mse_scores[key] = scores.mean()

    # Summary of results
    for model_name, score in neg_mse_scores.items():
        print(f"{model_name}: Mean Negative MSE = {score}")

    # Selection of best model with the highest neg_mse (closest to 0)
    best_model = max(neg_mse_scores, key=neg_mse_scores.get)
    print(f"The best model is {best_model} with a score of {neg_mse_scores[best_model]}")
    train_and_save_model(
                models[best_model],
                X,
                y,
                'clean_data/best_model.pickle'
            )
