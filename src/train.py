"""Training Module"""

import time
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
import os
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import ParameterGrid
import mlflow


def split_data(data, target_column="label", test_size=0.2, val_size=0.5, random_state=40):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
    data (pd.DataFrame): The input dataset containing features and target column.
    target_column (str): The name of the target column in the dataset. Default is "label".
    test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
    val_size (float): The proportion of the test set to include in the validation split. Default is 0.5.
    random_state (int): The random seed for reproducibility. Default is 40.

    Returns:
    data_train (pd.DataFrame): Training set including features and target.
    data_val (pd.DataFrame): Validation set including features and target.
    data_test (pd.DataFrame): Test set including features and target.
    """
    # Feature-target split
    X, y = data.drop(target_column, axis=1), data[target_column]

    # Train-test split (from complete data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    data_train = pd.concat([X_train, y_train], axis=1)

    # Validation-test split (from test data)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=val_size, random_state=random_state
    )
    data_val = pd.concat([X_val, y_val], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)

    return data_train, data_val, data_test


def plot_data_split_sizes(data_train, data_val, data_test, save_path):
    """
    Plots a pie chart comparing the sizes of training, validation, and test sets.

    Args:
    data_train (pd.DataFrame): Training set including features and target.
    data_val (pd.DataFrame): Validation set including features and target.
    data_test (pd.DataFrame): Test set including features and target.
    save_path (str): The folder where the image will be saved.
    """
    # Comparison of sizes of training set, validation set, and test set
    values = np.array([len(data_train), len(data_val), len(data_test)])
    labels = ["Training set", "Validation Set", "Test set"]
    fig = go.Figure(
        data=[go.Pie(values=values, labels=labels, hole=0.5, textinfo="percent", title=" ")]
    )
    text_title = "Comparison of sizes of training set, validation set and test set"
    fig.update_layout(
        height=500, width=800, showlegend=True, title=dict(text=text_title, x=0.5, y=0.95)
    )

    # Save the figure as an image in the specified folder
    image_path = os.path.join(save_path, 'comparison_train_val_test_sizes_pie_chart.png')
    fig.write_image(image_path)

    fig.show()


def normalize_split_data(data_train, data_val, data_test, text_normalizer):
    """
    Apply text normalization to the train, validation, and test datasets.
    
    Parameters:
    - data_train (pd.DataFrame): Training dataset with 'description' column.
    - data_val (pd.DataFrame): Validation dataset with 'description' column.
    - data_test (pd.DataFrame): Test dataset with 'description' column.
    - text_normalizer (function): Function to apply text normalization.

    Returns:
    - data_train_norm (pd.DataFrame): Normalized training dataset.
    - data_val_norm (pd.DataFrame): Normalized validation dataset.
    - data_test_norm (pd.DataFrame): Normalized test dataset.
    """
    # Initialize empty DataFrames for normalized data
    data_train_norm, data_val_norm, data_test_norm = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    # Apply text normalization
    data_train_norm["normalized description"] = data_train["description"].apply(text_normalizer)
    data_val_norm["normalized description"] = data_val["description"].apply(text_normalizer)
    data_test_norm["normalized description"] = data_test["description"].apply(text_normalizer)

    # Add labels
    data_train_norm["label"] = data_train["label"]
    data_val_norm["label"] = data_val["label"]
    data_test_norm["label"] = data_test["label"]

    return data_train_norm, data_val_norm, data_test_norm


# Function to return summary of baseline models
def score(X_train, y_train, X_val, y_val, names, models, experiment_id, current_date):
    """
    Trains and evaluates a list of models, logs metrics to MLflow, and returns a DataFrame with model performance.

    Parameters:
    - X_train (pd.DataFrame or np.array): Training feature data.
    - y_train (pd.Series or np.array): Training labels.
    - X_val (pd.DataFrame or np.array): Validation feature data.
    - y_val (pd.Series or np.array): Validation labels.
    - names (list of str): List of model names.
    - models (list of sklearn.base.ClassifierMixin): List of model instances to evaluate.
    - experiment_id (int): MLflow experiment ID for logging.
    - current_date (datetime): Current date for naming MLflow runs.

    Returns:
    - pd.DataFrame: DataFrame containing model names and their training and validation accuracies.
    """
    score_df, score_train, score_val = pd.DataFrame(), [], []
    start_time = time.time()
    
    for name, model in zip(names, models):
        with mlflow.start_run(run_name=f"{current_date.strftime('%Y%m%d_%H%m%S')}-ecommerce-{name}",
                            experiment_id=experiment_id,
                            tags={"version": "v1", "priority": "P1"},
                            description="ecommerce text classification",
                            ) as mlf_run:
            model.fit(X_train, y_train)
            logger.info(f"Trained {name} model")

            y_train_pred, y_val_pred = model.predict(X_train), model.predict(X_val)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            score_train.append(train_accuracy)
            score_val.append(val_accuracy)

            logger.info(f"{name} - Training accuracy: {train_accuracy}")
            logger.info(f"{name} - Validation accuracy: {val_accuracy}")

            # Log metrics to mlflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.sklearn.log_model(model, "model")

    score_df["Classifier"] = names
    score_df["Training accuracy"] = score_train
    score_df["Validation accuracy"] = score_val
    score_df.sort_values(by="Validation accuracy", ascending=False, inplace=True)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed scoring in {elapsed_time} seconds")
    
    return score_df



def tune_ridge_classifier(X_train_tfidf, y_train, X_val_tfidf, y_val, experiment_id, current_date):
    """
    Tune the RidgeClassifier model using grid search and log the metrics to MLflow.
    
    Parameters:
    - X_train_tfidf (pd.DataFrame): Training feature set in TF-IDF format.
    - y_train (pd.Series): Training labels.
    - X_val_tfidf (pd.DataFrame): Validation feature set in TF-IDF format.
    - y_val (pd.Series): Validation labels.
    - experiment_id (int): MLflow experiment ID.
    - current_date (datetime): Current date to format MLflow run names.

    Returns:
    - best_model_tfidf (RidgeClassifier): Best RidgeClassifier model found.
    - best_params_ridge (dict): Best parameters for the RidgeClassifier.
    - best_score_ridge (float): Best validation accuracy score.
    """
    # Define the model and the adjusted parameter grid
    ridge_classifier = RidgeClassifier()
    params_ridge = {
        "alpha": [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "solver": ["auto"],
    }

    # Initialize variables to track the best model, parameters, and score
    best_model_ridge, best_params_ridge, best_score_ridge = None, None, 0

    # Iterate over all combinations of the parameter grid
    for count, g in enumerate(ParameterGrid(params_ridge), 1):
        time_start = time.time()
        logger.info(f"Gridpoint #{count}: {g}")

        # Set the parameters and fit the model
        ridge_classifier.set_params(**g)
        ridge_classifier.fit(X_train_tfidf, y_train)

        # Make predictions and calculate accuracy scores
        y_train_pred = ridge_classifier.predict(X_train_tfidf)
        y_val_pred = ridge_classifier.predict(X_val_tfidf)
        score_train = accuracy_score(y_train, y_train_pred)
        score_val = accuracy_score(y_val, y_val_pred)

        # Calculate runtime
        time_stop = time.time()
        m, s = divmod(int(time_stop - time_start), 60)
        logger.info(
            f"Training accuracy: {score_train}, Validation accuracy: {score_val}, Runtime: {m}m{s}s"
        )

        # Update the best parameters if the current model is better
        if score_val > best_score_ridge:
            best_params_ridge, best_score_ridge = g, score_val

        # Log metrics to mlflow
        with mlflow.start_run(run_name=f"{current_date.strftime('%Y%m%d_%H%m%S')}-ecommerce-RidgeClassifier_grid_{count}", 
                                experiment_id=experiment_id,
                                tags={"version": "v1", "priority": "P1"},
                                description="ecommerce text classification. Fine tuning the RidgeClassifier model",
                                ) as mlf_run:
            mlflow.log_params(g)
            mlflow.log_metric("train_accuracy", score_train)
            mlflow.log_metric("val_accuracy", score_val)
            mlflow.log_metric("runtime", m * 60 + s)
            mlflow.sklearn.log_model(ridge_classifier, "model")

    # Train the best model with the best parameters
    best_model_tfidf = RidgeClassifier()
    best_model_tfidf.set_params(**best_params_ridge)
    best_model_tfidf.fit(X_train_tfidf, y_train)

    logger.info(f"Best model: {best_model_tfidf}")
    logger.info(f"Best parameters: {best_params_ridge}")
    logger.info(f"Best validation accuracy: {best_score_ridge}")

    # Log the best model to mlflow
    with mlflow.start_run(run_name=f"{current_date.strftime('%Y%m%d_%H%m%S')}-ecommerce-Best_RidgeClassifier_Model",
                        experiment_id=experiment_id,
                        tags={"version": "v1", "priority": "P1"},
                        description="ecommerce text classification. Best RidgeClassifier model",
                        ) as mlf_run:
        mlflow.log_params(best_params_ridge)
        mlflow.log_metric("best_val_accuracy", best_score_ridge)
        mlflow.sklearn.log_model(best_model_tfidf, "best_model")

    return best_model_tfidf, best_params_ridge, best_score_ridge


# Function to compute and print confusion matrix
def conf_mat(y_test, y_test_pred, figsize=(10, 8), font_scale=1.2, annot_kws_size=16, save_path=None):
    """
    Computes and prints the confusion matrix for the given true and predicted labels.

    Parameters:
    - y_test (array-like): True labels of the test set.
    - y_test_pred (array-like): Predicted labels by the model.
    - figsize (tuple, optional): Size of the figure to be displayed. Default is (10, 8).
    - font_scale (float, optional): Scaling factor for font size in the heatmap. Default is 1.2.
    - annot_kws_size (int, optional): Font size for annotations in the heatmap. Default is 16.
    - save_path (str, optional): Path to save the confusion matrix plot as an image file. If None, the plot will not be saved.

    Returns:
    - None: Displays and optionally saves the confusion matrix plot.
    """
    class_names = [
        0,
        1,
        2,
        3,
    ]  # ['Electronics', 'Household', 'Books', 'Clothing & Accessories']
    tick_marks_y = [0.5, 1.5, 2.5, 3.5]
    tick_marks_x = [0.5, 1.5, 2.5, 3.5]
    confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
    confusion_matrix_df = pd.DataFrame(confusion_matrix, range(4), range(4))
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)  # label size
    plt.title("Confusion Matrix")
    sns.heatmap(
        confusion_matrix_df, annot=True, annot_kws={"size": annot_kws_size}, fmt="d"
    )  # font size
    plt.yticks(tick_marks_y, class_names, rotation="vertical")
    plt.xticks(tick_marks_x, class_names, rotation="horizontal")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    

def evaluate_model(best_model, X_train_vec, y_train, X_test_vec, y_test, experiment_id, current_date, conf_mat, save_path):
    """
    Evaluate the model on the test set, log results to MLflow, and plot confusion matrix.

    Parameters:
    - best_model (sklearn.base.ClassifierMixin): The trained model to be evaluated.
    - X_train_vec (pd.DataFrame or np.array): Training features.
    - y_train (pd.Series or np.array): Training labels.
    - X_test_vec (pd.DataFrame or np.array): Test features.
    - y_test (pd.Series or np.array): Test labels.
    - experiment_id (int): MLflow experiment ID.
    - current_date (datetime): Current date to format MLflow run names.
    - conf_mat (function): Function to plot the confusion matrix.
    """
    # Prediction and evaluation on the test set
    logger.info("Starting prediction and evaluation on the test set")
    best_model.fit(X_train_vec, y_train)
    y_test_pred = best_model.predict(X_test_vec)
    score_test = accuracy_score(y_test, y_test_pred)
    logger.info(f"Test accuracy: {score_test}")

    # Log the test accuracy to MLflow
    with mlflow.start_run(run_name=f"{current_date.strftime('%Y%m%d_%H%m%S')}-ecommerce-Best_Model_Test_Evaluation", 
                            experiment_id=experiment_id,
                            tags={"version": "v1", "priority": "P1"},
                            description="ecommerce text classification. Best model test evaluation") as mlf_run:
        mlflow.log_metric("test_accuracy", score_test)
        mlflow.sklearn.log_model(best_model, "best_model")

    # Plot the confusion matrix and save it as an image
    
    # Save the figure as an image in the specified folder    
    conf_matrix_path = os.path.join(save_path, 'confusion_matrix.png')
    conf_mat(y_test, y_test_pred, figsize=(10, 8), font_scale=1.2, annot_kws_size=16, save_path=conf_matrix_path)