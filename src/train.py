"""Training Module"""

import time
import os
from loguru import logger
from matplotlib import pyplot as plt
import seaborn as sns
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, ParameterGrid


def split_data(
    data, target_column="label", test_size=0.2, val_size=0.5, random_state=40
):
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
    # Train-test split (from complete data)
    x_train, x_test, y_train, y_test = train_test_split(
        data.drop(target_column, axis=1), data[target_column], test_size=test_size, random_state=random_state
    )
    data_train = pd.concat([x_train, y_train], axis=1)

    # Validation-test split (from test data)
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=val_size, random_state=random_state
    )
    data_val = pd.concat([x_val, y_val], axis=1)
    data_test = pd.concat([x_test, y_test], axis=1)

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
    fig = go.Figure(
        data=[
            go.Pie(
                values=np.array([len(data_train), len(data_val), len(data_test)]),
                labels=["Training set", "Validation Set", "Test set"],
                hole=0.5,
                textinfo="percent",
                title=" "
            )
        ]
    )
    fig.update_layout(
        height=500,
        width=800,
        showlegend=True,
        title={
            "text": "Comparison of sizes of training set, validation set and test set",
            "x": 0.5, "y": 0.95
        },
    )

    # Save the figure as an image in the specified folder
    image_path = os.path.join(
        save_path, "comparison_train_val_test_sizes_pie_chart.png"
    )
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
    data_train_norm["normalized description"] = data_train["description"].apply(
        text_normalizer
    )
    data_val_norm["normalized description"] = data_val["description"].apply(
        text_normalizer
    )
    data_test_norm["normalized description"] = data_test["description"].apply(
        text_normalizer
    )

    # Add labels
    data_train_norm["label"] = data_train["label"]
    data_val_norm["label"] = data_val["label"]
    data_test_norm["label"] = data_test["label"]

    return data_train_norm, data_val_norm, data_test_norm


# Function to return summary of baseline models
def score(config):
    """
    Trains models and records their training and validation accuracy scores.

    Args:
    config (dict): Configuration dictionary containing training, validation data,
                   model names, models, experiment ID, and current date.

    Returns:
    pd.DataFrame: DataFrame containing classifier names, training accuracy, and validation accuracy.
    """
    results = {
        "Classifier": [],
        "Training accuracy": [],
        "Validation accuracy": []
    }
    start_time = time.time()

    for name, model in zip(config["names"], config["models"]):
        with mlflow.start_run(
            run_name=f"{config['current_date'].strftime('%Y%m%d_%H%m%S')}-ecommerce-{name}",
            experiment_id=config["experiment_id"],
            tags={"version": "v1", "priority": "P1"},
            description="ecommerce text classification",
        ):
            model.fit(config["x_train"], config["y_train"])
            logger.info(f"Trained {name} model")
            train_accuracy = accuracy_score(config["y_train"], model.predict(config["x_train"]))
            val_accuracy = accuracy_score(config["y_val"], model.predict(config["x_val"]))

            results["Classifier"].append(name)
            results["Training accuracy"].append(train_accuracy)
            results["Validation accuracy"].append(val_accuracy)

            logger.info(f"{name} - Training accuracy: {train_accuracy}")
            logger.info(f"{name} - Validation accuracy: {val_accuracy}")

            # Log metrics to mlflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.sklearn.log_model(model, "model")

    score_df = pd.DataFrame(results)
    score_df.sort_values(by="Validation accuracy", ascending=False, inplace=True)

    logger.info(f"Completed scoring in {time.time() - start_time} seconds")

    return score_df


def tune_ridge_classifier(config):
    """
    Tune the RidgeClassifier model using grid search and log the metrics to MLflow.

    Parameters:
    - config (dict): Configuration dictionary containing training and validation data,
                     experiment ID, and current date.

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
    best_params_ridge, best_score_ridge = None, 0

    # Iterate over all combinations of the parameter grid
    for count, g in enumerate(ParameterGrid(params_ridge), 1):
        time_start = time.time()
        logger.info(f"Gridpoint #{count}: {g}")

        # Set the parameters and fit the model
        ridge_classifier.set_params(**g)
        ridge_classifier.fit(config["x_train_tfidf"], config["y_train"])

        # Make predictions and calculate accuracy scores
        score_train = accuracy_score(
            config["y_train"],
            ridge_classifier.predict(config["x_train_tfidf"]))
        score_val = accuracy_score(
            config["y_val"],
            ridge_classifier.predict(config["x_val_tfidf"]))

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
        with mlflow.start_run(
            run_name=f"{config['current_date'].strftime('%Y%m%d_%H%m%S')}-ecommerce-RidgeClassifier_grid_{count}",
            experiment_id=config["experiment_id"],
            tags={"version": "v1", "priority": "P1"},
            description="ecommerce text classification. Fine tuning the RidgeClassifier model",
        ):
            mlflow.log_params(g)
            mlflow.log_metric("train_accuracy", score_train)
            mlflow.log_metric("val_accuracy", score_val)
            mlflow.log_metric("runtime", m * 60 + s)
            mlflow.sklearn.log_model(ridge_classifier, "model")

    # Train the best model with the best parameters
    best_model_tfidf = RidgeClassifier()
    best_model_tfidf.set_params(**best_params_ridge)
    best_model_tfidf.fit(config["x_train_tfidf"], config["y_train"])

    logger.info(f"Best model: {best_model_tfidf}")
    logger.info(f"Best parameters: {best_params_ridge}")
    logger.info(f"Best validation accuracy: {best_score_ridge}")

    # Log the best model to mlflow
    with mlflow.start_run(
        run_name=f"{config['current_date'].strftime('%Y%m%d_%H%m%S')}-ecommerce-Best_RidgeClassifier_Model",
        experiment_id=config["experiment_id"],
        tags={"version": "v1", "priority": "P1"},
        description="ecommerce text classification. Best RidgeClassifier model",
    ):
        mlflow.log_params(best_params_ridge)
        mlflow.log_metric("best_val_accuracy", best_score_ridge)
        mlflow.sklearn.log_model(best_model_tfidf, "best_model")

    return best_model_tfidf, best_params_ridge, best_score_ridge



# Function to compute and print confusion matrix
def conf_mat(config):
    """
    Computes and prints the confusion matrix for the given true and predicted labels.

    Parameters:
    - config (dict): Configuration dictionary containing true labels, predicted labels,
                     and optional parameters for the plot.

    Returns:
    - None: Displays and optionally saves the confusion matrix plot.
    """
    # Compute confusion matrix
    confusion_matrix = metrics.confusion_matrix(config["y_test"], config["y_test_pred"])
    confusion_matrix_df = pd.DataFrame(confusion_matrix, range(4), range(4))

    # Plot confusion matrix
    plt.figure(figsize=config.get("figsize", (10, 8)))
    sns.set(font_scale=config.get("font_scale", 1.2))
    plt.title("Confusion Matrix")
    sns.heatmap(
        confusion_matrix_df,
        annot=True,
        annot_kws={"size": config.get("annot_kws_size", 16)},
        fmt="d"
    )
    plt.yticks([0.5, 1.5, 2.5, 3.5], [0, 1, 2, 3], rotation="vertical")
    plt.xticks([0.5, 1.5, 2.5, 3.5], [0, 1, 2, 3], rotation="horizontal")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)

    # Save the plot if save_path is provided
    if config.get("save_path"):
        plt.savefig(config["save_path"], bbox_inches="tight")
    plt.show()


def evaluate_model(config):
    """
    Evaluate the model on the test set, log results to MLflow, and plot confusion matrix.

    Parameters:
    - config (dict): Configuration dictionary containing all parameters needed for evaluation.
    """
    # Prediction and evaluation on the test set
    logger.info("Starting prediction and evaluation on the test set")
    config["best_model"].fit(config["x_train_vec"], config["y_train"])
    y_test_pred = config["best_model"].predict(config["x_test_vec"])
    score_test = accuracy_score(config["y_test"], y_test_pred)
    logger.info(f"Test accuracy: {score_test}")

    # Log the test accuracy to MLflow
    with mlflow.start_run(
        run_name=f"{config['current_date'].strftime('%Y%m%d_%H%m%S')}-ecommerce-Best_Model_Test_Evaluation",
        experiment_id=config["experiment_id"],
        tags={"version": "v1", "priority": "P1"},
        description="ecommerce text classification. Best model test evaluation",
    ):
        mlflow.log_metric("test_accuracy", score_test)
        mlflow.sklearn.log_model(config["best_model"], "best_model")

    # Plot the confusion matrix and save it as an image
    conf_mat({
        "y_test": config["y_test"],
        "y_test_pred": y_test_pred,
        "save_path": os.path.join(config["save_path"], "confusion_matrix.png")
    })
