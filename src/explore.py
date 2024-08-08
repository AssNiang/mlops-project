"""Exploration Module"""

import os
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns


def split_by_label(data):
    """
    Function to split the dataset by label into separate DataFrames.

    Args:
    data (pd.DataFrame): Input dataframe with a 'label' column.

    Returns:
    tuple: Four DataFrames corresponding to each label (Electronics, Household, Books, Clothing & Accessories).
    """
    data_e = data[data["label"] == 0]  # Electronics
    data_h = data[data["label"] == 1]  # Household
    data_b = data[data["label"] == 2]  # Books
    data_c = data[data["label"] == 3]  # Clothing & Accessories

    return data_e, data_h, data_b, data_c


def visualize_and_save_class_frequencies(data_e, data_h, data_b, data_c, save_path):
    """
    Function to visualize the class frequencies as a pie chart and save the figure as an image.

    Args:
    data_e (pd.DataFrame): DataFrame for Electronics.
    data_h (pd.DataFrame): DataFrame for Household.
    data_b (pd.DataFrame): DataFrame for Books.
    data_c (pd.DataFrame): DataFrame for Clothing & Accessories.
    save_path (str): Directory path where the image will be saved. Default is "../reports".

    Returns:
    None
    """
    fig = go.Figure(
        data=[
            go.Pie(
                values=np.array([len(data_e), len(data_h), len(data_b), len(data_c)]),
                labels=["Electronics", "Household", "Books", "Clothing & Accessories"],
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
            "text": "Comparison of class frequencies",
            "x": 0.5, "y": 0.95
        },
    )

    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the figure as an image in the specified folder
    image_path = os.path.join(save_path, "class_frequencies_pie_chart.png")
    logger.info(f"Saving figure in {image_path}")
    fig.write_image(image_path)

    # Show the figure
    fig.show()


def visualize_and_save_character_distribution(
    data_e, data_h, data_b, data_c, save_path
):
    """
    Function to visualize the distribution of the number of characters in descriptions and save the figure as an image.

    Args:
    data_e (pd.DataFrame): DataFrame for Electronics.
    data_h (pd.DataFrame): DataFrame for Household.
    data_b (pd.DataFrame): DataFrame for Books.
    data_c (pd.DataFrame): DataFrame for Clothing & Accessories.
    save_path (str): Directory path where the image will be saved. Default is "../reports".

    Returns:
    None
    """
    # Distribution of number of characters in description
    data_e_char = data_e["description"].str.len()
    data_h_char = data_h["description"].str.len()
    data_b_char = data_b["description"].str.len()
    data_c_char = data_c["description"].str.len()

    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4), sharey=False)
    sns.histplot(x=data_e_char, bins=20, ax=ax[0, 0]).set_title("Class: Electronics")
    sns.histplot(x=data_h_char, bins=20, ax=ax[0, 1]).set_title("Class: Household")
    sns.histplot(x=data_b_char, bins=20, ax=ax[1, 0]).set_title("Class: Books")
    sns.histplot(x=data_c_char, bins=20, ax=ax[1, 1]).set_title(
        "Class: Clothing & Accessories"
    )

    fig.suptitle("Distribution of number of characters in description")
    for i in range(4):
        if i // 2 == 0:
            ax[i // 2, i % 2].set_xlabel(" ")
        else:
            ax[i // 2, i % 2].set_xlabel("Number of characters")
        if i % 2 != 0:
            ax[i // 2, i % 2].set_ylabel(" ")

    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the figure as an image in the specified folder
    image_path = os.path.join(save_path, "distribution_characters.png")
    fig.savefig(image_path, bbox_inches="tight")

    # Show the figure
    plt.show()


def visualize_and_save_word_distribution(data_e, data_h, data_b, data_c, save_path):
    """
    Function to visualize the distribution of the number of words in descriptions and save the figure as an image.

    Args:
    data_e (pd.DataFrame): DataFrame for Electronics.
    data_h (pd.DataFrame): DataFrame for Household.
    data_b (pd.DataFrame): DataFrame for Books.
    data_c (pd.DataFrame): DataFrame for Clothing & Accessories.
    save_path (str): Directory path where the image will be saved. Default is "../reports".

    Returns:
    None
    """
    # Distribution of number of words in description
    data_e_word = data_e["description"].str.split().str.len()
    data_h_word = data_h["description"].str.split().str.len()
    data_b_word = data_b["description"].str.split().str.len()
    data_c_word = data_c["description"].str.split().str.len()


    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4), sharey=False)
    sns.histplot(x=data_e_word, bins=20, ax=ax[0, 0]).set_title("Class: Electronics")
    sns.histplot(x=data_h_word, bins=20, ax=ax[0, 1]).set_title("Class: Household")
    sns.histplot(x=data_b_word, bins=20, ax=ax[1, 0]).set_title("Class: Books")
    sns.histplot(x=data_c_word, bins=20, ax=ax[1, 1]).set_title(
        "Class: Clothing & Accessories"
    )

    fig.suptitle("Distribution of number of words in description")
    for i in range(4):
        if i // 2 == 0:
            ax[i // 2, i % 2].set_xlabel(" ")
        else:
            ax[i // 2, i % 2].set_xlabel("Number of words")
        if i % 2 != 0:
            ax[i // 2, i % 2].set_ylabel(" ")

    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the figure as an image in the specified folder
    image_path = os.path.join(save_path, "distribution_words.png")
    fig.savefig(image_path, bbox_inches="tight")

    # Show the figure
    plt.show()


def visualize_and_save_avg_word_length_distribution(
    data_e, data_h, data_b, data_c, save_path
):
    """
    Function to visualize the distribution of average word-length in descriptions and save the figure as an image.

    Args:
    data_e (pd.DataFrame): DataFrame for Electronics.
    data_h (pd.DataFrame): DataFrame for Household.
    data_b (pd.DataFrame): DataFrame for Books.
    data_c (pd.DataFrame): DataFrame for Clothing & Accessories.
    save_path (str): Directory path where the image will be saved. Default is "../reports".

    Returns:
    None
    """
    def average_word_length(description):
        words = description.split()
        return np.mean([len(word) for word in words])

    # Distribution of average word-length in description
    data_e_avg = data_e["description"].apply(average_word_length)
    data_h_avg = data_h["description"].apply(average_word_length)
    data_b_avg = data_b["description"].apply(average_word_length)
    data_c_avg = data_c["description"].apply(average_word_length)


    fig, ax = plt.subplots(2, 2, figsize=(10, 8.4), sharey=False)
    sns.histplot(x=data_e_avg, bins=20, ax=ax[0, 0]).set_title("Class: Electronics")
    sns.histplot(x=data_h_avg, bins=20, ax=ax[0, 1]).set_title("Class: Household")
    sns.histplot(x=data_b_avg, bins=20, ax=ax[1, 0]).set_title("Class: Books")
    sns.histplot(x=data_c_avg, bins=20, ax=ax[1, 1]).set_title(
        "Class: Clothing & Accessories"
    )

    fig.suptitle("Distribution of average word-length in description")
    for i in range(4):
        if i // 2 == 0:
            ax[i // 2, i % 2].set_xlabel(" ")
        else:
            ax[i // 2, i % 2].set_xlabel("Average word-length")
        if i % 2 != 0:
            ax[i // 2, i % 2].set_ylabel(" ")

    # Ensure the save_path directory exists
    os.makedirs(save_path, exist_ok=True)
    # Save the figure as an image in the specified folder
    image_path = os.path.join(save_path, "distribution_avg_word_length.png")
    fig.savefig(image_path, bbox_inches="tight")

    # Show the figure
    plt.show()
