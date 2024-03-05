# Linear Regression Car Price Prediction

## Project Overview
This project demonstrates a simple application of Linear Regression to predict the price of a car based on its mileage. It utilizes a CSV file containing data on car prices as a function of their mileage. There are two main programs in this project:
- `train.py`: Trains the model using the provided dataset. It visualizes the trained model as a line within a scatter plot of the data points. This program also saves the trained model along with the necessary data normalization parameters.
- `predict.py`: Asks the user to input a car's mileage and predicts its price, which is then displayed in the console.

## Installation

Before running the project, you need to install the required Python libraries. This project depends on `matplotlib` for data visualization and `numpy` for numerical computations. You can install these dependencies by running the following command:

```bash
pip install matplotlib numpy
```

## Running the Programs
# Training the Model
To train the model, navigate to the project directory in your terminal and run:
```bash
python train.py
```
This will start the training process, display the model as a line in a scatter plot representing the dataset, and save the model and normalization parameters.

# Predicting Car Prices
After training the model, you can predict the price of a car by running:
```bash
python predict.py
```
You will be prompted to enter the mileage of the car. After inputting the mileage, the program will display the predicted price in the console.

## Dataset
The dataset used in this project is a CSV file containing car prices as a function of their mileage. Ensure that this file is located in the same directory as the scripts for the program to function correctly.

## Contributing
Feel free to fork this project and submit pull requests with improvements or contact me if you have any questions or suggestions.
