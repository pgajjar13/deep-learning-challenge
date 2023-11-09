# deep-learning-challenge

Module 21 Challenge

**Background**

The nonprofit foundation, Alphabet Soup, is seeking a tool that can assist in selecting the most promising applicants for funding, increasing their chances of success in their ventures. Leveraging your expertise in machine learning and neural networks, your task is to use the provided dataset's features to create a binary classifier capable of predicting whether applicants will achieve success if funded by Alphabet Soup.

From Alphabet Soup's business team, you've received a CSV file containing data on over 34,000 organizations that have received funding from Alphabet Soup throughout the years. This dataset includes various columns capturing metadata about each organization, such as:

- EIN and NAME: Identification columns
- APPLICATION_TYPE: Alphabet Soup application type
- AFFILIATION: Affiliated sector of industry
- CLASSIFICATION: Government organization classification
- USE_CASE: Use case for funding
- ORGANIZATION: Organization type
- STATUS: Active status
- INCOME_AMT: Income classification
- SPECIAL_CONSIDERATIONS: Special considerations for application
- ASK_AMT: Funding amount requested
- IS_SUCCESSFUL: Was the money used effectively

**Instructions**

**Step 1: Preprocess the Data**

Using your knowledge of Pandas and scikit-learn's `StandardScaler()`, preprocess the dataset to prepare for Step 2, where you will compile, train, and evaluate the neural network model.

1. Begin by uploading the starter file to Google Colab and follow the instructions provided in the Challenge files to complete the preprocessing steps.
2. Read in the 'charity_data.csv' to a Pandas DataFrame and identify the following in your dataset:
   - Identify the target variable(s) for your model.
   - Identify the feature variable(s) for your model.
   - Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns with more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to select a cutoff point for grouping "rare" categorical variables together in a new value, such as 'Other'. Verify the success of this binning process.
6. Use `pd.get_dummies()` to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Utilize these arrays and the `train_test_split` function to divide the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, and then using the `transform` function.

**Step 2: Compile, Train, and Evaluate the Model**

Utilize your knowledge of TensorFlow to design a neural network for binary classification. Your model should predict if an Alphabet Soup-funded organization will be successful based on the dataset features. Consider the number of input features before determining the number of neurons and layers in your model. Afterward, compile, train, and evaluate your binary classification model to compute the model's loss and accuracy.

1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by specifying the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and select an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file 'AlphabetSoupCharity.h5'.

**Step 3: Optimize the Model**

Using your TensorFlow knowledge, optimize the model to achieve a predictive accuracy higher than 72%. Implement any or all of the following methods to optimize your model:

- Adjust the input data to ensure that no variables or outliers are causing confusion in the model (e.g., dropping more or fewer columns, creating more bins for rare occurrences, increasing or decreasing the number of values for each bin).
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs in the training regimen.

Create a new Google Colab file and name it 'AlphabetSoupCharity.ipynb'.

1. Import your dependencies and read in the 'charity_data.csv' to a Pandas DataFrame.
2. Preprocess the dataset as you did in Step 1, accounting for any modifications resulting from optimizing the model.
3. Design a neural network model, adjusting for modifications to optimize the model for higher than 72% accuracy.
4. Save and export your results to an HDF5 file.

---