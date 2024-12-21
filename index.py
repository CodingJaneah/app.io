import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, explained_variance_score, mean_absolute_error
# Add this function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Streamlit App
st.title("ML Model Generator")

# Sidebar for data source selection
st.sidebar.header("Data Source")
st.sidebar.write("Please choose a data source:")
data_source = st.sidebar.radio("Choose Data Source", ("Generate Synthetic Data", "Upload Dataset"))

if data_source == "Generate Synthetic Data":
    st.sidebar.subheader("Synthetic Data Generation")
    problem_type = st.sidebar.selectbox("Select Problem Type", ("Classification", "Regression"))

    if problem_type == "Classification":
        st.sidebar.subheader("Data Generation Parameters")

        feature_names = st.sidebar.text_input(
            "Enter feature names (comma-separated)",
            value="Sepal Length, Sepal Width, Petal Length, Petal Width"
        )

        class_names = st.sidebar.text_input(
            "Enter class names (comma-separated)",
            value="Setosa, Versicolor, Virginica"
        )

        classes = class_names.split(',')

        # Create a dropdown for each class to show its settings
        class_feature_settings = {}
        for class_name in classes:
            class_name = class_name.strip()
            with st.sidebar.expander(f"{class_name} Settings"):
                class_feature_settings[class_name] = {}
                for feature in feature_names.split(','):
                    feature = feature.strip()
                    col1, col2 = st.columns(2)
                    with col1:
                        mean_value = st.number_input(f"Mean for {feature} in {class_name}", min_value=-100.0, max_value=100.0, value=0.0, key=f"mean_{class_name}_{feature}")
                    with col2:
                        std_value = st.number_input(f"Std Dev for {feature} in {class_name}", min_value=0.0, max_value=100.0, value=1.0, key=f"std_{class_name}_{feature}")
                    class_feature_settings[class_name][feature] = {"mean": mean_value, "std": std_value}

        st.info('Please generate data using the sidebar button to view visualizations and results.', icon="ℹ️")

        st.sidebar.subheader("Sample Size & Train/Test Split Configuration")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            num_samples = st.slider("Number of Samples", min_value=500, max_value=50000, value=500, step=1500)
        with col2:
            test_size_percentage = st.slider("Test Size", min_value=10, max_value=50, value=10, step=5)

        # Model selection
        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.radio(
            "Choose a Classification Model",
            ("Logistic Regression", "Decision Trees", "Random Forest", "Gradient Boosting", 
             "Support Vector Machine", "K-Nearest Neighbors", "Naive Bayes", 
             "Linear Discriminant Analysis", "Quadratic Discriminant Analysis")
        )

        load_data_button = st.sidebar.button("Generate Data and Train Models")

        if load_data_button:
            st.subheader("Dataset Split Information")
            test_samples = int(num_samples * test_size_percentage / 100)
            train_samples = num_samples - test_samples
            train_percentage = round((train_samples / num_samples) * 100)
            test_percentage = round((test_samples / num_samples) * 100)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Total Samples**: {num_samples}")
            with col2:
                st.write(f"**Training Samples**: {train_samples} ({train_percentage}%)")
            with col3:
                st.write(f"**Testing Samples**: {test_samples} ({test_percentage}%)")

            with st.spinner("Generating synthetic data and training models. Please wait..."):
                time.sleep(2)

                n_features = len(feature_names.split(','))
                n_informative = max(1, min(n_features, len(classes) - 1))
                n_redundant = max(0, n_features - n_informative - 1)
                n_repeated = max(0, n_features - n_informative - n_redundant)
                max_clusters = 2 ** n_informative
                n_clusters_per_class = max(1, min(max_clusters // len(classes), 1))

                X, y = make_classification(n_samples=num_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=len(classes), n_clusters_per_class=n_clusters_per_class, random_state=42)

                df = pd.DataFrame(X, columns=feature_names.split(','))
                df['Class'] = [classes[i] for i in y]
                st.write("Generated Synthetic Data Preview:")
                st.write(df)

                # Create a download button
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Synthetic Data as CSV",
                    data=csv,
                    file_name='synthetic_data.csv',
                    mime='text/csv'
                )

                st.subheader("Exploratory Data Analysis (EDA)")
                for feature in feature_names.split(','):
                    plt.figure()
                    sns.histplot(df[feature], kde=True)
                    plt.title(f'Distribution of {feature}')
                    st.pyplot(plt)

                st.subheader("Pairplot of Features")
                sns.pairplot(df, hue='Class')
                st.pyplot(plt)

                st.subheader("Correlation Heatmap")
                plt.figure(figsize=(10, 6))
                correlation = df.drop(columns=['Class']).corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
                st.pyplot(plt)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100, random_state=42)

                # Modeling based on user selection
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=200)
                elif model_choice == "Decision Trees":
                    model = DecisionTreeClassifier()
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier()
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif model_choice == "Support Vector Machine":
                    model = SVC()
                elif model_choice == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                elif model_choice == "Naive Bayes":
                    model = GaussianNB()
                elif model_choice == "Linear Discriminant Analysis":
                    model = LinearDiscriminantAnalysis()
                elif model_choice == "Quadratic Discriminant Analysis":
                    model = QuadraticDiscriminantAnalysis()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("Model Performance")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred, target_names=classes))

                # Calculate additional metrics
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Display additional metrics
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                disp.plot(cmap='Blues')
                st.pyplot(plt)

               # Create a mapping from class names to integers
                class_mapping = {name.strip(): i for i, name in enumerate(classes)}

                # Simulation for Classification
                st.subheader("Simulation of Outcomes")

                # Use mean and std from training data to generate realistic simulated data
                means = df[feature_names.split(',')].mean().values
                stds = df[feature_names.split(',')].std().values

                # Generate simulated data based on normal distribution
                simulated_data = np.random.normal(loc=means, scale=stds, size=(500, len(feature_names.split(','))) )  # Increased size for better simulation

                # Ensure simulated data is within a reasonable range
                simulated_data = np.clip(simulated_data, -100, 100)  # Adjust these limits as necessary

                # Simulated Predictions
                simulated_predictions = model.predict(simulated_data)

                # Create a DataFrame for the simulated data and predictions
                simulated_df = pd.DataFrame(simulated_data, columns=feature_names.split(','))
                simulated_df['Predicted Class'] = simulated_predictions

                # Display the simulated outcomes
                st.write("Simulated Outcomes:")
                st.write(simulated_df)

                # Create synthetic true classes for the simulated data for comparison
                synthetic_true_classes = np.random.choice(classes, size=simulated_predictions.shape[0])

                # Convert synthetic true classes to their integer representations
                synthetic_true_classes_int = np.array([class_mapping[name.strip()] for name in synthetic_true_classes])

                # Calculate metrics
                accuracy = accuracy_score(synthetic_true_classes_int, simulated_predictions)
                precision = precision_score(synthetic_true_classes_int, simulated_predictions, average='weighted')
                recall = recall_score(synthetic_true_classes_int, simulated_predictions, average='weighted')
                f1 = f1_score(synthetic_true_classes_int, simulated_predictions, average='weighted')

                # Display metrics
                st.subheader("Model Performance on Simulated Data")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Confusion Matrix for Simulated Outcomes
                st.subheader("Confusion Matrix for Simulated Outcomes")
                cm_simulation = confusion_matrix(synthetic_true_classes_int, simulated_predictions, labels=range(len(classes)))
                disp_simulation = ConfusionMatrixDisplay(confusion_matrix=cm_simulation, display_labels=classes)
                disp_simulation.plot(cmap='Blues')
                st.pyplot(plt)

    else:  # Regression
        st.sidebar.subheader("Data Generation Parameters")

        feature_names = st.sidebar.text_input(
            "Enter feature names (comma-separated)",
            value="Feature 1, Feature 2, Feature 3"
        )

        st.sidebar.subheader("Sample Size & Noise Configuration")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            num_samples = st.slider("Number of Samples", min_value=500, max_value=50000, value=500, step=1500)
        with col2:
            noise_level = st.slider("Noise Level", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        # Model selection
        st.sidebar.subheader("Model Selection")
        model_choice = st.sidebar.radio(
            "Choose a Regression Model",
            ("Linear Regression", "Ridge Regression", "Lasso Regression", 
             "Decision Tree Regressor", "Random Forest Regressor", 
             "Gradient Boosting Regressor", "Support Vector Regressor")
        )

        load_data_button = st.sidebar.button("Generate Data and Train Models")

        if load_data_button:
            with st.spinner("Generating synthetic data and training models. Please wait..."):
                time.sleep(2)

                n_features = len(feature_names.split(','))
                X, y = make_regression(n_samples=num_samples, n_features=n_features, noise=noise_level, random_state=42)

                df = pd.DataFrame(X, columns=feature_names.split(','))
                df['Target'] = y
                st.write("Generated Synthetic Data Preview:")
                st.write(df)

                # Create a download button
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="Download Synthetic Data as CSV",
                    data=csv,
                    file_name='synthetic_data.csv',
                    mime='text/csv'
                )

                # EDA for Regression
                plt.figure()
                plt.scatter(df[feature_names.split(',')[0]], df['Target'], alpha=0.5, color='blue', marker='o', s=50)
                plt.title('Feature 1 vs Target')
                plt.xlabel(feature_names.split(',')[0])
                plt.ylabel('Target')
                st.pyplot(plt)

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Modeling based on user selection
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Ridge Regression":
                    model = Ridge()
                elif model_choice == "Lasso Regression":
                    model = Lasso()
                elif model_choice == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif model_choice == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_choice == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor()
                elif model_choice == "Support Vector Regressor":
                    model = SVR()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Model performance for Regression
                st.subheader("Model Performance")

                # Assuming y_test and y_pred are defined
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                explained_variance = explained_variance_score(y_test, y_pred)

                # Mean Absolute Percentage Error - MAPE
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Display the metrics
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"R² Score: {r2:.2f}")
                st.write(f"Explained Variance Score: {explained_variance:.2f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

                # Visualizing predictions vs actual values
                plt.figure()
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                plt.title('Actual vs Predicted')
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                st.pyplot(plt)

elif data_source == "Upload Dataset":
    st.sidebar.subheader("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV, Excel, etc.)", type=["csv", "xlsx"])

    # Check if no file is uploaded
    if uploaded_file is None:
        st.warning("Please upload a dataset to proceed.")
    else:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.sidebar.write("Dataset Preview:", df.head())
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            st.sidebar.write("Dataset Preview:", df.head())

        test_size_percentage = st.sidebar.slider("Test Size", min_value=10, max_value=50, value=10, step=10)

        problem_type = st.sidebar.selectbox("Select Problem Type", ("Classification", "Regression"))

    # Model selection based on problem type
        if problem_type == "Classification":
                st.sidebar.subheader("Select Classification Model")
                model_choice = st.sidebar.radio(
                    "Choose a Classification Model",
                    ("Logistic Regression", "Decision Trees", "Random Forest", "Gradient Boosting", 
                    "Support Vector Machine", "K-Nearest Neighbors", "Naive Bayes", 
                    "Linear Discriminant Analysis", "Quadratic Discriminant Analysis")
                )
        else:  # Regression
                st.sidebar.subheader("Select Regression Model")
                model_choice = st.sidebar.radio(
                    "Choose a Regression Model",
                    ("Linear Regression", "Ridge Regression", "Lasso Regression", 
                    "Decision Tree Regressor", "Random Forest Regressor", 
                    "Gradient Boosting Regressor", "Support Vector Regressor")
                )

                # Ensure only numeric columns are selected
                numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

                if numeric_df.shape[1] < 2:  # At least one feature and one target
                    st.error("The dataset must contain at least one feature and one target variable.")
                else:
                    target_col = st.sidebar.selectbox("Select Target Variable", numeric_df.columns)

                    # Remove target column from features
                    X = numeric_df.drop(columns=[target_col]).values
                    y = numeric_df[target_col].values

        train_button = st.sidebar.button("Train Models")

        with st.spinner("Training uploaded dataset. Please wait..."):
                    time.sleep(2)

        if train_button:
                if problem_type == "Classification":
                    # Assume the last column is the target variable for classification
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100, random_state=42)

                    # Model selection
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=200)
                    elif model_choice == "Decision Trees":
                        model = DecisionTreeClassifier()
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier()
                    elif model_choice == "Gradient Boosting":
                        model = GradientBoostingClassifier()
                    elif model_choice == "Support Vector Machine":
                        model = SVC()
                    elif model_choice == "K-Nearest Neighbors":
                        model = KNeighborsClassifier()
                    elif model_choice == "Naive Bayes":
                        model = GaussianNB()
                    elif model_choice == "Linear Discriminant Analysis":
                        model = LinearDiscriminantAnalysis()
                    elif model_choice == "Quadratic Discriminant Analysis":
                        model = QuadraticDiscriminantAnalysis()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Display results (similar to your synthetic data section)
                    st.subheader("Model Performance")
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap='Blues')
                    st.pyplot(plt)

                else:  # Regression
                    # Assume the last column is the target variable for regression
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100, random_state=42)

                    # Model selection
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    elif model_choice == "Ridge Regression":
                        model = Ridge()
                    elif model_choice == "Lasso Regression":
                        model = Lasso()
                    elif model_choice == "Decision Tree Regressor":
                        model = DecisionTreeRegressor()
                    elif model_choice == "Random Forest Regressor":
                        model = RandomForestRegressor()
                    elif model_choice == "Gradient Boosting Regressor":
                        model = GradientBoostingRegressor()
                    elif model_choice == "Support Vector Regressor":
                        model = SVR()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Model performance for Regression
                    st.subheader("Model Performance")
                    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

                    # Visualizing predictions vs actual values
                    plt.figure()
                    plt.scatter(y_test, y_pred, alpha=0.5)
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                    plt.title('Actual vs Predicted')
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    st.pyplot(plt)
