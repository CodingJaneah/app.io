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

# Add team member section here
st.sidebar.header("About the Team")
st.sidebar.write("Team Members:")
st.sidebar.write("Mariana Jane L. Ramos")
st.sidebar.write("Anne Louis Tampoco")
st.sidebar.write("Ellyza Mae Periabras")
st.sidebar.write("Derick Emmanuel Marpuri")

# Add a horizontal line
st.sidebar.markdown("---")

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
                        mean_value = st.number_input(f"Mean for {feature} in {class_name}", min_value=-100.0, max_value=500.0, value=0.0, key=f"mean_{class_name}_{feature}")
                    with col2:
                        std_value = st.number_input(f"Std Dev for {feature} in {class_name}", min_value=0.0, max_value=500.0, value=1.0, key=f"std_{class_name}_{feature}")
                    class_feature_settings[class_name][feature] = {"mean": mean_value, "std": std_value}

        st.info('Please generate data using the sidebar button to view visualizations and results.', icon="ℹ️")

        st.sidebar.subheader("Sample Size & Train/Test Split Configuration")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            num_samples = st.slider("Number of Samples", min_value=500, max_value=50000, value=500, step=1500)
        with col2:
            test_size_percentage = st.slider("Test Size", min_value=10, max_value=50, value=10, step=5)

            # Calculate the training size percentage
            train_size_percentage = 100 - test_size_percentage
            
            # Display the text with the calculated percentages
            st.write(f"Test: {test_size_percentage}% / Train: {train_size_percentage}%")

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

                 # Dictionary to store model performance
                model_performance = []

                # List of models to train for classification
                classification_models = {
                    "Logistic Regression": LogisticRegression(max_iter=200),
                    "Decision Trees": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Gradient Boosting": GradientBoostingClassifier(),
                    "Support Vector Machine": SVC(),
                    "K-Nearest Neighbors": KNeighborsClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                }

                # After training the models and predicting, store the confusion matrices
                confusion_matrices = []

                # Train models and evaluate as well as iteration through each model 
                for model_name, model in classification_models.items():
                    start_time = time.time()  # Start timer
                    model.fit(X_train, y_train)  # Train model
                    y_pred = model.predict(X_test)  # Predict on test set

                    # Calculate performance metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)

                    # Store performance metrics
                    model_performance.append({
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": report['weighted avg']['precision'],
                        "Recall": report['weighted avg']['recall'],
                        "F1-Score": report['weighted avg']['f1-score'],
                        "Training Time (s)": time.time() - start_time,
                        "Status": "Success"
                    })

                    # Create confusion matrix and store it
                    cm = confusion_matrix(y_test, y_pred)
                    confusion_matrices.append((model_name, cm))

                            
                # Convert to DataFrame
                comparison_df = pd.DataFrame(model_performance)

                # Sort the DataFrame by Accuracy in descending order
                comparison_df = comparison_df.sort_values(by="Accuracy", ascending=False)

                    # Save to session state
                st.session_state.comparison_df = comparison_df

                # Identify the best model
                best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
                best_model = comparison_df.loc[comparison_df['Model'] == best_model_name]

                # Display results for the best model
                st.subheader("Best Model: {}".format(best_model_name))
                st.write("Accuracy: {:.2f}".format(best_model["Accuracy"].values[0]))
                st.write("Classification Report:")
                classification_report_df = pd.DataFrame(report).transpose()
                st.table(classification_report_df)

                # Display Model Comparison Table right after the classification report
                st.subheader("Model Comparison")
                st.table(comparison_df)

                # Create a summary DataFrame for visualization
                if problem_type == "Classification":
                    metrics_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
                    metrics_df.set_index('Model', inplace=True)
                else:  # Regression
                    metrics_df = comparison_df[['Model', 'RMSE', 'R² Score', 'MAE']]
                    metrics_df.set_index('Model', inplace=True)

    
                    # Alternatively, use Plotly for interactive charts
                import plotly.express as px

                # Melt the DataFrame for Plotly
                metrics_melted = metrics_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')

                # Create a bar chart with Plotly
                fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group',
                            title='Model Performance Metrics Comparison')
                st.plotly_chart(fig)


                 # Display confusion matrices in a 3-column format
                num_models = len(confusion_matrices)
                columns = st.columns(3)

                for index, (model_name, cm) in enumerate(confusion_matrices):
                    col_index = index % 3  # Get the current column index
                    with columns[col_index]:
                        st.subheader(f"Confusion Matrix for {model_name}")
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                        disp.plot(cmap=plt.cm.Blues)
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

                 # Dictionary to store model performance
                model_performance = []

                # List of models to train for regression
                regression_models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge Regression": Ridge(),
                    "Lasso Regression": Lasso(),
                    "Decision Tree Regressor": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "Gradient Boosting Regressor": GradientBoostingRegressor(),
                    "Support Vector Regressor": SVR(),
                }

                # Train models and evaluate
                for model_name, model in regression_models.items():
                    start_time = time.time()  # Start timer
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                        # Calculate performance metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    explained_variance = explained_variance_score(y_test, y_pred)
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

                    # Store performance metrics
                    model_performance.append({
                        "Model": model_name,
                        "RMSE": rmse,
                        "R² Score": r2,
                        "MAE": mae,
                        "Explained Variance": explained_variance,
                        "MAPE": mape,
                        "Training Time (s)": time.time() - start_time,
                        "Status": "Success"
                    })

                # Convert to DataFrame
                comparison_df = pd.DataFrame(model_performance)

                # Sort the DataFrame by R² Score in descending order
                comparison_df = comparison_df.sort_values(by="R² Score", ascending=False)

                # Identify the best model
                best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax()]['Model']
                best_model = comparison_df.loc[comparison_df['Model'] == best_model_name]
                
        
                # Display results for the best model
                st.subheader("Best Model: {}".format(best_model_name))
                st.write("R² Score: {:.2f}".format(best_model["R² Score"].values[0]))
                st.write("RMSE: {:.2f}".format(best_model["RMSE"].values[0]))

                 # Optionally, display the full comparison table
                st.subheader("Model Comparison")
                st.table(comparison_df)
                
                # Create a summary DataFrame for visualization
                metrics_df = comparison_df[['Model', 'RMSE', 'R² Score', 'MAE']]
                metrics_df.set_index('Model', inplace=True)

                # Melt the DataFrame for Plotly
                metrics_melted = metrics_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')

                # Create a bar chart with Plotly
                import plotly.express as px
                fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group',
                            title='Model Performance Metrics Comparison (Regression)')
                st.plotly_chart(fig)

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

        # Select target variable
        numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
        categorical_df = df.select_dtypes(include=['object', 'category'])  # Select categorical columns

        # Combine numeric and categorical columns for target variable selection
        all_target_options = pd.concat([numeric_df, categorical_df], axis=1)

        if all_target_options.shape[1] < 2:  # At least one feature and one target
            st.error("The dataset must contain at least one feature and one target variable.")
        else:
            target_col = st.sidebar.selectbox("Select Target Variable", all_target_options.columns)

            # Define X and y based on the selected target variable
            X = df.drop(columns=[target_col])  # Features
            y = df[target_col]  # Target variable

            # Add manual selection for problem type
            st.sidebar.subheader("Select Problem Type")
            problem_type = st.sidebar.selectbox("Select Problem Type", ("Classification", "Regression"))

            train_button = st.sidebar.button("Train Models")

            if train_button:
                with st.spinner("Training uploaded dataset. Please wait..."):
                    time.sleep(2)

                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100, random_state=42)

                    # Prepare to store model performance
                    model_performance = []


                    # Initialize confusion_matrices
                    confusion_matrices = []

                    # Select models based on problem type
                    if problem_type == "Classification":
                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=200),
                            "Decision Trees": DecisionTreeClassifier(),
                            "Random Forest": RandomForestClassifier(),
                            "Gradient Boosting": GradientBoostingClassifier(),
                            "Support Vector Machine": SVC(),
                            "K-Nearest Neighbors": KNeighborsClassifier(),
                            "Naive Bayes": GaussianNB(),
                            "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                            "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
                        }

                        # Train models and evaluate
                        for model_name, model in models.items():
                            start_time = time.time()  # Start timer
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, output_dict=True)

                            # Store performance metrics
                            model_performance.append({
                                "Model": model_name,
                                "Accuracy": accuracy,
                                "Precision": report['weighted avg']['precision'],
                                "Recall": report['weighted avg']['recall'],
                                "F1-Score": report['weighted avg']['f1-score'],
                                "Training Time (s)": time.time() - start_time,
                                "Status": "Success"
                            })         

                            # Create confusion matrix and store it
                            cm = confusion_matrix(y_test, y_pred)
                            confusion_matrices.append((model_name, cm))

                        # Convert to DataFrame
                        comparison_df = pd.DataFrame(model_performance)

                        # Sort the DataFrame
                        comparison_df = comparison_df.sort_values(by="Accuracy", ascending=False)

                        # Best model
                        best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax()]['Model']
                        best_model = comparison_df.loc[comparison_df['Model'] == best_model_name]

                        # Display results for the best model
                        st.subheader("Best Model: {}".format(best_model_name))
                        st.write("Accuracy: {:.2f}".format(best_model["Accuracy"].values[0]))
                        st.write("Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                        # Create a summary DataFrame for visualization
                        metrics_df = comparison_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
                        metrics_df.set_index('Model', inplace=True)

                        # Melt the DataFrame for Plotly
                        metrics_melted = metrics_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')

                        # Display Model Comparison Table before Confusion Matrices
                        st.subheader("Model Comparison")
                        st.table(comparison_df)

                        # Create a bar chart with Plotly
                        import plotly.express as px
                        fig = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group',
                                    title='Model Performance Metrics Comparison (Classification)')
                        st.plotly_chart(fig)

                        # Now display confusion matrices in a 3-column format
                        num_models = len(confusion_matrices)
                        columns = st.columns(3)

                        for index, (model_name, cm) in enumerate(confusion_matrices):
                            col_index = index % 3  # Get the current column index
                            with columns[col_index]:
                                st.subheader(f"Confusion Matrix for {model_name}")
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                disp.plot(cmap=plt.cm.Blues)
                                st.pyplot(plt)

                    else:  # Regression
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Ridge Regression": Ridge(),
                            "Lasso Regression": Lasso(),
                            "Decision Tree Regressor": DecisionTreeRegressor(),
                            "Random Forest Regressor": RandomForestRegressor(),
                            "Gradient Boosting Regressor": GradientBoostingRegressor(),
                            "Support Vector Regressor": SVR(),
                        }

                        # Train models and evaluate
                        for model_name, model in models.items():
                            start_time = time.time()  # Start timer
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Calculate performance metrics
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)

                            # Store performance metrics
                            model_performance.append({
                                "Model": model_name,
                                "RMSE": rmse,
                                "R² Score": r2,
                                "MAE": mae,
                                "Training Time (s)": time.time() - start_time,
                                "Status": "Success"
                            })

                        # Convert to DataFrame
                        comparison_df = pd.DataFrame(model_performance)

                        # Sort the DataFrame
                        comparison_df = comparison_df.sort_values(by="R² Score", ascending=False)

                        # Best model
                        best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax()]['Model']
                        best_model = comparison_df.loc[comparison_df['Model'] == best_model_name]

                        # Display results for the best model
                        st.subheader("Best Model: {}".format(best_model_name))
                        st.write("R² Score: {:.2f}".format(best_model["R² Score"].values[0]))
                        st.write("RMSE: {:.2f}".format(best_model["RMSE"].values[0]))
                        st.write("Mean Absolute Error: {:.2f}".format(best_model["MAE"].values[0]))
                                                
                        # Display Model Comparison Table before Confusion Matrices
                        st.subheader("Model Comparison")
                        st.table(comparison_df)
