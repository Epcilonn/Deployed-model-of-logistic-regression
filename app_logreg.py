from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from io import BytesIO
import base64
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates_logreg")

def create_visualizations(data, target_variable):
    plt.figure(figsize=(18, 12))

    # Create a boxplot
    plt.subplot(2, 2, 1)
    sns.boxplot(data)
    plt.title('Boxplot of Data')
    plt.xlabel('Features')
    plt.ylabel('Values')

    # Create a pie chart
    plt.subplot(2, 2, 2)
    pie_data = data[target_variable].value_counts()
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Pie Chart of {target_variable}')

    # Create a correlation heatmap
    plt.subplot(2, 2, 3)
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')

    # Save the boxplot, pie chart, and correlation heatmap to BytesIO objects
    combined_stream = BytesIO()
    plt.savefig(combined_stream, format='png')
    combined_stream.seek(0)

    # Encode the combined visualizations to base64 for HTML rendering
    encoded_combined = base64.b64encode(combined_stream.read()).decode('utf-8')
    plt.close()  # Close the plots to free up resources

    # Return the combined images as an HTML img tag
    combined_image = f'<img src="data:image/png;base64,{encoded_combined}" alt="Combined Visualizations">'
    return combined_image






def logistic_regression(data, target_variable):
    result = "<p>Exploratory Data Analysis (EDA)</p><br>"

     # Create visualizations (combined box plot, pie chart, and correlation heatmap)
    combined_image = create_visualizations(data, target_variable)
    result += f'{combined_image}<br>'


    x = list(data.columns)
    ## Treating outliers
    
    for i in x:
        if data[i].dtypes in ("float64","int64"):
            q1 = data[i].quantile(0.25)
            q3 = data[i].quantile(0.75)
            iqr = q3 - q1
            ul = q3 + 1.5 * iqr
            ll = q1 - 1.5 * iqr
            data[i].clip(upper=ul, lower=ll, inplace=True)

    ## Filling null values
    result += "<br><br>Missing values:-"
    for i in x:
        if (data[i].isnull().sum() / len(data[i]) * 100) >= 40:
            data.drop(i, axis=1, inplace=True)
        elif data[i].isna().sum() > 0:
            result += f"missing values in {i} is {data[i].isna().sum()}<br>"
        else:
            result += "<br>There are no missing values in the columns<br>"
            break

    for i in x:
        if data[i].dtypes in ("float64", "int64"):
            q1 = data[i].quantile(0.25)
            q3 = data[i].quantile(0.75)
            iqr = q3 - q1
            upperL = q3 + 1.5 * iqr
            lowerL = q1 - 1.5 * iqr
            if data[i].min() < lowerL or data[i].max() > upperL:
                data[i].fillna(data[i].median(), inplace=True)
            else:
                data[i].fillna(data[i].mean(), inplace=True)
        else:
            data[i].fillna(data[i].mode()[0], inplace=True)

    # Handling missing values in the target variable
    data.dropna(subset=[target_variable], inplace=True)

    

    Quantitative_Variable = data[data.select_dtypes(include=[np.number]).columns.tolist()]
    
    Qualitative_Variable = data[data.select_dtypes(include=['object']).columns.tolist()]
    
    Qualitative_Variable=Qualitative_Variable.apply(LabelEncoder().fit_transform) # label in ascending order
    
    Final_combined = pd.concat([Qualitative_Variable, Quantitative_Variable],axis=1)
    
    
    Y = pd.DataFrame(Final_combined[target_variable])
    if Y.isna().sum().any() > 0:
        Y.fillna(Y.mean(), inplace=True)

    X = Final_combined.drop(target_variable, axis=1)

   
    ## Model building
    result += "<br><br>Train dataset contains 70% of the dataset and Test Dataset contains 30% of the dataset<br>"
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=134)
    train = pd.concat([y_train, X_train], axis=1)
    logreg = LogisticRegression()

    # Feature selection
    Model = sfs(logreg, n_features_to_select=5, direction='backward', scoring='accuracy', cv=5)
    Model.fit(X_train, y_train)
    f_out = list(Model.get_feature_names_out().flatten())
    X_train = X_train.loc[:, f_out]
    
    Model = LogisticRegression()
    Model.fit(X_train, y_train)
        
    ## Prediction on train data
    train = pd.concat([y_train, X_train], axis=1)
    train_target = train.iloc[:, 0]
    train['Probability']= Model.predict_proba(X_train)[:,1]
    
    train['Predicted']=np.where(train['Probability'] >= 0.7,1,0)
    
    train_result=classification_report( train_target,train['Predicted'])
    
    result+="The Classification report for the Train data:<br><br>"+str(train_result)
    
        

    #Prediction on test data
    X_test = X_test.loc[:, f_out]
    test = pd.concat([y_test,X_test],axis=1)
    test_target = test.iloc[:, 0]
    test['Probability']=Model.predict_proba(X_test)[:,1]
    test['Predicted']=np.where(test['Probability'] >= 0.7,1,0) 
    test_result=classification_report(test_target,test['Predicted'])
    result+="The Classification report for the Test data:<br><br>"+str(test_result)

    result = result.replace('\n', '<br>')
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded dataset
        dataset = request.files['dataset']

        # Read the dataset into a DataFrame
        data = pd.read_csv(dataset)  # Assuming CSV format, modify as needed

        # Get the target variable from the form
        target_variable = request.form['target_variable']

        # Call the linear_regression function and capture the result
        result = logistic_regression(data, target_variable)

        # Add any additional processing or rendering logic here

        # Return the result to be displayed in the template
        return render_template('index.html', result=result)

    return render_template('index.html')


# Add a new route to handle the form submission
@app.route('/process_data', methods=['POST'])
def process_data():
    # Get the uploaded dataset
    dataset = request.files['dataset']

    # Read the dataset into a DataFrame
    data = pd.read_csv(dataset)  # Assuming CSV format, modify as needed

    # Get the target variable from the form
    target_variable = request.form['target_variable']

    # Call the linear_regression function and capture the result
    result = logistic_regression(data, target_variable)

    # Return the result to be displayed in the template
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
