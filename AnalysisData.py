import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Press the green button in the gutter to run the script.
class AnalysisData():
    def __init__(self,Directory):
        self.path = Directory
    def Analysis_all_particitenpts(self):
        dataset_path = f'{self.path}\Participants\Dataset\Dataset.csv'
        dataset_file_path = f'{self.path}\Participants\Dataset'
        data = pd.read_csv(dataset_path)
        data = data.drop(columns=['participant'])
        data = data.drop(columns=['Part'])
        mask = (data['Stress Report'] != 'nane')
        data = data[mask]
        data['Stress Report'] = pd.to_numeric(data['Stress Report'], errors='coerce')
        # data = data[(data != 0).all(axis=1)]


        # sns.pairplot(data)
        # plt.suptitle("Scatter Plot Matrix of Features vs. Stress Report", y=1.02)
        # plt.show()

        g = sns.pairplot(data, diag_kind="kde")
        g.map_lower(sns.kdeplot, levels=4, color=".2")
        g_path = fr'{dataset_file_path}\pairplot_Dataset.png'
        plt.savefig(g_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Correlation matrix
        correlation_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.savefig(fr'{dataset_file_path}\Correlation Matrix__Dataset.png', dpi=300, bbox_inches='tight')
        plt.show()

        X = data[['ECG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20']]
        y = data[['Stress Report']]

        # Add a constant to the model (intercept)
        model = LinearRegression()
        sfs = SequentialFeatureSelector(model, n_features_to_select=3, cv=5, scoring='neg_mean_squared_error')
        sfs.fit(X, y)
        Selected_Features = sfs.get_feature_names_out()
        X_selected = X[Selected_Features]
        # Add a constant (intercept) to the model
        X_selected_with_const = sm.add_constant(X_selected)
        X_with_const = sm.add_constant(X)

        # Fit the model with statsmodels
        model_selected = sm.OLS(y, X_selected_with_const).fit()
        model_full = sm.OLS(y, X_with_const).fit()

        # Print the summary of the selected model
        print(model_selected.summary())

        # Print the summary of the full model
        print(model_full.summary())

    def Analysis_per_particitenpt(self):
        Participants_path = f'{self.path}\Participants\participation management.xlsx'
        Participants_df = pd.read_excel(Participants_path, header=1)
        Participants_df = Participants_df.dropna(axis=1, how='all')
        Participants_df['code'] = pd.to_numeric(Participants_df['code'], errors='coerce').astype('Int64')
        TotalCorr=pd.DataFrame()
        for j, row in Participants_df.iterrows():
            ID = row['code']
            Group = row['Group']
            print(ID)
            # ID = 9
            # Group = 'music'
            directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            dataParticipent_path = fr'{directory}\data_{ID}.csv'
            data=pd.read_csv(dataParticipent_path)
            data=data.drop(columns=['participant'])
            data=data.drop(columns=['Part'])


            # sns.pairplot(data)
            # plt.suptitle("Scatter Plot Matrix of Features vs. Stress Report", y=1.02)
            # plt.show()

            g = sns.pairplot(data, diag_kind="kde")
            g.map_lower(sns.kdeplot, levels=4, color=".2")
            g_path=fr'{directory}\pairplot_{ID}.png'
            plt.savefig(g_path, dpi=300, bbox_inches='tight')
            plt.show()

            # Correlation matrix
            correlation_matrix = data.corr()
            first_row_corr = correlation_matrix.iloc[0, :]
            features_df = pd.DataFrame(first_row_corr).T  # Transpose to match participant as a row
            features_df['Participant_ID'] = ID  # Add participant ID to track
            cols = ['Participant_ID'] + [col for col in features_df if col != 'Participant_ID']
            features_df = features_df[cols]
            # Concatenate with TotalCorr to accumulate results
            TotalCorr = pd.concat([TotalCorr, features_df], axis=0, ignore_index=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix')
            plt.savefig(fr'{directory}\Correlation Matrix_{ID}.png', dpi=300, bbox_inches='tight')
            plt.show()

            X = data[['ECG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20']]
            y = data[['Stress Report']]


            # Add a constant to the model (intercept)
            model = LinearRegression()
            sfs = SequentialFeatureSelector(model, n_features_to_select=3, cv=5, scoring='neg_mean_squared_error')
            sfs.fit(X, y)
            Selected_Features = sfs.get_feature_names_out()
            X_selected = X[Selected_Features]
            # Add a constant (intercept) to the model
            X_selected_with_const = sm.add_constant(X_selected)
            X_with_const = sm.add_constant(X)

            # Fit the model with statsmodels
            model_selected = sm.OLS(y, X_selected_with_const).fit()
            model_full = sm.OLS(y, X_with_const).fit()

            # Print the summary of the selected model
            print(model_selected.summary())

            # Print the summary of the full model
            print(model_full.summary())


            # Create a 3D scatter plot
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            #
            # # Plot the selected features against y
            # ax.scatter(X_selected['ECG_Rate_Mean'], X_selected['HRV_MeanNN'], X_selected['HRV_SDNN'], c=y,
            #            cmap='viridis', marker='o')
            #
            # # Set labels and title
            # ax.set_xlabel('ECG_Rate_Mean')
            # ax.set_ylabel('HRV_MeanNN')
            # ax.set_zlabel('HRV_SDNN')
            # ax.set_title('3D Scatter Plot: Selected Features vs Target')
            #
            # # Show the plot
            # plt.show()

            # summary_str = model.summary().as_text()
            # # Split the summary string into lines
            # summary_lines = summary_str.split('\n')
            # # Convert summary lines into a DataFrame
            # summary_df = pd.DataFrame({'Summary': summary_lines})
            # # Save the DataFrame to a CSV file
            # summary_df.to_csv(fr'{directory}\Reggresion_summary_{ID}.png', index=False)
            # # Print out the results
        dataset_path = f'{self.path}\Participants\Dataset\DatasetCorr.csv'
        TotalCorr.to_csv(dataset_path)

        # Example of loading your dataset (replace with your data)
        # data = pd.read_csv('your_data.csv')

        # Here, 'dependent_variable' is the outcome variable, 'fixed_effects_variable' is the fixed effect,
        # and 'random_effect_grouping' is the random effect grouping (e.g., participant IDs).

        # Mixed Linear Model
        # Replace 'dependent_variable' with the column name of the outcome,
        # 'fixed_effects_variable' with your fixed effect predictor, and
        # 'random_effect_grouping' with the grouping factor for the random effect.

        model = mixedlm("dependent_variable ~ fixed_effects_variable",
                        data,
                        groups=data["random_effect_grouping"],
                        re_formula="~fixed_effects_variable")  # random effect with respect to the fixed variable

        # Fit the model
        result = model.fit()

        # Summary of the model
        print(result.summary())

        # Extract random effects
        random_effects = result.random_effects

        # Print random effects for each grouping level
        print("Random Effects by Group:")
        for group, effect in random_effects.items():
            print(f"Group {group}: {effect}")

        # If you want to plot or further analyze the random effects
        random_effects_df = pd.DataFrame(random_effects).T  # Transpose for easier viewing
        print(random_effects_df)
