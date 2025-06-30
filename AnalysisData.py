import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
import scikit_posthocs as sp
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import kruskal
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

class AnalysisData():
    def __init__(self,Directory):
        self.path = Directory
        # self.sorted_DATA = sorted_DATA
        # self.sampling_frequency = sampling_frequency
        self.segment_DATA = pd.DataFrame()
        self.preprocessed_DATA = pd.DataFrame()
        self.window_samples = 0

    def _binary_metrics(y_true, y_pred):
        """Return accuracy, precision, recall (== sensitivity) and specificity."""
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        sens = recall_score(y_true, y_pred,   zero_division=0)     # sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan            # specificity
        return acc, prec, sens, spec

    @staticmethod
    def chrono_split(df, train_pct=0.6, val_pct=0.2, time_col="Time"):
        """
        Chronological split → (train_df, val_df, test_df).

        val_df is returned even if you do not use it later, so the signature
        stays general.
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        n       = len(df)
        t_end   = int(train_pct * n)
        v_end   = int((train_pct + val_pct) * n)
        return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]

    # ── 2. main routine ────────────────────────────────────────────────
    def _feature_importance(self,model, feat_cols):
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feat_cols)
        if hasattr(model, "coef_"):
            w = np.abs(model.coef_).ravel()
            return pd.Series(w / w.sum(), index=feat_cols)
        raise ValueError("Unsupported model for importance extraction.")

    @staticmethod
    def _best_cutoff(y_true: np.ndarray, y_prob: np.ndarray, step: float = 0.01):
        """
        Scan thresholds ∈ (0,1] and return the one with maximal F1.
        Returns (best_threshold, metrics_dict)
        """
        best_thr, best_f1 = 0.5, -1
        best_scores = {}
        for thr in np.arange(step, 1.0, step):
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_scores = {
                    "Accuracy":  accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall":    recall_score(y_true, y_pred, zero_division=0),
                    "F1":        f1
                }
        return best_thr, best_scores

    def ML_models_Prediction(self, n_repeats=9):
        window_sizes = [5, 10, 30, 60]
        overlaps = [0.0, 0.5]

        # Fixed base_models dictionary
        base_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        # Fixed param_grids dictionary
        param_grids = {
            'DecisionTree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        }

        participants_csv = os.path.join(self.path, 'Participants', 'participation management.csv')
        participants = pd.read_csv(participants_csv)
        all_ids = participants['code'].dropna().astype(int).unique()
        Prediction_list=['Stress','Fatigue','Accuracy','RT']
        # signals=['HRV','RSP_C', 'RSP_D', 'EDA', 'All']
        signals=['All']
        for prediction_type in tqdm(Prediction_list, desc=f"🔍 Prediction Type"):
            for signal in tqdm(signals, desc=f"📡 Signal {signal} {prediction_type}", leave=False):
                print(f"\n📊 Evaluating signal: {signal}")
                results = []
                importances = {name: [] for name in base_models}
                best_ws = {name: {'window': None, 'overlap': None, 'f1': -np.inf} for name in base_models}
                best_params = {}

                for repeat in tqdm(range(n_repeats), desc=f"🔁 Repeats {repeat}", leave=False):
                    print("Repeat:", repeat + 1)
                    # -----------Iteration 1-Split Train Test------------------------------
                    train_ids, test_ids = train_test_split(
                        all_ids, test_size=0.2, random_state=42 + repeat
                    )
                    run_full = (repeat == 0)
                    iter_to_run = [1, 2, 3] if run_full else [1, 3]
                    # -----------Iteration 2-Choose Window Size------------------------------
                    # -----------Iteration 2-Grid Search for Window, Overlap, and Hyperparameters------------------------------
                    if 2 in iter_to_run:
                        print("  Grid search for best window, overlap, and hyperparameters for each model...")

                        for name, base_model in tqdm(base_models.items(), desc=f"🧠 Models{base_model}"):
                            print(f"    Processing {name}...")
                            best_config = {'window': None, 'overlap': None, 'params': {}, 'f1': -np.inf}

                            for ws in tqdm(window_sizes, desc=f"🪟 Window for {name}", leave=False):
                                for ov in tqdm(overlaps, desc=f"🔁 Overlap {int(ov * 100)}%", leave=False):
                                    iteration_name = f"{name} | Window={ws}s | Overlap={int(ov * 100)}%"
                                    print(f"🔄 {iteration_name}")
                                    file_path = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data\Dataset_{ws}s_{int(ov * 100)}.csv'
                                    if not os.path.exists(file_path):
                                        print(f"Missing file: WS={ws}s, OV={int(ov * 100)}%")
                                        continue

                                    try:
                                        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
                                        if signal != 'All':
                                            meta_columns = ['ID', 'Group', 'Time', 'Class', 'Test_Type', 'Level',
                                                            'Accuracy', 'RT', 'Stress', 'Fatigue']
                                            selected_columns = meta_columns + [col for col in df.columns if
                                                                               col.startswith(signal + '_')]
                                            df = df[selected_columns]
                                        df_train = df[df['ID'].isin(train_ids)]
                                        feature_cols = [c for c in df.columns if
                                                        c not in ['Time', 'ID', 'Group', 'Class', 'Stress', 'Fatigue']]

                                        if prediction_type=='Stress':
                                            y_tr = df_train['Stress']
                                        elif prediction_type == 'Fatigue':
                                            y_tr = df_train['Fatigue']
                                        elif prediction_type == 'Accuracy':
                                            y_tr = df_train['Accuracy']
                                        else:
                                            y_tr=df_train['RT']
                                        groups = df_train['ID']

                                        if len(df_train) == 0:
                                            continue

                                        X_tr = df_train[feature_cols]
                                        gkf = GroupKFold(n_splits=5)

                                        # Grid search for hyperparameters on this window/overlap combination
                                        grid = GridSearchCV(
                                            clone(base_model),
                                            param_grid=param_grids[name],
                                            cv=gkf,
                                            scoring='f1',
                                            n_jobs=1
                                        )
                                        grid.fit(X_tr, y_tr, groups=groups)

                                        mean_f1 = grid.best_score_
                                        print(
                                            f"      WS={ws}s, OV={int(ov * 100)}%, F1={mean_f1:.3f}, Params={grid.best_params_}")

                                        # Update best configuration if this is better
                                        if mean_f1 > best_config['f1']:
                                            best_config.update({
                                                'window': ws,
                                                'overlap': ov,
                                                'params': grid.best_params_,
                                                'f1': mean_f1
                                            })
                                            print(f"        → New best for {name}!")

                                    except Exception as e:
                                        print(f"      Error with WS={ws}s, OV={int(ov * 100)}%: {str(e)}")

                            # Store best configuration for this model
                            best_ws[name] = {
                                'window': best_config['window'],
                                'overlap': best_config['overlap'],
                                'f1': best_config['f1']
                            }
                            best_params[name] = best_config['params']

                            if best_config['window'] is not None:
                                print(
                                    f"    Best config for {name}: WS={best_config['window']}s, OV={int(best_config['overlap'] * 100)}%, F1={best_config['f1']:.3f}")
                                print(f"    Best params: {best_config['params']}")
                            else:
                                print(f"    No valid configuration found for {name}")

                        print("  Grid search completed for all models.")
                    # -----------Iteration 3 Evaluation on Test Set------------------------------
                    for name in base_models:
                        ws = best_ws[name]['window']
                        ov = best_ws[name]['overlap']
                        if ws is None or ov is None:
                            continue
                        file_path = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data\Dataset_{ws}s_{int(ov * 100)}.csv'
                        df = pd.read_csv(file_path).dropna().reset_index(drop=True)
                        if signal != 'All':
                            meta_columns = ['ID', 'Group', 'Time', 'Class', 'Test_Type', 'Level',
                                            'Accuracy', 'RT', 'Stress', 'Fatigue']
                            selected_columns = meta_columns + [col for col in df.columns if
                                                               col.startswith(signal + '_')]
                            df = df[selected_columns]
                        df_train = df[df['ID'].isin(train_ids)]
                        feature_cols = [c for c in df.columns if
                                        c not in ['Time', 'ID', 'Group', 'Class', 'Stress', 'Fatigue']]
                        if prediction_type == 'Stress':
                            y = df_train['Stress']
                        elif prediction_type == 'Fatigue':
                            y = df_train['Fatigue']
                        elif prediction_type == 'Accuracy':
                            y = df_train['Accuracy']
                        else:
                            y = df_train['RT']
                        model = clone(base_models[name])
                        model = base_models[name].set_params(**best_params[name])
                        model.fit(df_train[feature_cols], y)
                        params = best_params[name]

                        df_test = df[df['ID'].isin(test_ids)]
                        feature_cols = [c for c in df.columns if
                                        c not in ['Time', 'ID', 'Group', 'Class', 'Stress', 'Fatigue']]
                        X_te = df_test[feature_cols]
                        if prediction_type == 'Stress':
                            y_te = df_train['Stress']
                        elif prediction_type == 'Fatigue':
                            y_te = df_train['Fatigue']
                        elif prediction_type == 'Accuracy':
                            y_te = df_train['Accuracy']
                        else:
                            y_te = df_train['RT']

                        y_pred = model.predict(X_te)
                        mse = mean_squared_error(y_te, y_pred)

                        result_row = {
                            'Signal': signal,
                            'Repeat': repeat + 1,
                            'Model': name,
                            'Window (s)': ws,
                            'Overlap (%)': int(ov * 100),
                            'Accuracy': accuracy_score(y_te, y_pred) * 100,
                            'Precision': precision_score(y_te, y_pred, zero_division=0) * 100,
                            'Recall': recall_score(y_te, y_pred, zero_division=0) * 100,
                            'F1': f1_score(y_te, y_pred, zero_division=0) * 100,
                            'MSE': mse
                        }
                        result_row.update({f'param_{k}': v for k, v in params.items()})
                        results.append(result_row)

                        # Save feature importance plot, CSV, and collect for summary
                        if hasattr(model, 'feature_importances_'):
                            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                            importances[name].append(imp)
                            out_dir = fr"{self.path}\Participants\Dataset\ML\{name}\Repeat{repeat + 1}"
                            os.makedirs(out_dir, exist_ok=True)
                            imp.to_csv(os.path.join(out_dir, "Feature_Importance.csv"))
                            plt.figure(figsize=(10, 5))
                            imp.plot.bar()
                            plt.title(f"{name} Feature Importances - Repeat {repeat + 1}")
                            plt.tight_layout()
                            plt.savefig(os.path.join(out_dir, "Feature_Importance_Plot.png"))
                            plt.close()

                # Save per-signal results
                results_df = pd.DataFrame(results)
                results_df=results_df.round(2)
                if prediction_type == 'Stress':
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Prediction','Stress')
                elif prediction_type == 'Fatigue':
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Prediction','Fatigue')
                elif prediction_type == 'Accuracy':
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Prediction','Performance','Accuracy')
                else:
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Prediction','Performance','RT')
                os.makedirs(out_dir, exist_ok=True)
                results_df.to_csv(os.path.join(out_dir, f'NestedCV_Results_{signal}.csv'), index=False)
                print(f"✅ Saved results for {signal} to NestedCV_Results_{signal}_{prediction_type}.csv")

                # Aggregate stats per model
                # Summary metrics per model
                summary_metrics = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].agg(
                    ['mean', 'std']).round(2)

                # Get first row per model (or use .mode().iloc[0] if you want most frequent)
                # optimal_settings = results_df.groupby('Model')[['Window (s)', 'Overlap (%)','param_max_depth','param_min_samples_split','param_n_estimators','param_learning_rate']].first()
                optimal_settings = results_df.groupby('Model')[['Window (s)', 'Overlap (%)']].first()

                # Combine metrics and optimal settings
                summary = pd.concat([summary_metrics, optimal_settings], axis=1)
                summary.to_csv(os.path.join(out_dir,fr'NestedCV_{signal}_{prediction_type}_Summary.csv'), index=False)
                print("Summary statistics saved")

                # Feature importance summary
                for name, imps in importances.items():
                    if imps:
                        imp_df = pd.concat(imps, axis=1).fillna(0)
                        imp_df.columns = [f'Repeat_{i + 1}' for i in range(len(imps))]
                        imp_df['Mean'] = imp_df.mean(axis=1)
                        imp_df['Std'] = imp_df.std(axis=1)
                        out_subdir = os.path.join(out_dir, signal, name)
                        os.makedirs(out_subdir, exist_ok=True)
                        imp_df.sort_values('Mean', ascending=False).to_csv(
                            os.path.join(out_subdir, "Feature_Importance_Summary.csv"))
                        plt.figure(figsize=(10, 5))
                        imp_df['Mean'].sort_values(ascending=False).plot.bar(yerr=imp_df['Std'])
                        plt.title(f"{name} - Mean Feature Importances over {n_repeats} Repeats ({signal})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_subdir, "Feature_Importance_Summary_Plot.png"))
                        plt.close()

                    # Feature importance summary
                all_imps_long = []
                combined_df = pd.DataFrame()

                for name, imps in importances.items():
                    if imps:
                        imp_df = pd.concat(imps, axis=1).fillna(0)
                        imp_df.columns = [f'Repeat_{i + 1}' for i in range(len(imps))]
                        imp_df['Mean'] = imp_df.mean(axis=1)
                        imp_df['Std'] = imp_df.std(axis=1)
                        out_subdir = os.path.join(out_dir, signal, name)
                        os.makedirs(out_subdir, exist_ok=True)
                        imp_df.sort_values('Mean', ascending=False).to_csv(
                            os.path.join(out_subdir, "Feature_Importance_Summary.csv"))
                        plt.figure(figsize=(10, 5))
                        imp_df['Mean'].sort_values(ascending=False).plot.bar(yerr=imp_df['Std'])
                        plt.title(f"{name} - Mean Feature Importances over {n_repeats} Repeats ({signal})")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_subdir, "Feature_Importance_Summary_Plot.png"))
                        plt.close()

                        model_imp_df = pd.concat(imps, axis=1).fillna(0)
                        model_imp_df.columns = [f'{name}_Repeat_{i + 1}' for i in range(len(imps))]
                        model_imp_df[f'{name}_Mean'] = model_imp_df.mean(axis=1)
                        combined_df = pd.concat([combined_df, model_imp_df[[f'{name}_Mean']]], axis=1)

                        for i, imp in enumerate(imps):
                            temp = imp.reset_index()
                            temp.columns = ['Feature', 'Importance']
                            temp['Model'] = name
                            temp['Repeat'] = i + 1
                            all_imps_long.append(temp)

                if not combined_df.empty:
                    combined_df['Combined_Mean'] = combined_df.mean(axis=1)
                    combined_df = combined_df.sort_values('Combined_Mean', ascending=False)
                    comb_dir = os.path.join(out_dir, signal, 'Combined')
                    os.makedirs(comb_dir, exist_ok=True)
                    combined_df.to_csv(os.path.join(comb_dir, "Combined_Feature_Importance.csv"))
                    plt.figure(figsize=(12, 6))
                    combined_df['Combined_Mean'].plot(kind='bar')
                    plt.title(f"Combined Feature Importances Across All Models ({signal})")
                    plt.ylabel("Mean Importance")
                    plt.tight_layout()
                    plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_Plot.png"))
                    plt.close()

                if all_imps_long:
                    all_df = pd.concat(all_imps_long, axis=0)
                    mean_df = all_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
                    plt.figure(figsize=(12, 6))
                    mean_df.plot(kind='bar')
                    plt.title(f"Combined Mean Feature Importance Across All Models ({signal})")
                    plt.ylabel("Mean Importance")
                    plt.tight_layout()
                    plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_MeanPlot.png"))
                    plt.close()

                    plt.figure(figsize=(14, 6))
                    sns.boxplot(data=all_df, x='Feature', y='Importance')
                    plt.xticks(rotation=90)
                    plt.title(f"Feature Importance Distribution (All Models & Repeats) ({signal})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_BoxPlot.png"))
                    plt.close()

    def ML_models_Classification(self, n_repeats=9, no_breath_data=False, clases_3=False):
        window_sizes = [5, 10, 30, 60]
        overlaps = [0.0, 0.5]
        base_models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }

        param_grids = {
            'DecisionTree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']  # only relevant for ‘rbf’
            }
        }
        if clases_3:
            if no_breath_data:
                out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification',
                                       '3 class', 'No breath group')
            else:
                out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification',
                                       '3 class', 'All Data')
        else:
            if no_breath_data:
                out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification',
                                       '2 class', 'No breath group')
            else:
                out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification',
                                       '2 class', 'All Data')
        meta_columns = ['Window', 'Overlap', 'ID', 'Group', 'Time', 'Class', 'Test_Type', 'Level', 'Accuracy', 'RT', 'Stress', 'Fatigue']
        participants_csv = os.path.join(self.path, 'Participants', 'participation management.csv')
        participants = pd.read_csv(participants_csv)
        all_ids = participants['code'].dropna().astype(int).unique()
        if no_breath_data:
            signals = ['All']
            # Filter participants to exclude the 'breath' group
            filtered_participants = participants[participants['Group'] != 'breath']
            # Extract unique IDs (assuming 'code' column contains them)
            all_ids = filtered_participants['code'].dropna().astype(int).unique()
        else:
            signals = ['HRV', 'RSP_C', 'RSP_D', 'EDA', 'All']
        for signal in signals:
            print(f"\n📊 Evaluating signal: {signal}")
            results = []
            importances = {name: [] for name in base_models}
            best_ws = {name: {'window': None, 'overlap': None, 'f1': -np.inf} for name in base_models}
            best_params = {}
            for repeat in range(n_repeats):
                print("Repeat:", repeat + 1)
                # -----------Iteration 1-Split Train Test------------------------------
                train_ids, test_ids = train_test_split(
                    all_ids, test_size=0.2, random_state=42 + repeat
                )
                run_full = (repeat == 0)
                iter_to_run = [1, 2, 3, 4] if run_full else [1, 4]
                # -----------Iteration 2-Choose Window Size------------------------------
                # -----------Iteration 2-Grid Search for Window, Overlap, and Hyperparameters------------------------------
                if 2 in iter_to_run:
                    # 🔹 נתיב לתיקייה
                    dir_path = fr"{out_dir}\hyperparameters"

                    # 🔹 שמות הקבצים
                    files = ["best_config.csv", "best_params.csv", "best_ws.csv"]

                    # 🔹 מסלולים מלאים
                    file_paths = [os.path.join(dir_path, f) for f in files]

                    # 🔹 בדיקה אם הקבצים קיימים
                    if all(os.path.exists(p) for p in file_paths):
                        # קריאה
                        df_config = pd.read_csv(file_paths[0])
                        df_params = pd.read_csv(file_paths[1])
                        df_ws = pd.read_csv(file_paths[2])

                        print("✅ Files loaded successfully.")

                        # --- בניית best_ws ---
                        df_ws_transposed = df_ws.set_index("Unnamed: 0").transpose()

                        best_ws = {}
                        for model, row in df_ws_transposed.iterrows():
                            best_ws[model.strip()] = {
                                "window": int(row["window"]),
                                "overlap": float(row["overlap"]),
                                "f1": float(row["f1"])
                            }

                        # --- פונקציה לניקוי פרמטרים ---
                        def clean_params(params_dict):
                            clean = {}
                            for k, v in params_dict.items():
                                if isinstance(v, str):
                                    v_str = v.strip()
                                    if v_str.lower() == "none":
                                        clean[k] = None
                                    else:
                                        try:
                                            num = float(v_str)
                                            if num.is_integer():
                                                clean[k] = int(num)
                                            else:
                                                clean[k] = num
                                        except ValueError:
                                            clean[k] = v_str
                                elif isinstance(v, float) and v.is_integer():
                                    clean[k] = int(v)
                                else:
                                    clean[k] = v
                            return clean

                        # --- בניית best_params ---
                        best_params = {}

                        # Transpose so models are rows
                        df_params_T = df_params.set_index("Unnamed: 0").transpose()

                        for model, row in df_params_T.iterrows():
                            params_raw = row.dropna().to_dict()
                            params_clean = clean_params(params_raw)
                            best_params[model.strip()] = params_clean

                        # ✅ הצגת מפתחות לבדיקה
                        print("Best WS Models:", list(best_ws.keys()))
                        print("Best Params Models:", list(best_params.keys()))

                        # ✅ דוגמה לשימוש
                        name = "DecisionTree"
                        ws = best_ws[name]["window"]
                        ov = best_ws[name]["overlap"]
                        print(f"\n{name}: window={ws}, overlap={ov}, params={best_params[name]}")
                    else:
                        print("  Grid search for best window, overlap, and hyperparameters for each model...")
                        for name, base_model in base_models.items():
                            print(f"    Processing {name}...")
                            best_config = {'window': None, 'overlap': None, 'params': {}, 'f1': -np.inf}

                            for ws in window_sizes:
                                for ov in overlaps:
                                    file_path = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data\Dataset_{ws}s_{int(ov * 100)}.csv'
                                    if not os.path.exists(file_path):
                                        print(f"      Missing file: WS={ws}s, OV={int(ov * 100)}%")
                                        continue
                                    try:
                                        df = pd.read_csv(file_path)
                                        df=df.dropna(subset=['Class'])
                                        if signal != 'All':
                                            selected_columns = meta_columns + [col for col in df.columns if
                                                                               col.startswith(signal + '_')]
                                            df = df[selected_columns]
                                        df_train = df[df['ID'].isin(train_ids)]
                                        feature_cols = [c for c in df.columns if
                                                        c not in meta_columns]
                                        if clases_3:
                                            df_train['Class'] = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                                            df_train.loc[df_train['Class'] == 1, 'Class'] = df_train.loc[df_train['Class'] == 1, 'Level'].map({'easy': 1, 'hard': 2})
                                            y_tr = df_train['Class']
                                        else:
                                            y_tr = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})

                                        groups = df_train['ID']

                                        if len(df_train) == 0:
                                            continue

                                        X_tr = df_train[feature_cols]
                                        gkf = GroupKFold(n_splits=5)

                                        # Grid search for hyperparameters on this window/overlap combination
                                        grid = GridSearchCV(
                                            clone(base_model),
                                            param_grid=param_grids[name],
                                            cv=gkf,
                                            scoring='f1',
                                            n_jobs=-1
                                        )
                                        grid.fit(X_tr, y_tr, groups=groups)

                                        mean_f1 = grid.best_score_
                                        print(
                                            f"WS={ws}s, OV={int(ov * 100)}%, F1={mean_f1:.3f}, Params={grid.best_params_}")

                                        # Update best configuration if this is better
                                        if mean_f1 > best_config['f1']:
                                            best_config.update({
                                                'window': ws,
                                                'overlap': ov,
                                                'params': grid.best_params_,
                                                'f1': mean_f1
                                            })
                                            print(f"        → New best for {name}!")

                                    except Exception as e:
                                        print(f"      Error with WS={ws}s, OV={int(ov * 100)}%: {str(e)}")

                            # Store best configuration for this model
                            best_ws[name] = {
                                'window': best_config['window'],
                                'overlap': best_config['overlap'],
                                'f1': best_config['f1']
                            }
                            best_params[name] = best_config['params']
                            best_params.to_csv(fr'{out_dir}/hyperparameters/best_params.csv')
                            best_ws.to_csv(fr'{out_dir}/hyperparameters/best_ws.csv')
                            best_config(fr'{out_dir}/hyperparameters/best_config.csv')
                            os.makedirs(out_dir, exist_ok=True)
                            if best_config['window'] is not None:
                                print(
                                    f"    Best config for {name}: WS={best_config['window']}s, OV={int(best_config['overlap'] * 100)}%, F1={best_config['f1']:.3f}")
                                print(f"    Best params: {best_config['params']}")
                            else:
                                print(f"    No valid configuration found for {name}")


                        print("  Grid search completed for all models.")
                # -----------Iteration 3 Evaluation on Test Set------------------------------
                for name in base_models:
                    ws = best_ws[name]['window']
                    ov = best_ws[name]['overlap']
                    if ws is None or ov is None:
                        continue

                    file_path = fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data\Dataset_{ws}s_{int(ov * 100)}.csv'
                    df = pd.read_csv(file_path)
                    df = df.dropna(subset=['Class'])
                    if signal != 'All':
                        selected_columns = meta_columns + [col for col in df.columns if
                                                           col.startswith(signal + '_')]
                        df = df[selected_columns]
                    df_train = df[df['ID'].isin(train_ids)]
                    feature_cols = [c for c in df.columns if
                                    c not in meta_columns]
                    if clases_3:
                        df_train['Class'] = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                        df_train.loc[df_train['Class'] == 1, 'Class'] = df_train.loc[df_train['Class'] == 1, 'Level'].map({'easy': 1, 'hard': 2})
                        y =df_train['Class']
                    else:
                        y = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                    model = clone(base_models[name])
                    model = base_models[name].set_params(**best_params[name])
                    model.fit(df_train[feature_cols], y)
                    params = best_params[name]

                    df_test = df[df['ID'].isin(test_ids)]
                    X_te = df_test[feature_cols]
                    if clases_3:
                        df_test['Class'] = df_test['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                        df_test.loc[df_test['Class'] == 1, 'Class'] = df_test.loc[df_test['Class'] == 1, 'Level'].map({'easy': 1, 'hard': 2})
                        y_te = df_train['Class']
                    else:
                        y_te = df_test['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                    y_pred = model.predict(X_te)
                    result_row = {
                        'Signal': signal,
                        'Repeat': repeat + 1,
                        'Model': name,
                        'Window (s)': ws,
                        'Overlap (%)': int(ov * 100),
                        'Accuracy': accuracy_score(y_te, y_pred)*100,
                        'Precision': precision_score(y_te, y_pred, zero_division=0)*100,
                        'Recall': recall_score(y_te, y_pred, zero_division=0)*100,
                        'F1': f1_score(y_te, y_pred, zero_division=0)*100
                    }
                    result_row.update({f'param_{k}': v for k, v in params.items()})
                    print(result_row)
                    results.append(result_row)

                    # Save feature importance plot, CSV, and collect for summary
                    if hasattr(model, 'feature_importances_'):
                        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                        importances[name].append(imp)
                        out_dir = fr"{self.path}\Participants\Dataset\ML\{name}\Repeat{repeat + 1}"
                        os.makedirs(out_dir, exist_ok=True)
                        imp.to_csv(os.path.join(out_dir, "Feature_Importance.csv"))
                        # plt.figure(figsize=(10, 5))
                        # imp.plot.bar()
                        # plt.title(f"{name} Feature Importances - Repeat {repeat + 1}")
                        # plt.tight_layout()
                        # plt.savefig(os.path.join(out_dir, "Feature_Importance_Plot.png"))
                        # plt.close()

            # Save per-signal results
            results_df = pd.DataFrame(results)
            results_df=results_df.round(2)
            if clases_3:
                if no_breath_data:
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification', '3 class','All Data')
                else:
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification', '3 class', 'No breath group')
            else:
                if no_breath_data:
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification', '2 class','All Data')
                else:
                    out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'ML', 'Classification', '2 class', 'No breath group')
            os.makedirs(out_dir, exist_ok=True)
            results_df.to_csv(os.path.join(out_dir, f'NestedCV_Results_{signal}.csv'), index=False)
            print(f"✅ Saved results for {signal} to NestedCV_Results_{signal}.csv")

            # Summary metrics per model
            summary_metrics = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].agg(
                ['mean', 'std']).round(2)

            # Get first row per model (or use .mode().iloc[0] if you want most frequent)
            # optimal_settings = results_df.groupby('Model')[['Window (s)', 'Overlap (%)','param_max_depth','param_min_samples_split','param_n_estimators','param_learning_rate']].first()
            optimal_settings = results_df.groupby('Model')[['Window (s)', 'Overlap (%)']].first()

            # Combine metrics and optimal settings
            summary = pd.concat([summary_metrics, optimal_settings], axis=1)
            summary.to_csv(os.path.join(out_dir,fr'NestedCV_{signal}_Summary.csv'), index=False)
            print("Summary statistics saved")

            # Feature importance summary
            for name, imps in importances.items():
                if imps:
                    imp_df = pd.concat(imps, axis=1).fillna(0)
                    imp_df.columns = [f'Repeat_{i + 1}' for i in range(len(imps))]
                    imp_df['Mean'] = imp_df.mean(axis=1)
                    imp_df['Std'] = imp_df.std(axis=1)
                    out_subdir = os.path.join(out_dir, signal, name)
                    os.makedirs(out_subdir, exist_ok=True)
                    imp_df.sort_values('Mean', ascending=False).to_csv(
                        os.path.join(out_subdir, "Feature_Importance_Summary.csv"))
                    plt.figure(figsize=(10, 5))
                    imp_df['Mean'].sort_values(ascending=False).plot.bar(yerr=imp_df['Std'])
                    plt.title(f"{name} - Mean Feature Importances over {n_repeats} Repeats ({signal})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_subdir, "Feature_Importance_Summary_Plot.png"))
                    plt.close()

                # Feature importance summary
            all_imps_long = []
            combined_df = pd.DataFrame()

            for name, imps in importances.items():
                if imps:
                    imp_df = pd.concat(imps, axis=1).fillna(0)
                    imp_df.columns = [f'Repeat_{i + 1}' for i in range(len(imps))]
                    imp_df['Mean'] = imp_df.mean(axis=1)
                    imp_df['Std'] = imp_df.std(axis=1)
                    out_subdir = os.path.join(out_dir, signal, name)
                    os.makedirs(out_subdir, exist_ok=True)
                    imp_df.sort_values('Mean', ascending=False).to_csv(
                        os.path.join(out_subdir, "Feature_Importance_Summary.csv"))
                    plt.figure(figsize=(10, 5))
                    imp_df['Mean'].sort_values(ascending=False).plot.bar(yerr=imp_df['Std'])
                    plt.title(f"{name} - Mean Feature Importances over {n_repeats} Repeats ({signal})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_subdir, "Feature_Importance_Summary_Plot.png"))
                    plt.close()

                    model_imp_df = pd.concat(imps, axis=1).fillna(0)
                    model_imp_df.columns = [f'{name}_Repeat_{i + 1}' for i in range(len(imps))]
                    model_imp_df[f'{name}_Mean'] = model_imp_df.mean(axis=1)
                    combined_df = pd.concat([combined_df, model_imp_df[[f'{name}_Mean']]], axis=1)

                    for i, imp in enumerate(imps):
                        temp = imp.reset_index()
                        temp.columns = ['Feature', 'Importance']
                        temp['Model'] = name
                        temp['Repeat'] = i + 1
                        all_imps_long.append(temp)

            if not combined_df.empty:
                combined_df['Combined_Mean'] = combined_df.mean(axis=1)
                combined_df = combined_df.sort_values('Combined_Mean', ascending=False)
                comb_dir = os.path.join(out_dir, signal, 'Combined')
                os.makedirs(comb_dir, exist_ok=True)
                combined_df.to_csv(os.path.join(comb_dir, "Combined_Feature_Importance.csv"))
                plt.figure(figsize=(12, 6))
                combined_df['Combined_Mean'].plot(kind='bar')
                plt.title(f"Combined Feature Importances Across All Models ({signal})")
                plt.ylabel("Mean Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_Plot.png"))
                plt.close()

            if all_imps_long:
                all_df = pd.concat(all_imps_long, axis=0)
                mean_df = all_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
                plt.figure(figsize=(12, 6))
                mean_df.plot(kind='bar')
                plt.title(f"Combined Mean Feature Importance Across All Models ({signal})")
                plt.ylabel("Mean Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_MeanPlot.png"))
                plt.close()

                # Calculate mean importance for each feature to determine sorting order
                feature_mean_importance = all_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)

                # Create the plot with sorted features
                plt.figure(figsize=(14, 6))
                sns.boxplot(data=all_df, x='Feature', y='Importance', order=feature_mean_importance.index)
                plt.xticks(rotation=90)
                plt.title(f"Feature Importance Distribution (All Models & Repeats) ({signal})")
                plt.tight_layout()
                plt.savefig(os.path.join(comb_dir, "Combined_Feature_Importance_BoxPlot.png"))
                plt.close()

    def Cor(self):
        # ── 1. load data ──────────────────────────────────────────────
        stress_all = pd.read_excel(r"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Participants\All_HRV_stress_30s.xlsx"
        )

        hrv_feats = [
            "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
            "HRV_CVNN", "HRV_pNN20", "HRV_pNN50",
        ]

        # ── 2. helper to build one full scatter-matrix figure ─────────
        def _plot(df, y_col, title, out_png):
            n_feats = len(hrv_feats)
            fig, axes = plt.subplots(n_feats, 1, figsize=(6, 3 * n_feats), sharey=True)

            if not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

            for ax, feat in zip(axes, hrv_feats):
                # coloured dots
                sns.scatterplot(
                    data=df, x=feat, y=y_col, hue="Group",
                    palette="Set2", s=40, ax=ax, legend=False
                )
                # regression line
                sns.regplot(
                    data=df, x=feat, y=y_col,
                    scatter=False, ci=95, line_kws=dict(lw=1.5, alpha=0.8), ax=ax
                )
                # Pearson r
                r, p = linregress(df[feat], df[y_col])[:2]
                ax.set_title(f"{feat}  (r = {r:.2f},  p = {p:.3g})")
                ax.set_xlabel("Mean value")
                ax.grid(True)

            axes[0].set_ylabel(y_col)
            fig.suptitle(title, fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(out_png, dpi=300)
            plt.close(fig)  # keep memory footprint small

        # ── 3. figure #1 – raw stress ─────────────────────────────────
        _plot(
            stress_all,
            y_col="Stress",
            title="Stress vs HRV features (raw)",
            out_png=r"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Participants\All_particapents\stress_HRV_scatter.png"
        )

        # ── 4. create z-scored Stress inside each participant ─────────
        stress_all["Stress_z"] = (
            stress_all.groupby("ID")["Stress"]
            .transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        )

        # ── 5. figure #2 – normalized stress ──────────────────────────
        _plot(
            stress_all,
            y_col="Stress_z",
            title="Stress (z-score within participant) vs HRV features",
            out_png=r"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Participants\All_particapents\stress_HRV_scatter_norm.png")


    def Analysis_per_particitenpt(self):
        dataset_path = f'{self.path}\Participants\Dataset\Dataset.csv'
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
            # directory = fr'{self.path}\Participants\{Group}_group\P_{ID}'
            # dataParticipent_path = fr'{directory}\data_{ID}.csv'
            data=pd.read_csv(dataset_path)
            data=data[data['participant']==ID]
            data=data.drop(columns=['participant'])
            data=data.drop(columns=['Part'])
            # data.replace('-', np.nan, inplace=True)
            # Replace invalid entries like 'nane' with NaN
            data.replace('nane', np.nan, inplace=True)
            # Convert all columns to numeric where possible, forcing errors to NaN
            data = data.apply(pd.to_numeric, errors='coerce')

            # sns.pairplot(data)
            # plt.suptitle("Scatter Plot Matrix of Features vs. Stress Report", y=1.02)
            # plt.show()

            # g = sns.pairplot(data, diag_kind="kde")
            # g.map_lower(sns.kdeplot, levels=4, color=".2")
            # g_path=fr'{directory}\pairplot_{ID}.png'
            # plt.savefig(g_path, dpi=300, bbox_inches='tight')
            # plt.show()

            # Correlation matrix
            correlation_matrix = data.corr()
            Corr_path = fr'{self.path}\Participants\{Group}_group\P_{ID}\Corr_{ID}.csv'
            correlation_matrix.to_csv(Corr_path)
            first_row_corr = correlation_matrix.iloc[0, :]
            features_df = pd.DataFrame(first_row_corr).T  # Transpose to match participant as a row
            features_df['Participant_ID'] = ID  # Add participant ID to track
            cols = ['Participant_ID'] + [col for col in features_df if col != 'Participant_ID']
            features_df = features_df[cols]
            # Concatenate with TotalCorr to accumulate results
            TotalCorr = pd.concat([TotalCorr, features_df], axis=0, ignore_index=True)
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            # plt.title('Correlation Matrix')
            # plt.savefig(fr'{directory}\Correlation Matrix_{ID}.png', dpi=300, bbox_inches='tight')
            # plt.show()

            # X = data[['ECG_Rate_Mean', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', 'HRV_pNN20']]
            # y = data[['Stress Report']]
            #
            #
            # # Add a constant to the model (intercept)
            # model = LinearRegression()
            # sfs = SequentialFeatureSelector(model, n_features_to_select=3, cv=5, scoring='neg_mean_squared_error')
            # sfs.fit(X, y)
            # Selected_Features = sfs.get_feature_names_out()
            # X_selected = X[Selected_Features]
            # # Add a constant (intercept) to the model
            # X_selected_with_const = sm.add_constant(X_selected)
            # X_with_const = sm.add_constant(X)
            #
            # # Fit the model with statsmodels
            # model_selected = sm.OLS(y, X_selected_with_const).fit()
            # model_full = sm.OLS(y, X_with_const).fit()
            #
            # # Print the summary of the selected model
            # print(model_selected.summary())
            #
            # # Print the summary of the full model
            # print(model_full.summary())


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
        datasetCorr_path = f'{self.path}\Participants\Dataset\Corr_all.csv'
        TotalCorr.to_csv(datasetCorr_path)

        # Example of loading your dataset (replace with your data)
        # data = pd.read_csv('your_data.csv')

        # Here, 'dependent_variable' is the outcome variable, 'fixed_effects_variable' is the fixed effect,
        # and 'random_effect_grouping' is the random effect grouping (e.g., participant IDs).

        # Mixed Linear Model
        # Replace 'dependent_variable' with the column name of the outcome,
        # 'fixed_effects_variable' with your fixed effect predictor, and
        # 'random_effect_grouping' with the grouping factor for the random effect.
        # Load your data


    def Linear_Mixed_Effects_Models(self):

        # Load the dataset
        data = pd.read_csv(r'D:\Human Bio Signals Analysis\Participants\Dataset\Dataset_1_EDA.csv')

        # Perform mean imputation for all columns except 'participant' and 'Part'
        data_imputed = data.copy()
        numeric_columns = data_imputed.columns.difference(['participant', 'Part'])
        data_imputed[numeric_columns] = data_imputed[numeric_columns].fillna(data_imputed[numeric_columns].mean())

        # Fit the Linear Mixed-Effects Model (LMM)
        # Using 'Stress_Report' as the dependent variable and 'participant' as the random effect
        # The other physiological signals will be the fixed effects

        model = smf.mixedlm(
            "Stress_Report ~ ECG_Rate_Mean + HRV_MeanNN + HRV_SDNN + HRV_RMSSD + HRV_pNN50 + HRV_pNN20 + "
            "HRV_VHF + HRV_VLF + HRV_LF + HRV_HF + HRV_LFHF + HRV_LFn + HRV_HFn + HRV_LnHF + HRV_TP + "
            "HRV_ShanEn + Diaph_RSP_Rate_Mean + Diaph_BRV + Diaph_BRavNN + Diaph_RSP_Phase_Duration_Expiration + "
            "Diaph_RSP_Phase_Duration_Inspiration + Diaph_RSP_Phase_Duration_Ratio + Diaph_RSP_RVT + "
            "Diaph_RSP_Symmetry_PeakTrough + Diaph_RSP_Symmetry_RiseDecay + Chest_RSP_Rate_Mean + Chest_BRV + "
            "Chest_BRavNN + Chest_RSP_Phase_Duration_Expiration + Chest_RSP_Phase_Duration_Inspiration + "
            "Chest_RSP_Phase_Duration_Ratio + Chest_RSP_RVT + Chest_RSP_Symmetry_PeakTrough + Chest_RSP_Symmetry_RiseDecay + "
            "EDA_Tonic_Mean + EDA_Phasic_Mean + SCR_Peaks_Count + SCR_Amplitude_Mean",
            data_imputed,
            groups=data_imputed["participant"])

        # Fit the model
        result = model.fit()

        # Print the model summary
        print(result.summary())

    def StatisticalTest(self):
        # Load file paths
        SubjectData_path = fr'{self.path}\Participants\Dataset\Subjective\SubjectiveDataset.csv'
        PerformanceData_path = fr'{self.path}\Participants\Dataset\Performance\performance.csv'

        # Load datasets
        df_subjective = pd.read_csv(SubjectData_path)
        df_perf = pd.read_csv(PerformanceData_path)

        # Get unique tasks from both datasets
        subjective_tasks = df_subjective['Task'].unique()
        performance_tasks = df_perf['Task'].unique()
        posthoc_results = {}

        # ---- Subjective Measures: Stress & Fatigue ----
        for task in subjective_tasks:
            task_data = df_subjective[df_subjective['Task'] == task]

            # Stress
            stress_groups = [task_data[task_data['Group'] == g]['Stress'].dropna() for g in task_data['Group'].unique()]
            stress_groups = [g for g in stress_groups if len(g) > 0]
            if len(stress_groups) >= 2 and len(np.unique(np.concatenate(stress_groups))) > 1:
                stat, p = kruskal(*stress_groups)
                if p < 0.05:
                    dunn = sp.posthoc_dunn(task_data, val_col='Stress', group_col='Group', p_adjust='fdr_bh')
                    posthoc_results[f"{task} - Stress"] = dunn

            # Fatigue
            fatigue_groups = [task_data[task_data['Group'] == g]['Fatigue'].dropna() for g in
                              task_data['Group'].unique()]
            fatigue_groups = [g for g in fatigue_groups if len(g) > 0]
            if len(fatigue_groups) >= 2 and len(np.unique(np.concatenate(fatigue_groups))) > 1:
                stat, p = kruskal(*fatigue_groups)
                if p < 0.05:
                    dunn = sp.posthoc_dunn(task_data, val_col='Fatigue', group_col='Group', p_adjust='fdr_bh')
                    posthoc_results[f"{task} - Fatigue"] = dunn

        # ---- Performance Measure: Accuracy (correct) ----
        for task in performance_tasks:
            task_data = df_perf[df_perf['Task'] == task]

            accuracy_groups = [task_data[task_data['Group'] == g]['correct'].dropna() for g in
                               task_data['Group'].unique()]
            accuracy_groups = [g for g in accuracy_groups if len(g) > 0]

            if len(accuracy_groups) >= 2 and len(np.unique(np.concatenate(accuracy_groups))) > 1:
                stat, p = kruskal(*accuracy_groups)
                if p < 0.05:
                    dunn = sp.posthoc_dunn(task_data, val_col='correct', group_col='Group', p_adjust='fdr_bh')
                    posthoc_results[f"{task} - Accuracy"] = dunn

        # ---- Display Results ----
        if posthoc_results:
            for label, table in posthoc_results.items():
                print(f"\n🔍 Post-hoc Dunn Test for {label}")
                print(table)
        else:
            print("No significant Kruskal-Wallis results found for post-hoc testing.")

    def BetaReggresion(self):
        # Load performance data
        performance_path = fr"{self.path}\Participants\Dataset\Performance\performance.csv"
        df = pd.read_csv(performance_path)
        df.columns = [col.strip() for col in df.columns]

        # Compute accuracy
        df['accuracy'] = df['Correct'] / df['Total']
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['accuracy'])

        # Adjust accuracy to avoid 0 or 1
        df['accuracy_adj'] = (df['accuracy'] * (len(df) - 1) + 0.5) / len(df)

        # Aggregate
        agg_df = df.groupby(['ID', 'Task', 'Level', 'Group'], as_index=False)['accuracy_adj'].mean()

        # Beta regression with interaction
        model = smf.glm(
            formula="accuracy_adj ~ C(Group) * C(Level) + C(Task)",
            data=agg_df,
            family=sm.families.Binomial(link=sm.families.links.logit())
        ).fit()

        print("\n=== Beta Regression Summary ===")
        print(model.summary())

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(data=agg_df, x='Group', y='accuracy_adj', hue='Level', ci='sd')
        plt.title('Adjusted Accuracy by Group and Level')
        plt.ylabel('Adjusted Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def GroupDiffPlot(self):
        # ── Load and plot Subjective Ratings ───────────────────────────────
        SubjectData_path = fr'{self.path}\Participants\Dataset\Subjective\SubjectiveDataset.csv'
        SubjectDat = pd.read_csv(SubjectData_path)

        group_palette = {
            'breath': '#FF9999',
            'music': '#99CCFF',
            'control': '#99FF99'
        }
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(
            data=SubjectDat,
            x="Task",
            y="Stress",
            hue="Group",
            palette=group_palette
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title("Stress Ratings per Task by Group (Boxplot)")
        plt.xlabel("Cognitive Task")
        plt.ylabel("Stress Rating")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fr'{self.path}\Participants\Dataset\Subjective\Stress Rating_boxplot.png')
        plt.show()
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(
            data=SubjectDat,
            x="Task",
            y="Stress_N",
            hue="Group",
            palette=group_palette
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title("Stress Normalized by start Ratings per Task by Group (Boxplot)")
        plt.xlabel("Cognitive Task")
        plt.ylabel("Stress Change %")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fr'{self.path}\Participants\Dataset\Subjective\Stress Change boxplot.png')
        plt.show()
        # ── Load and prepare Performance Data ─────────────────────────────
        PreformanceData_path = fr'{self.path}\Participants\Dataset\Performance\performance.csv'
        PerformanceData = pd.read_csv(PreformanceData_path)

        # Fix column names to match the updated dataset structure
        PerformanceData['Task'] = PerformanceData['Task'].astype(str).str.strip()
        PerformanceData['Task_Level'] = PerformanceData['Task'] + ' | ' + PerformanceData['Level']
        PerformanceData['correct'] = PerformanceData['correct'].astype(int)

        # ── Plot RT and Accuracy Side-by-Side ─────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=False)

        # ── Response Time Boxplot ──────────────────────
        sns.boxplot(
            data=PerformanceData,
            x='Task_Level',
            y='RT',
            hue='Group',
            palette=group_palette,
            ax=axes[0]
        )
        axes[0].set_title("Response Time by Task and Level, Split by Group")
        axes[0].set_xlabel("Task | Level")
        axes[0].set_ylabel("Response Time (RT)")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True)

        # ── Accuracy Barplot with 95% Confidence Intervals ───────────────
        sns.barplot(
            data=PerformanceData,
            x='Task_Level',
            y='correct',
            hue='Group',
            estimator='mean',
            ci=95,
            n_boot=5000,
            palette=group_palette,
            ax=axes[1]
        )

        axes[1].set_title("Mean Accuracy by Task and Level (95% CI)")
        axes[1].set_xlabel("Task | Level")
        axes[1].set_ylabel("Accuracy (Proportion Correct ± 95% CI)")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True)
        axes[1].legend_.remove()

        # Save and show
        output_path = fr'{self.path}\Participants\Dataset\Performance\Performance_boxplot_from_summary_CI.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()

