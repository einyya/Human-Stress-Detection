from matplotlib import patheffects
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.inspection import permutation_importance
import scikit_posthocs as sp
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,confusion_matrix)
from scipy.stats import kruskal
from sklearn.metrics import ConfusionMatrixDisplay
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBRegressor, plot_tree as xgb_plot_tree
from tqdm import tqdm
import statsmodels.formula.api as smf


from sklearn.pipeline import Pipeline

class AnalysisData():
    def __init__(self,Directory,ex_col,Prediction_Targets):
        self.path = Directory
        # self.sorted_DATA = sorted_DATA
        # self.sampling_frequency = sampling_frequency
        self.segment_DATA = pd.DataFrame()
        self.preprocessed_DATA = pd.DataFrame()
        self.window_samples = 0
        self.ex_col=ex_col
        self.Prediction_Targets=Prediction_Targets

    def _binary_metrics(y_true, y_pred):
        """Return accuracy, precision, recall (== sensitivity) and specificity."""
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        sens = recall_score(y_true, y_pred,   zero_division=0)     # sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else np.nan            # specificity
        return acc, prec, sens, spec

    @staticmethod
    def safe_plot_xgb_tree(xgb_model, out_dir, model_name="XGBoost", tree_idx=0):
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(20, 12))
        xgb_plot_tree(xgb_model, tree_idx=tree_idx, rankdir="LR")
        plt.title(f"{model_name} - tree {tree_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_tree{tree_idx}.png"), dpi=200)
        plt.close()

    @staticmethod
    def save_linear_equation(model, feature_names, out_dir, model_name):
        """Save regression equation in human-readable form + CSV of coefficients"""
        os.makedirs(out_dir, exist_ok=True)
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", 0.0)
        if coef is None:
            return
        if hasattr(coef, "shape") and coef.ndim > 1:  # flatten if multi-output
            coef = coef.ravel()

        # Sort coefficients by absolute value
        order = np.argsort(-np.abs(coef))
        lines = [f"y = {intercept:.6f} "]
        for idx in order:
            lines.append(f"+ ({coef[idx]:.6f})*{feature_names[idx]}")
        eq_text = " \\\n    ".join(lines)

        # Save equation as text
        with open(os.path.join(out_dir, f"{model_name}_equation.txt"), "w", encoding="utf-8") as f:
            f.write(eq_text)

        # Save coefficients as CSV
        pd.DataFrame({
            "feature": np.array(feature_names)[order],
            "coef": coef[order]
        }).to_csv(os.path.join(out_dir, f"{model_name}_coefficients.csv"), index=False)

    @staticmethod
    def save_decision_tree_plots(tree_model, feature_names, out_dir, title="DecisionTree"):
        """Save decision tree as text rules + PNG plot (limited depth for readability)"""
        os.makedirs(out_dir, exist_ok=True)

        # Export rules as text
        rules = export_text(tree_model, feature_names=list(feature_names))
        with open(os.path.join(out_dir, f"{title}_rules.txt"), "w", encoding="utf-8") as f:
            f.write(rules)

        # Plot tree (top 3 levels only for readability)
        plt.figure(figsize=(20, 12))
        plot_tree(tree_model, feature_names=feature_names, filled=True, impurity=False,
                  rounded=True, max_depth=3)
        plt.title(f"{title} - top depth 3")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{title}_tree_depth3.png"), dpi=200)
        plt.close()

    @staticmethod
    def pick_representative_rf_tree(rf_model):
        """Pick the most complex tree from RandomForest (largest number of nodes)"""
        ests = getattr(rf_model, "estimators_", [])
        if not ests:
            return None
        sizes = [est.tree_.node_count for est in ests]
        return ests[int(np.argmax(sizes))]

    @staticmethod
    def save_xgb_tree_plots(xgb_model, feature_names, out_dir, model_name="XGBoost"):
        """Save XGBoost tree visualization + dump text of all trees"""
        os.makedirs(out_dir, exist_ok=True)
        # Plot the first tree
        plt.figure(figsize=(20, 12))
        xgb_plot_tree(xgb_model, tree_idx=0, rankdir="LR")
        plt.title(f"{model_name} - tree 0")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_tree0.png"), dpi=200)
        plt.close()


        # Save dump of all trees (text format)
        try:
            booster = xgb_model.get_booster()
            dump = booster.get_dump(with_stats=True)
            with open(os.path.join(out_dir, f"{model_name}_dump.txt"), "w", encoding="utf-8") as f:
                for i, tree_txt in enumerate(dump):
                    f.write(f"----- Tree {i} -----\n")
                    f.write(tree_txt)
                    f.write("\n\n")
        except Exception as e:
            with open(os.path.join(out_dir, f"{model_name}_dump_error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))
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
        base_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }

        # Hyperparameter grids
        param_grids = {
            'LinearRegression': {},  # <-- Add this line to fix the KeyError
            'Ridge': {'alpha': [0.1, 1, 10]},
            'Lasso': {'alpha': [0.01, 0.1, 1]},
            'DecisionTree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1]
            }
        }

        window_sizes = [5, 10, 30, 60]
        overlaps = [0.0, 0.5]        # Normalize X using Z-score normalization

        # Load participant IDs
        participants_csv = os.path.join(self.path, 'Participants', 'participation management.csv')
        participants = pd.read_csv(participants_csv)
        all_ids = participants['code'].dropna().astype(int).unique()

        signals = ['All']
        for prediction in tqdm(self.Prediction_Targets, desc="🔍 Prediction Target"):
            print(f"\n=== Prediction Target: {prediction} ===")

            config_options = {
                fr'{prediction}_Add_Level_Type': ["Level", "Test_Type"],
                fr'{prediction}_Add_Level': ["Level"],
                fr'{prediction}_Base': []

            }
            for config_name, add_categoricals in config_options.items():
                out_dir = fr"C:\Users\e3bom\Desktop\Human Bio Signals Analysis\Participants\Dataset\ML\Prediction\{prediction}\{config_name}"
                os.makedirs(out_dir, exist_ok=True)

                all_summary = []

                for signal in tqdm(signals, desc="📡 Signals", leave=False):
                    print(f"\n--- Signal: {signal} ---")
                    results = []
                    importances = {name: [] for name in base_models}
                    best_ws = {name: {'window': None, 'overlap': None, 'mse': np.inf} for name in base_models}
                    best_params = {}

                    hyper_dir = os.path.join(out_dir, "hyperparameters")
                    ws_path = os.path.join(hyper_dir, "best_ws.csv")
                    params_path = os.path.join(hyper_dir, "best_params.csv")
                    for repeat in tqdm(range(n_repeats), desc="🔁 Repeats", leave=False):
                        print(f"▶️ Repeat {repeat + 1}/{n_repeats}")
                        train_ids, test_ids = train_test_split(
                            all_ids, test_size=0.2, random_state=42 + repeat
                        )

                        if os.path.exists(ws_path) and os.path.exists(params_path):
                            ws_df = pd.read_csv(ws_path)
                            params_df = pd.read_csv(params_path)
                            for _, row in ws_df.iterrows():
                                best_ws[row['model']] = {'window': int(row['window']), 'overlap': float(row['overlap']),
                                                         'mse': float(row['mse'])}
                            for _, row in params_df.iterrows():
                                model = row['model']
                                param_dict = {}
                                for k in row.index:
                                    if k != 'model' and pd.notnull(row[k]):
                                        v = row[k]
                                        # Convert known integer parameters to int
                                        if k in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
                                            v = int(v)
                                        param_dict[k] = v
                                best_params[model] = param_dict
                            print("✅ Loaded saved hyperparameters.")
                        else:
                            for name, base_model in tqdm(base_models.items(), desc="🧠 Models", leave=False):
                                best_config = {'window': None, 'overlap': None, 'params': {}, 'mse': np.inf}
                                for ws in window_sizes:
                                    for ov in overlaps:
                                        print(fr'{name}_{ws}_{ov}')
                                        file_path = os.path.join(
                                            self.path, 'Participants', 'Dataset', 'Dataset_By_Window',
                                            'Clean_Data', f'Dataset_{ws}s_{int(ov * 100)}.csv'
                                        )
                                        if not os.path.exists(file_path):
                                            continue

                                        df = pd.read_csv(file_path).dropna(subset=[prediction])
                                        df[prediction] = pd.to_numeric(df[prediction], errors="coerce")
                                        df = df.dropna(subset=[prediction])
                                        if signal != 'All':
                                            cols = ['ID', 'Group', 'Time'] + [c for c in df.columns if c.startswith(signal)]
                                            df = df[cols + [prediction]]

                                        df_train = df[df['ID'].isin(train_ids)].copy()
                                        if df_train.empty:
                                            continue

                                        feature_cols = [c for c in df.columns if c not in self.ex_col]
                                        scaler = StandardScaler()
                                        X_train = df_train[feature_cols].copy()
                                        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns,
                                                               index=X_train.index)
                                        if add_categoricals:
                                            cat_df = pd.get_dummies(df_train[add_categoricals], drop_first=True)
                                            X_train = pd.concat([X_train, cat_df], axis=1)
                                        feature_cols_extended = X_train.columns.tolist()
                                        gkf = GroupKFold(n_splits=3)
                                        grid = HalvingGridSearchCV(
                                            estimator=clone(base_model),
                                            param_grid=param_grids[name],
                                            cv=gkf,
                                            scoring="neg_mean_squared_error",
                                            n_jobs=-1,
                                            factor=2,
                                            resource="n_samples",
                                            max_resources="auto",
                                            aggressive_elimination=False,
                                            refit=False
                                        )
                                        grid.fit(X_train, df_train[prediction], groups=df_train['ID'])
                                        mean_mse = -grid.best_score_
                                        print(f"Model: {name}, Window: {ws}, Overlap: {ov}, MSE: {mean_mse:.4f}")

                                        if mean_mse < best_config['mse']:
                                            best_config.update({
                                                'window': ws,
                                                'overlap': ov,
                                                'params': grid.best_params_,
                                                'mse': mean_mse
                                            })

                                best_ws[name] = {
                                    'window': best_config['window'],
                                    'overlap': best_config['overlap'],
                                    'mse': best_config['mse']
                                }
                                best_params[name] = best_config['params']
                            os.makedirs(hyper_dir, exist_ok=True)
                            pd.DataFrame([{'model': k, **v} for k, v in best_ws.items()]).to_csv(ws_path, index=False)
                            pd.DataFrame([dict({'model': k}, **v) for k, v in best_params.items()]).to_csv(params_path, index=False)
                            print("✅ Saved hyperparameters.")
                        # Evaluation
                        for name in tqdm(base_models, desc="🔍 Evaluation", leave=False):
                            ws = best_ws[name]['window']
                            ov = best_ws[name]['overlap']
                            if ws is None:
                                continue

                            file_path = os.path.join(
                                self.path, 'Participants', 'Dataset', 'Dataset_By_Window',
                                'Clean_Data', f'Dataset_{ws}s_{int(ov * 100)}.csv'
                            )
                            df = pd.read_csv(file_path).dropna(subset=[prediction])
                            df[prediction] = pd.to_numeric(df[prediction], errors="coerce")
                            df = df.dropna(subset=[prediction])

                            if signal != 'All':
                                cols = ['ID', 'Group', 'Time'] + [c for c in df.columns if c.startswith(signal)]
                                df = df[cols + [prediction]]

                            df_train = df[df['ID'].isin(train_ids)]
                            df_test = df[df['ID'].isin(test_ids)]
                            feature_cols = [c for c in df.columns if c not in self.ex_col]

                            model = clone(base_models[name]).set_params(**best_params[name])
                            scaler = StandardScaler()
                            X_train = df_train[feature_cols].copy()
                            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns,
                                                   index=X_train.index)
                            if add_categoricals:
                                cat_df = pd.get_dummies(df_train[add_categoricals], drop_first=True)
                                X_train = pd.concat([X_train, cat_df], axis=1)

                            feature_cols_extended = X_train.columns.tolist()
                            model.fit(X_train, df_train[prediction])
                            feat_names_for_output = feature_cols_extended if 'feature_cols_extended' in locals() else feature_cols

                            # Create output directory
                            model_out_dir = os.path.join(out_dir, "Model_Visuals", signal, name, f"Repeat_{repeat + 1}")
                            os.makedirs(model_out_dir, exist_ok=True)

                            # --- Linear models ---
                            if hasattr(model, "coef_"):
                                self.save_linear_equation(model, feat_names_for_output, model_out_dir, model_name=name)
                            # --- DecisionTree ---
                            if isinstance(model, DecisionTreeRegressor):
                                try:
                                    self.save_decision_tree_plots(model, feat_names_for_output, model_out_dir, title=f"{name}")
                                except Exception as e:
                                    with open(os.path.join(model_out_dir, f"{name}_tree_error.txt"), "w",
                                              encoding="utf-8") as f:
                                        f.write(str(e))

                            # --- RandomForest ---
                            if isinstance(model, RandomForestRegressor):
                                try:
                                    rep_tree = self.pick_representative_rf_tree(model)
                                    if rep_tree is not None:
                                        self.save_decision_tree_plots(rep_tree, feat_names_for_output, model_out_dir,
                                                                 title=f"{name}_repTree")
                                except Exception as e:
                                    with open(os.path.join(model_out_dir, f"{name}_rf_tree_error.txt"), "w",
                                              encoding="utf-8") as f:
                                        f.write(str(e))

                            # --- XGBoost ---
                            if isinstance(model, XGBRegressor):
                                try:
                                    self.safe_plot_xgb_tree(model, model_out_dir, model_name=name, tree_idx=0)
                                except Exception as e:
                                    with open(os.path.join(model_out_dir, f"{name}_xgb_plot_error.txt"), "w",
                                              encoding="utf-8") as f:
                                        f.write(str(e))
                            X_test = df_test[feature_cols].copy()
                            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                            if add_categoricals:
                                cat_df_test = pd.get_dummies(df_test[add_categoricals], drop_first=True)
                                cat_df_test = cat_df_test.reindex(columns=cat_df.columns, fill_value=0)
                                X_test = pd.concat([X_test, cat_df_test], axis=1)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(df_test[prediction], y_pred)
                            r2 = r2_score(df_test[prediction], y_pred)
                            n = len(y_pred)
                            k = df_test[feature_cols].shape[1]
                            rss = np.sum((df_test[prediction] - y_pred) ** 2)
                            # Adjusted R2
                            r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                            # AIC
                            aic = n * np.log(rss / n) + 2 * k
                            # BIC
                            bic = n * np.log(rss / n) + k * np.log(n)
                            var_y = np.var(df_test[prediction], ddof=1)
                            print(
                                f"✅ {name}: MSE={mse:.3f}, R2={r2:.3f}, Adj.R2={r2_adj:.3f}, AIC={aic:.1f}, BIC={bic:.1f}")

                            results.append({
                                'Signal': signal,
                                'Repeat': repeat + 1,
                                'Model': name,
                                'Window (s)': ws,
                                'Overlap (%)': int(ov * 100),
                                'MSE': mse,
                                'R2': r2,
                                'Adj_R2': r2_adj,
                                'AIC': aic,
                                'BIC': bic,
                                'Var_Y': var_y,
                                **{f'param_{k}': v for k, v in best_params[name].items()}
                            })
                            print({
                                'Signal': signal,
                                'Repeat': repeat + 1,
                                'Model': name,
                                'Window (s)': ws,
                                'Overlap (%)': int(ov * 100),
                                'MSE': mse,
                                'R2': r2,
                                'Adj_R2': r2_adj,
                                'AIC': aic,
                                'BIC': bic,
                                **{f'param_{k}': v for k, v in best_params[name].items()}
                            })

                            # Save feature importance or coefficients
                            if hasattr(model, 'feature_importances_'):
                                # Tree-based models
                                imp_values = model.feature_importances_
                                imp_name = "Feature_Importance"
                            elif hasattr(model,'coef_'):
                                # Linear models
                                imp_values = np.abs(model.coef_)
                                imp_name = "Coefficients"
                            else:
                                imp_values = None

                            if imp_values is not None:
                                try:
                                    if 'feature_cols_extended' in locals():
                                        imp = pd.Series(imp_values, index=feature_cols_extended).sort_values(
                                            ascending=False)
                                    else:
                                        imp = pd.Series(imp_values, index=feature_cols).sort_values(ascending=False)
                                except Exception as e:
                                    print(f"⚠️ Feature importance length mismatch: {e}")
                                    imp = None
                                imp_dir = os.path.join(out_dir, "Feature Importance", signal, name, f"Repeat_{repeat + 1}")
                                os.makedirs(imp_dir, exist_ok=True)

                                imp.to_csv(os.path.join(imp_dir, f"{imp_name}.csv"))
                                plt.figure(figsize=(10, 5))
                                imp.plot.bar()
                                plt.title(f"{name} {imp_name} - {signal} - Repeat {repeat + 1}")
                                plt.tight_layout()
                                plt.savefig(os.path.join(imp_dir, f"{imp_name}.png"))
                                plt.close()

                                importances[name].append(imp)
                        # Save results per signal
                    results_df = pd.DataFrame(results).round(3)
                    results_df.to_csv(os.path.join(out_dir, f'Results_{signal}.csv'), index=False)

                    # Save summary including all metrics and window/overlap
                    summary = (
                        results_df
                        .groupby(['Model', 'Window (s)', 'Overlap (%)'])[
                            ['MSE', 'R2', 'Adj_R2', 'AIC', 'BIC', 'Var_Y']
                        ]
                        .agg(['mean', 'std'])
                        .reset_index()
                    )
                    summary.insert(0, 'Signal', signal)
                    summary.to_csv(os.path.join(out_dir, f'Summary_{signal}.csv'), index=False)
                    all_summary.append(summary)

                    # Combined feature importance
                    for name, imps in importances.items():
                        if imps:
                            imp_df = pd.concat(imps, axis=1).fillna(0)
                            imp_df.columns = [f"Repeat_{i + 1}" for i in range(len(imps))]
                            imp_df["Mean"] = imp_df.mean(axis=1)
                            imp_df["Std"] = imp_df.std(axis=1)

                            summary_dir = os.path.join(out_dir, "Feature Importance", signal, name, "Summary")
                            os.makedirs(summary_dir, exist_ok=True)

                            imp_df.sort_values("Mean", ascending=False).to_csv(
                                os.path.join(summary_dir, "Feature_Importance_Summary.csv")
                            )

                            plt.figure(figsize=(12, 5))
                            imp_df["Mean"].sort_values(ascending=False).plot.bar(yerr=imp_df["Std"])
                            plt.title(f"{name} Feature Importance Summary - {signal}")
                            plt.tight_layout()
                            plt.savefig(os.path.join(summary_dir, "Feature_Importance_Summary.png"))
                            plt.close()

                # Save combined summary across signals
                combined_df = pd.concat(all_summary, ignore_index=True)
                combined_df.to_csv(os.path.join(out_dir, 'Summary_AllSignals.csv'), index=False)
    def ML_models_Classification(self, n_repeats=9, no_breath_data=False, clases_3=False):
        window_sizes = [5, 10, 30, 60]
        overlaps = [0.0, 0.5]

        base_models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SVM linear': LinearSVC(random_state=42, max_iter=5000),
            'SVM rbf': SVC(probability=True, random_state=42),
            'LogReg': LogisticRegression(random_state=42, max_iter=5000),
        }

        param_grids = {
            'DecisionTree': {
                'max_depth': [None, 3, 6, 10, 20],
                'min_samples_split': [2, 5, 10],
            },
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 3, 6, 10, 20],
                'min_samples_split': [2, 5, 10],
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [None,3, 6, 10, 20],
                'learning_rate': [0.01, 0.1]
            },
            'SVM linear': {
                'C': [0.1, 1, 10]
            },
            'SVM rbf': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'LogReg': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],  # נשארים עם L2 כדי להימנע מתלות ב-saga ללוקאל
                'solver': ['lbfgs', 'liblinear']  # lbfgs תומך בריבוי מחלקות, liblinear טוב לבינארי/OvR
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
            # signals = ['All']
            signals = ['HRV', 'RSP_C', 'RSP_D', 'EDA', 'All']
        all_summary = []
        for signal in signals:
            print(f"\n📊 Evaluating signal: {signal} no_breath_data {no_breath_data} and clases_3 {clases_3}")
            results = []
            importances = {name: [] for name in base_models}
            best_ws = {name: {'window': None, 'overlap': None, 'f1': -np.inf} for name in base_models}
            best_params = {}
            for repeat in tqdm(range(n_repeats), desc="🔁 Repeats", leave=False):
                print(f"▶️ Repeat {repeat + 1}/{n_repeats}")
                # -----------Iteration 1-Split Train Test------------------------------
                train_ids, test_ids = train_test_split(
                    all_ids, test_size=0.2, random_state=42 + repeat
                )
                run_full = (repeat == 0)
                iter_to_run = [1, 2, 3, 4] if run_full else [1, 4]
                # -----------Iteration 2-Choose Window Size------------------------------
                # -----------Iteration 2-Grid Search for Window, Overlap, and Hyperparameters------------------------------
                if 2 in iter_to_run:
                    dir_path = fr"{out_dir}\hyperparameters\{signal}"

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
                                            selected_columns = self.ex_col + [col for col in df.columns if
                                                                               col.startswith(signal + '_')]
                                            df = df[selected_columns]
                                        df_train = df[df['ID'].isin(train_ids)].copy()
                                        feature_cols = [c for c in df.columns if
                                                        c not in self.ex_col]
                                        if clases_3:
                                            df_train['Class'] = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                                            df_train.loc[df_train['Level'] == 'hard', 'Class'] = 2
                                            df_train.loc[df_train['Level'] == 'medium', 'Class'] = 2
                                            y_tr = df_train['Class'].astype(int)
                                        else:
                                            y_tr = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0}).astype(int)

                                        groups = df_train['ID']

                                        if len(df_train) == 0:
                                            continue
                                        scaler = StandardScaler()
                                        X_tr = df_train[feature_cols].copy()
                                        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
                                        gkf = GroupKFold(n_splits=3)
                                        grid = HalvingGridSearchCV(
                                            estimator=clone(base_model),
                                            param_grid=param_grids[name],
                                            cv=gkf,
                                            scoring="f1_weighted",
                                            n_jobs=-1,
                                            factor=2,
                                            resource="n_samples",
                                            max_resources="auto",
                                            aggressive_elimination=False,
                                            refit=False,
                                            verbose=3
                                        )

                                        grid.fit(X_tr, y_tr, groups=groups)
                                        mean_f1 = grid.best_score_
                                        print(
                                            f"WS={ws}s, OV={int(ov * 100)}%, F1={mean_f1:.3f}, Model={name},Params={grid.best_params_}")

                                        # Update best configuration if this is better
                                        if mean_f1 > best_config['f1']:
                                            best_config.update({
                                                'window': ws,
                                                'overlap': ov,
                                                'params': grid.best_params_,
                                                'f1': mean_f1
                                            })
                                            print(f" → New best for {name}!")

                                    except Exception as e:
                                        print(f"      Error with WS={ws}s, OV={int(ov * 100)}%: {str(e)}")

                            # Store best configuration for this model
                            best_ws[name] = {
                                'window': best_config['window'],
                                'overlap': best_config['overlap'],
                                'f1': best_config['f1']
                            }
                            best_params[name] = best_config['params']
                            print(fr"{name} best_params no_breath_data {no_breath_data} and clases_3 {clases_3}.to_csv")
                            hyper_dir = os.path.join(out_dir, 'hyperparameters', signal)
                            os.makedirs(hyper_dir, exist_ok=True)

                            pd.DataFrame(best_params).to_csv(os.path.join(hyper_dir, 'best_params.csv'))
                            pd.DataFrame(best_ws).to_csv(os.path.join(hyper_dir, 'best_ws.csv'))
                            pd.DataFrame([best_config]).to_csv(os.path.join(hyper_dir, 'best_config.csv'))

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
                        selected_columns = self.ex_col + [col for col in df.columns if
                                                           col.startswith(signal + '_')]
                        df = df[selected_columns]
                    df_train = df[df['ID'].isin(train_ids)].copy()
                    feature_cols = [c for c in df.columns if
                                    c not in self.ex_col]

                    if clases_3:
                        df_train['Class'] = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                        df_train.loc[df_train['Level'] == 'hard', 'Class'] = 2
                        df_train.loc[df_train['Level'] == 'medium', 'Class'] = 2
                        y =df_train['Class'].astype(int)
                    else:
                        y = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0}).astype(int)
                    model = clone(base_models[name]).set_params(**best_params.get(name, {}))
                    X_train = df_train[feature_cols]
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    model.fit(X_train, y)
                    params = best_params[name]

                    df_test = df[df['ID'].isin(test_ids)].copy()
                    X_te = df_test[feature_cols]
                    X_te = scaler.transform(X_te)

                    if clases_3:
                        df_test['Class'] = df_test['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                        df_test.loc[df_test['Level'] == 'hard', 'Class'] = 2
                        df_test.loc[df_test['Level'] == 'medium', 'Class'] = 2
                        y_te = df_test['Class'].astype(int)
                    else:
                        y_te = df_test['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0}).astype(int)

                    y_pred = model.predict(X_te)
                    # ----------------- Model visuals per run -----------------
                    vis_dir = os.path.join(out_dir, "Model Visuals", signal, name)
                    os.makedirs(vis_dir, exist_ok=True)
                    run_tag = f"Run{repeat + 1}_WS{ws}_OV{int(ov * 100)}"

                    try:
                        if name == 'DecisionTree' and hasattr(model, "tree_"):
                            plt.figure(figsize=(18, 10))
                            plot_tree(
                                model,
                                feature_names=feature_cols,
                                class_names=[str(c) for c in sorted(np.unique(y))],
                                filled=True,
                                impurity=True,
                                rounded=True,
                                fontsize=8
                            )
                            plt.title(f"Decision Tree - {signal} - {run_tag}")
                            plt.tight_layout()
                            tree_path = os.path.join(vis_dir, f"{run_tag}_DecisionTree.png")
                            plt.savefig(tree_path, dpi=200)
                            plt.close()
                            # שמירה טקסטואלית אופציונלית של העץ כבונוס
                            # from sklearn.tree import export_text
                            # with open(os.path.join(vis_dir, f"{run_tag}_DecisionTree.txt"), "w", encoding="utf-8") as f:
                            #     f.write(export_text(model, feature_names=list(feature_cols)))

                        if name == 'LogReg' and hasattr(model, "coef_"):
                            coefs = model.coef_  # (n_classes or 1, n_features) או (n_features,)
                            classes_ = getattr(model, "classes_", np.arange(coefs.shape[0] if coefs.ndim > 1 else 1))
                            coef_df_list = []

                            # --- Save the logistic regression equation as a text file ---
                            try:
                                eq_path = os.path.join(vis_dir, f"{run_tag}_LogReg_Equation.txt")
                                with open(eq_path, "w", encoding="utf-8") as f:
                                    f.write("Logistic Regression equations - features are StandardScaler scaled\n")
                                    f.write(f"Signal: {signal} | Run: {run_tag}\n")
                                    f.write(f"Classes: {list(classes_)}\n\n")

                                    # Binary case: single equation
                                    if coefs.ndim == 1 or coefs.shape[0] == 1:
                                        intercept = float(model.intercept_[0]) if hasattr(model, "intercept_") else 0.0
                                        terms = []
                                        for j, feat in enumerate(feature_cols):
                                            coef_j = coefs[j] if coefs.ndim == 1 else coefs[0, j]  # ← התיקון כאן
                                            terms.append(f"{float(coef_j):.6f}*{feat}")
                                        f.write("Binary setting\n")
                                        f.write(f"logit = {intercept:.6f} + " + " + ".join(terms) + "\n")
                                        f.write("P(y=positive) = 1 / (1 + exp(-logit))\n")
                                    else:
                                        # Multiclass OvR: one equation per class
                                        for i, cls in enumerate(np.atleast_1d(classes_)):
                                            intercept = float(model.intercept_[i]) if hasattr(model,
                                                                                              "intercept_") else 0.0
                                            terms = [f"{float(coefs[i, j]):.6f}*{feature_cols[j]}" for j in
                                                     range(len(feature_cols))]
                                            f.write(f"Class {cls} - one vs rest\n")
                                            f.write(f"logit_{cls} = {intercept:.6f} + " + " + ".join(terms) + "\n\n")
                                        f.write("P(y=cls) = exp(logit_cls) / sum_k exp(logit_k)\n")
                                print(f"📝 Saved logistic regression equation to {eq_path}")
                            except Exception as e:
                                print(f"⚠️ Could not save logistic regression equation: {e}")

                            # נשמור CSV של המקדמים לכל מחלקה
                            for i, cls in enumerate(np.atleast_1d(classes_)):
                                cls_coefs = coefs[i] if coefs.ndim > 1 else coefs
                                coef_df = pd.DataFrame({
                                    "Feature": feature_cols,
                                    "Coefficient": cls_coefs
                                }).sort_values("Coefficient", ascending=False)
                                coef_df["AbsCoeff"] = coef_df["Coefficient"].abs()
                                coef_df_list.append(coef_df.assign(Class=cls))

                                # גרף בר של המאפיינים הבולטים - ניקח טופ 20 לפי |coef|
                                top_k = coef_df.sort_values("AbsCoeff", ascending=False).head(20)
                                plt.figure(figsize=(12, 6))
                                plt.bar(top_k["Feature"], top_k["Coefficient"])
                                plt.xticks(rotation=90)
                                plt.xlabel("Feature")
                                plt.ylabel("Coefficient")
                                plt.title(f"Logistic Regression Coefficients - class {cls} - {signal} - {run_tag}")
                                plt.tight_layout()
                                coef_plot_path = os.path.join(vis_dir, f"{run_tag}_LogReg_Coeffs_class{cls}.png")
                                plt.savefig(coef_plot_path, dpi=200)
                                plt.close()

                            # CSV מאוחד לכל המחלקות
                            coef_all = pd.concat(coef_df_list, ignore_index=True)
                            coef_csv_path = os.path.join(vis_dir, f"{run_tag}_LogReg_Coefficients.csv")
                            coef_all.to_csv(coef_csv_path, index=False)
                    except Exception as e:
                        print(f"⚠️ Could not build model visuals for {name} at {run_tag}: {e}")
                    # ---------------------------------------------------------

                    result_row = {
                        'Signal': signal,
                        'Repeat': repeat + 1,
                        'Model': name,
                        'Window (s)': ws,
                        'Overlap (%)': int(ov * 100),
                        'Accuracy': accuracy_score(y_te, y_pred) * 100,
                        'Precision': precision_score(y_te, y_pred, average='macro', zero_division=0) * 100,
                        'Recall': recall_score(y_te, y_pred, average='macro', zero_division=0) * 100,
                        'F1': f1_score(y_te, y_pred, average='macro', zero_division=0) * 100
                    }
                    result_row.update({f'param_{k}': v for k, v in params.items()})
                    print(result_row)
                    results.append(result_row)

                    # Save feature importance plot, CSV, and collect for summary
                    imp = None
                    if hasattr(model, 'feature_importances_'):
                        # Tree-based models
                        imp_values = model.feature_importances_
                        imp = pd.Series(imp_values, index=feature_cols).sort_values(ascending=False)

                    elif hasattr(model, 'coef_'):
                        # Linear SVM (LinearSVC)
                        try:
                            imp_values = np.abs(model.coef_).mean(axis=0)
                            imp = pd.Series(imp_values, index=feature_cols).sort_values(ascending=False)
                        except:
                            pass

                    else:
                        # For non-linear SVM or other models: use permutation importance
                        try:
                            result = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
                            imp_values = result.importances_mean
                            imp = pd.Series(imp_values, index=feature_cols).sort_values(ascending=False)
                        except Exception as e:
                            print(f"⚠️ Could not compute permutation importance for {name}: {e}")

                    if imp is not None:
                        importances[name].append(imp)

            # Save per-signal results
            results_df = pd.DataFrame(results).round(2)
            os.makedirs(out_dir, exist_ok=True)

            output_path = os.path.join(out_dir, "CV_Results", f"NestedCV_Results_{signal}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"✅ Saved results for no_breath_data {no_breath_data} and clases_3 {clases_3} {signal} to NestedCV_Results_{signal}.csv")

            # Summary metrics per model
            summary_metrics = results_df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1"]].agg(
                ["mean", "std"]).round(2)
            optimal_settings = results_df.groupby("Model")[["Window (s)", "Overlap (%)"]].first()
            summary = pd.concat([summary_metrics, optimal_settings], axis=1).reset_index()
            summary.insert(0, "Signal", signal)
            # --- reorder columns in NestedCV_{signal}_Summary.csv ---
            # current columns: 'Model', 'Window (s)', 'Overlap (%)' + MultiIndex metrics
            front = ["Signal", "Model", "Window (s)", "Overlap (%)"]
            metrics_order = [
                ("Accuracy", "mean"), ("Accuracy", "std"),
                ("Precision", "mean"), ("Precision", "std"),
                ("Recall", "mean"), ("Recall", "std"),
                ("F1", "mean"), ("F1", "std"),
            ]

            # make sure columns exist (robust to missing ones)
            existing_metrics = [c for c in metrics_order if c in summary.columns]
            existing_front = [c for c in front if c in summary.columns]

            summary = summary[existing_front + existing_metrics]

            output_path = os.path.join(out_dir, "CV_Summary", f"NestedCV_{signal}_Summary.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            summary.to_csv(output_path, index=False)
            all_summary.append(summary)

            all_imps_long = []
            combined_df = pd.DataFrame()

            # Process each model
            for name, imps in importances.items():
                if imps:
                    # Per-model summary
                    imp_df = pd.concat(imps, axis=1).fillna(0)
                    imp_df.columns = [f"Repeat_{i + 1}" for i in range(len(imps))]
                    imp_df["Mean"] = imp_df.mean(axis=1)
                    imp_df["Std"] = imp_df.std(axis=1)

                    model_data_dir = os.path.join(out_dir, "Feature Importance", signal, name, "data")
                    os.makedirs(model_data_dir, exist_ok=True)
                    imp_df.sort_values("Mean", ascending=False).to_csv(
                        os.path.join(model_data_dir, "Feature_Importance_Summary.csv")
                    )
                    # Create and save feature importance plot (all features)
                    model_plot_dir = os.path.join(out_dir, "Feature Importance", signal, name, "plot")
                    os.makedirs(model_plot_dir, exist_ok=True)

                    plt.figure(figsize=(12, 6))
                    imp_df["Mean"].sort_values(ascending=False).plot(kind='bar')
                    plt.title(f"Feature Importances - {name} ({signal})")
                    plt.ylabel("Mean Importance")
                    plt.xlabel("Feature")
                    plt.tight_layout()
                    plt.savefig(os.path.join(model_plot_dir, "Feature_Importance_Plot.png"))
                    plt.close()
                    # Add mean column to combined DataFrame
                    model_imp_df = pd.concat(imps, axis=1).fillna(0)
                    model_imp_df.columns = [f"{name}_Repeat_{i + 1}" for i in range(len(imps))]
                    model_imp_df[f"{name}_Mean"] = model_imp_df.mean(axis=1)
                    combined_df = pd.concat([combined_df, model_imp_df[[f"{name}_Mean"]]], axis=1)

                    # Long format
                    for i, imp in enumerate(imps):
                        temp = imp.reset_index()
                        temp.columns = ["Feature", "Importance"]
                        temp["Model"] = name
                        temp["Repeat"] = i + 1
                        all_imps_long.append(temp)

            # Combined summary
            if not combined_df.empty:
                combined_df["Combined_Mean"] = combined_df.mean(axis=1)
                combined_df = combined_df.sort_values("Combined_Mean", ascending=False)

                comb_dir_data = os.path.join(out_dir, "Feature Importance", signal, "All Models", "data")
                comb_dir_plot = os.path.join(out_dir, "Feature Importance", signal, "All Models", "plot")
                os.makedirs(comb_dir_data, exist_ok=True)
                os.makedirs(comb_dir_plot, exist_ok=True)

                combined_df.to_csv(os.path.join(comb_dir_data, "Combined_Feature_Importance.csv"))

                # Bar plot
                plt.figure(figsize=(12, 6))
                combined_df["Combined_Mean"].plot(kind="bar")
                if clases_3 and no_breath_data:
                    title_prefix = "3-Class, No Breath -"
                elif clases_3:
                    title_prefix = "3-Class -"
                elif no_breath_data:
                    title_prefix = "No Breath -"
                else:
                    title_prefix = ""

                plt.title(fr"{title_prefix} Combined Feature Importances Across All Models ({signal})")
                plt.ylabel("Mean Importance")
                plt.xlabel("Feature")
                plt.tight_layout()
                plt.savefig(os.path.join(comb_dir_plot, "Combined_Feature_Importance_Plot.png"))
                plt.close()

            # Boxplot across all models/repeats
            if all_imps_long:
                all_df = pd.concat(all_imps_long, axis=0)
                mean_df = all_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
                mean_df.to_csv(os.path.join(comb_dir_data, "Combined_Feature_Importance_Mean.csv"))

                plt.figure(figsize=(14, 6))
                sns.boxplot(
                    data=all_df,
                    x="Feature",
                    y="Importance",
                    order=mean_df.index
                )
                plt.xticks(rotation=90)
                if clases_3 and no_breath_data:
                    title_prefix = "3-Class, No Breath -"
                elif clases_3:
                    title_prefix = "3-Class -"
                elif no_breath_data:
                    title_prefix = "No Breath -"
                else:
                    title_prefix = ""

                plt.title(f"{title_prefix} Feature Importance Distribution (All Models & Repeats) ({signal})")
                plt.tight_layout()
                plt.savefig(os.path.join(comb_dir_plot, "Combined_Feature_Importance_BoxPlot.png"))
                plt.close()

            # Combine all per-signal summaries
        if all_summary:
            combined_summary_df = pd.concat(all_summary, ignore_index=True)
            output_path = os.path.join(out_dir, "CV_Summary", "NestedCV_AllSignals_combined_Summary.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Flatten MultiIndex columns
            combined_summary_df.columns = [
                f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
                for col in combined_summary_df.columns
            ]
            combined_summary_df = combined_summary_df.sort_values(['F1_mean'], ascending=False)
            combined_summary_df.to_csv(output_path, index=False)
            print("✅ Saved combined summary of all signals to NestedCV_AllSignals_combined_Summary.csv")
            # 🎯 Locate the best model by highest F1_mean
            best_row = combined_summary_df.iloc[0]
            best_signal = best_row["Signal"]
            best_model_name = best_row["Model"]
            best_window = int(best_row["Window (s)"])
            best_overlap = float(best_row["Overlap (%)"])

            dataset_path = os.path.join(
                self.path,
                "Participants",
                "Dataset",
                "Dataset_By_Window",
                "Clean_Data",
                f"Dataset_{best_window}s_{int(best_overlap)}.csv"
            )

            df_best = pd.read_csv(dataset_path).dropna(subset=["Class"])
            if best_signal != "All":
                selected_columns = self.ex_col + [col for col in df_best.columns if col.startswith(best_signal + "_")]
                df_best = df_best[selected_columns]

            feature_cols = [c for c in df_best.columns if c not in self.ex_col]

            if clases_3:
                df_best["Class"] = df_best["Class"].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                df_best.loc[df_best["Level"] == "hard", "Class"] = 2
                df_best.loc[df_best['Level'] == 'medium', 'Class'] = 2
                y_true = df_best["Class"]
            else:
                y_true = df_best["Class"].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})

            X = df_best[feature_cols]

            # Clone and set parameters
            best_model = clone(base_models[best_model_name])
            best_model.set_params(**best_params[best_model_name])

            # Train
            best_model.fit(X, y_true)

            # Predict
            y_pred = best_model.predict(X)
            labels = sorted(y_true.unique())

            cm_display = ConfusionMatrixDisplay.from_predictions(
                y_true,
                y_pred,
                display_labels=labels,
                cmap="Blues",
                normalize=None
            )

            if clases_3 and no_breath_data:
                title_prefix = "3-Class, No Breath -"
            elif clases_3:
                title_prefix = "3-Class -"
            elif no_breath_data:
                title_prefix = "No Breath -"
            else:
                title_prefix = ""

            plt.title(f"{title_prefix} Confusion Matrix - {best_model_name} ({best_signal})")
            plt.tight_layout()

            cm_dir = fr'{out_dir}/CV_Summary'
            os.makedirs(cm_dir, exist_ok=True)

            plot_path = os.path.join(cm_dir, f"ConfusionMatrix_{best_signal}_{best_model_name}.png")
            plt.savefig(plot_path)
            plt.close()

            print(f"✅ Saved confusion matrix plot to {plot_path}")

    def Cor(self):
        df = pd.read_csv(fr'{self.path}\Participants\Dataset\Dataset_By_Window\Clean_Data\Dataset_60s_0.csv')

        df = df.dropna(subset=['Stress', 'RSP_C_RRV_MedianBB'])

        x = df['RSP_C_RRV_MedianBB']
        y = df['Stress']

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.7, label='Data points')

        slope, intercept = np.polyfit(x, y, 1)
        reg_line = slope * x + intercept

        # ציור קו הרגרסיה
        plt.plot(x, reg_line, color='red', label=f'Regression line\ny={slope:.2f}x+{intercept:.2f}')

        # כותרות
        plt.xlabel('RSP_C_RRV_MedianBB')
        plt.ylabel('Stress')
        plt.title('Stress vs RSP_C_RRV_MedianBB with Regression Line')
        plt.legend()
        plt.grid(True)

        plt.show()

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

    import pandas as pd
    import numpy as np
    import os
    import re
    from scipy.stats import kruskal
    import scikit_posthocs as sp

    def StatisticalTest(self):
        """
        Perform statistical tests (Kruskal-Wallis with post-hoc Dunn tests)
        on subjective and performance data.
        """
        # --- Load file paths ---
        subject_data_path = os.path.join(self.path, 'Participants', 'Dataset', 'Subjective', 'SubjectiveDataset.csv')
        performance_data_path = os.path.join(self.path, 'Participants', 'Dataset', 'Performance', 'performance.csv')

        # --- Load datasets ---
        try:
            df_subjective = pd.read_csv(subject_data_path)
            df_perf = pd.read_csv(performance_data_path)
        except FileNotFoundError as e:
            print(f"Error: Could not find required data files: {e}")
            return
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # --- Replace control group ---
        df_subjective['Group'] = df_subjective['Group'].replace('control', 'natural')
        df_perf['Group'] = df_perf['Group'].replace('control', 'natural')

        # --- Output directory ---
        out_dir = os.path.join(self.path, 'Participants', 'Dataset', 'Statistical Tests')
        os.makedirs(out_dir, exist_ok=True)

        posthoc_results = {}
        kruskal_records = []

        # === Subjective Measures by Task_phase2 ===
        subjective_variables = ['Stress', 'Stress_S', 'Stress_S_std',
                                'Fatigue', 'Fatigue_S', 'Fatigue_S_std']

        if 'Task_phase2' in df_subjective.columns:
            subjective_tasks = df_subjective['Task_phase2'].dropna().unique()

            print("=== Analyzing Subjective Measures ===")
            for task in subjective_tasks:
                task_data = df_subjective[df_subjective['Task_phase2'] == task]

                for var in subjective_variables:
                    if var not in task_data.columns:
                        print(f"Warning: Variable {var} not found in subjective data")
                        continue

                    groups = [task_data[task_data['Group'] == g][var].dropna()
                              for g in task_data['Group'].unique()]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) >= 2 and len(np.unique(np.concatenate(groups))) > 1:
                        try:
                            stat, p = kruskal(*groups)
                            kruskal_records.append((f"{task}", var, stat, p))
                            print(f"Kruskal-Wallis for {task} - {var}: p={p:.4f}")

                            if p < 0.05:
                                dunn = sp.posthoc_dunn(task_data, val_col=var,
                                                       group_col='Group', p_adjust='fdr_bh')
                                key = f"{task} {var}"
                                posthoc_results[key] = dunn

                                # Create safe filename
                                safe_task = re.sub(r'[<>:"/\\|?*]', '_', task)
                                safe_var = re.sub(r'[<>:"/\\|?*]', '_', var)
                                filename = f"Dunn_{safe_task}_Task_phase2_{safe_var}.csv"
                                dunn_path = os.path.join(out_dir, filename)
                                dunn.to_csv(dunn_path)
                                print(f"✅ Dunn test saved to: {dunn_path}")
                        except Exception as e:
                            print(f"Error in statistical test for {task} - {var}: {e}")
                            kruskal_records.append((f"{task}", var, np.nan, np.nan))
                    else:
                        kruskal_records.append((f"{task}", var, np.nan, np.nan))

        # === Performance Analysis by Task and Level ===
        print("\n=== Analyzing Performance Data ===")

        # Get unique tasks and levels
        if 'Task' in df_perf.columns and 'Level' in df_perf.columns:
            tasks = df_perf['Task_level'].unique()

            for task in tasks:
                task_data = df_perf[df_perf['Task_level'] == task]

                # === Accuracy Analysis ===
                if 'correct' in task_data.columns:
                    groups = [task_data[task_data['Group'] == g]['correct'].dropna()
                              for g in task_data['Group'].unique()]
                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) >= 2 and len(np.unique(np.concatenate(groups))) > 1:
                        try:
                            stat, p = kruskal(*groups)
                            kruskal_records.append((f"{task}", 'Accuracy', stat, p))
                            print(f"Kruskal-Wallis for {task} - Accuracy: p={p:.4f}")

                            if p < 0.05:
                                dunn = sp.posthoc_dunn(task_data, val_col='correct',
                                                       group_col='Group', p_adjust='fdr_bh')
                                key = f"{task}_Accuracy"
                                posthoc_results[key] = dunn

                                # Create safe filename
                                safe_task = re.sub(r'[<>:"/\\|?*]', '_', task)
                                filename = f"Dunn_{safe_task}_Accuracy.csv"
                                dunn_path = os.path.join(out_dir, filename)
                                dunn.to_csv(dunn_path)
                                print(f"✅ Dunn test saved to: {dunn_path}")
                        except Exception as e:
                            print(f"Error in accuracy analysis for {task}: {e}")
                            kruskal_records.append((f"{task}", 'Accuracy', np.nan, np.nan))
                    else:
                        kruskal_records.append((f"{task}", 'Accuracy', np.nan, np.nan))

                # === Reaction Time Analysis ===
                    if 'RT' in task_data.columns:
                        # Filter out invalid RT values (negative or extremely high)
                        rt_data = task_data[task_data['RT'] > 0]  # Remove negative RTs
                        rt_data = rt_data[rt_data['RT'] < 10000]  # Remove RTs > 10 seconds

                        groups = [rt_data[rt_data['Group'] == g]['RT'].dropna()
                                  for g in rt_data['Group'].unique()]
                        groups = [g for g in groups if len(g) > 0]

                        if len(groups) >= 2 and len(np.unique(np.concatenate(groups))) > 1:
                            try:
                                stat, p = kruskal(*groups)
                                kruskal_records.append((f"{task} ", 'RT', stat, p))
                                print(f"Kruskal-Wallis for {task}  - RT: p={p:.4f}")

                                if p < 0.05:
                                    dunn = sp.posthoc_dunn(rt_data, val_col='RT',
                                                           group_col='Group', p_adjust='fdr_bh')
                                    key = f"{task}_RT"
                                    posthoc_results[key] = dunn

                                    # Create safe filename
                                    safe_task = re.sub(r'[<>:"/\\|?*]', '_', task)
                                    filename = f"Dunn_{safe_task}_RT.csv"
                                    dunn_path = os.path.join(out_dir, filename)
                                    dunn.to_csv(dunn_path)
                                    print(f"✅ Dunn test saved to: {dunn_path}")
                            except Exception as e:
                                print(f"Error in RT analysis for {task} : {e}")
                                kruskal_records.append((f"{task} ", 'RT', np.nan, np.nan))
                        else:
                            kruskal_records.append((f"{task} ", 'RT', np.nan, np.nan))

        # === Save Kruskal-Wallis Summary ===
        if kruskal_records:
            df_kruskal = pd.DataFrame(kruskal_records,
                                      columns=['Task', 'Measure', 'Statistic', 'P_value'])
            df_kruskal = df_kruskal.sort_values(by='P_value', ascending=True, na_position='last')

            kruskal_path = os.path.join(out_dir, "Kruskal_Wallis_Results.csv")
            df_kruskal.to_csv(kruskal_path, index=False)
            print(f"\n✅ Kruskal-Wallis results saved to: {kruskal_path}")

            # Print summary statistics
            significant_results = df_kruskal[df_kruskal['P_value'] < 0.05]
            print(f"\n📊 Summary:")
            print(f"Total tests performed: {len(df_kruskal)}")
            print(f"Significant results (p < 0.05): {len(significant_results)}")
            print(f"Post-hoc tests performed: {len(posthoc_results)}")

            if len(significant_results) > 0:
                print(f"\nMost significant results:")
                print(significant_results.head().to_string(index=False))
        else:
            print("No statistical tests were performed successfully.")

        return {
            'kruskal_results': df_kruskal if kruskal_records else None,
            'posthoc_results': posthoc_results,
            'output_directory': out_dir
        }
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

        sns.set_context('notebook', font_scale=1.6)
        # --- Load Data ---
        ks_df = pd.read_csv(fr'{self.path}\Participants\Dataset\Statistical Tests\Kruskal_Wallis_Results.csv')
        SubjectDat = pd.read_csv(fr'{self.path}\Participants\Dataset\Subjective\SubjectiveDataset.csv')
        SubjectDat['Group'] = SubjectDat['Group'].replace('control', 'natural')
        group_order = ['breath', 'music', 'natural']

        # --- Color Palette ---
        group_palette = {
            'breath': '#FF9999',
            'music': '#99CCFF',
            'natural': '#99FF99'
        }

        # --- Styling helper ---
        def style_axis(ax, title=None, xlabel=None, ylabel=None):
            if title: ax.set_title(title, fontsize=20)
            if xlabel: ax.set_xlabel(xlabel, fontsize=18)
            if ylabel: ax.set_ylabel(ylabel, fontsize=18)
            ax.tick_params(axis='both', labelsize=16)

        # --- Asterisk helper ---
        def add_asterisks_to_xticklabels(ax, measure_name):
            ticks = ax.get_xticks()
            labels = [label.get_text() for label in ax.get_xticklabels()]
            new_labels = []
            for text in labels:
                match = ks_df[
                    (ks_df['Task'] == text) &
                    (ks_df['Measure'] == measure_name) &
                    (ks_df['P_value'] < 0.05)
                    ]
                if not match.empty:
                    text = '*  ' + text
                new_labels.append(text)
            ax.set_xticks(ticks)
            ax.set_xticklabels(new_labels, rotation=45, fontsize=16)

        # --- Subjective Measures ---
        subjective_measures = [
            ("Stress", "Stress Rating", "Stress Rating"),
            ("Stress_S", "Stress Normalized by Start", "Stress Normalized by Start"),
            ("Stress_S_std", "Stress Normalized by Start and SD", "Stress Normalized by Start and SD"),
            ("Fatigue", "Fatigue Rating", "Fatigue Rating"),
            ("Fatigue_S", "Fatigue Normalized by Start", "Fatigue Normalized by Start"),
            ("Fatigue_S_std", "Fatigue Normalized by Start and SD", "Fatigue Normalized by Start and SD")
        ]

        phase2_order = [
            'Break1', 'Break2', 'Break3', 'Break4',
            'Stroop | easy', 'Stroop | hard',
            'PASAT | easy', 'PASAT | medium', 'PASAT | hard',
            'TwoColAdd | easy', 'TwoColAdd | hard'
        ]
        SubjectDat['Task_phase2'] = pd.Categorical(SubjectDat['Task_phase2'], categories=phase2_order, ordered=True)

        for col, ylabel, title in subjective_measures:
            subset = SubjectDat.copy()
            if col in ['Stress_S', 'Stress_S_std', 'Fatigue_S', 'Fatigue_S_std']:
                subset = subset[(subset['Task_phase2'] != 'start') & (subset['Task_phase1'] != 'Start')]

            fig, axes = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

            sns.boxplot(data=subset, x="Task_phase1", y=col, hue="Group",
                        palette=group_palette, hue_order=group_order, ax=axes[0])
            style_axis(axes[0], f"{title} by Task", "Task", ylabel)
            axes[0].grid(True)
            add_asterisks_to_xticklabels(axes[0], col)

            sns.boxplot(data=subset, x="Task_phase2", y=col, hue="Group",
                        palette=group_palette, hue_order=group_order, ax=axes[1])
            style_axis(axes[1], f"{title} by Task Level", "Task Level", "")
            axes[1].grid(True)
            add_asterisks_to_xticklabels(axes[1], col)

            handles, labels = axes[1].get_legend_handles_labels()
            axes[0].legend().remove()
            axes[1].legend(handles, labels, title='Group', loc='upper right', fontsize=16, title_fontsize=17)

            plt.suptitle(f"{title} - Group Comparison", fontsize=22)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = fr'{self.path}\Participants\Dataset\Subjective\{col}_plot.png'
            plt.savefig(out_path, dpi=300)
            plt.close()

        # --- Performance Plots ---
        PerformanceData = pd.read_csv(fr'{self.path}\Participants\Dataset\Performance\performance.csv')
        PerformanceData['Group'] = PerformanceData['Group'].replace('control', 'natural')
        PerformanceData['Task'] = PerformanceData['Task'].astype(str).str.strip()
        PerformanceData['Task_Level'] = PerformanceData['Task'] + ' | ' + PerformanceData['Level']
        PerformanceData = PerformanceData.dropna(subset=['correct']).copy()
        PerformanceData['correct'] = PerformanceData['correct'].astype(int)

        tasklevel_order = [
            'Stroop | easy', 'Stroop | hard',
            'PASAT | easy', 'PASAT | medium', 'PASAT | hard',
            'TwoColAdd | easy', 'TwoColAdd | hard'
        ]
        PerformanceData['Task_Level'] = pd.Categorical(PerformanceData['Task_Level'], categories=tasklevel_order,
                                                       ordered=True)

        # --- RT and Accuracy ---
        fig, axes = plt.subplots(1, 2, figsize=(22, 8))

        sns.boxplot(data=PerformanceData, x='Task_Level', y='RT', hue='Group',
                    palette=group_palette, hue_order=group_order, ax=axes[0])
        style_axis(axes[0], "Response Time by Task and Level", "Task | Level", "Response Time (RT)")
        axes[0].grid(True)

        sns.barplot(data=PerformanceData, x='Task_Level', y='correct', hue='Group',
                    estimator='mean', errorbar=('ci', 95),
                    palette=group_palette, hue_order=group_order, ax=axes[1])
        style_axis(axes[1], "Mean Accuracy by Task and Level (95% CI)", "Task | Level", "Accuracy (Proportion Correct)")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(fr'{self.path}\Participants\Dataset\Performance\Performance_TaskLevel.png', dpi=300)
        plt.close()

        # --- Accuracy by Task & Level ---
        fig, axes = plt.subplots(1, 2, figsize=(22, 8))

        sns.barplot(data=PerformanceData, x='Task', y='correct', hue='Group',
                    estimator='mean', errorbar=('ci', 95),
                    palette=group_palette, hue_order=group_order, ax=axes[0])
        style_axis(axes[0], "Mean Accuracy by Task (95% CI)", "Task", "Accuracy (Proportion Correct)")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True)
        add_asterisks_to_xticklabels(axes[0], 'Accuracy')

        sns.barplot(data=PerformanceData, x='Task_Level', y='correct', hue='Group',
                    estimator='mean', errorbar=('ci', 95),
                    palette=group_palette, hue_order=group_order, ax=axes[1])
        style_axis(axes[1], "Mean Accuracy by Task Level (95% CI)", "Task | Level", "Accuracy (Proportion Correct)")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True)
        add_asterisks_to_xticklabels(axes[1], 'Accuracy')

        plt.tight_layout()
        plt.savefig(fr'{self.path}\Participants\Dataset\Performance\Performance_Accuracy_Task_and_Level.png', dpi=300)
        plt.close()

        # Define a consistent color palette for groups
        group_palette = {
            # English labels
            'breath': '#1f77b4',  # blue
            'music': '#ff7f0e',  # orange
            'natural': '#2ca02c',  # green
        }

        # --- Group Mean Table + Correlation Plots ---
        perf_clean = PerformanceData[['Task_level', 'Group', 'correct']].copy()
        subj_clean = SubjectDat[['Task_phase2', 'Stress_S', 'Fatigue_S', 'Group']].copy()

        perf_mean = (
            perf_clean.groupby(['Group', 'Task_level'])['correct']
            .mean().reset_index(name='Mean_Accuracy')
        )
        subj_mean = (
            subj_clean.groupby(['Group', 'Task_phase2'], observed=True)[['Stress_S', 'Fatigue_S']]
            .mean().reset_index()
        ).rename(columns={'Task_phase2': 'Task_level'})

        group_means = pd.merge(perf_mean, subj_mean, on=['Group', 'Task_level'], how='inner')

        out_dir = fr'{self.path}\Participants\Dataset\Performance_Subjective'
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(fr'{out_dir}\regression', exist_ok=True)
        group_means.to_csv(fr'{out_dir}\GroupMeans_By_TaskLevel.csv', index=False)

        group_palette = {
            'breath': '#1f77b4',
            'music': '#ff7f0e',
            'natural': '#2ca02c',
        }

        def _infer_baseline_group(model, all_groups):
            terms = [p for p in model.params.index if p.startswith('C(Group)[T.')]
            groups_in_params = [t.split('C(Group)[T.')[1].split(']')[0] for t in terms]
            for g in all_groups:
                if g not in groups_in_params:
                    return g
            return sorted(all_groups, key=lambda z: str(z))[0]

        def _group_slope(model, x, group, baseline):
            base = model.params.get(x, 0.0)
            if group == baseline:
                return base
            key1 = f'{x}:C(Group)[T.{group}]'
            key2 = f'C(Group)[T.{group}]:{x}'
            inter = model.params.get(key1, model.params.get(key2, 0.0))
            return base + inter

        def _legend_groups_plus_pooled(ax, pooled_handle, legend_loc="lower right"):
            allowed_order = ["breath", "music", "natural"]
            handles, labels = ax.get_legend_handles_labels()
            label_to_handle = {}
            for h, l in zip(handles, labels):
                if l in allowed_order:
                    label_to_handle[l] = h
            legend_labels = [l for l in allowed_order if l in label_to_handle]
            legend_handles = [label_to_handle[l] for l in legend_labels]
            if pooled_handle is not None:
                legend_handles.append(pooled_handle)
                legend_labels.append('pooled')
            ax.legend(
                legend_handles, legend_labels, title='Group',
                loc=legend_loc, frameon=True, facecolor='white', framealpha=0.7
            )

        def scatter_reg_and_save(x, y, xlabel, ylabel, title, filename_base, legend_loc="lower right"):
            formula = f"{y} ~ {x} * C(Group)"
            model = smf.ols(formula, data=group_means).fit()

            with open(fr'{out_dir}\{filename_base}_model_summary.txt', 'w', encoding='utf-8') as f:
                f.write(model.summary().as_text())
            coef_df = model.params.rename('coef').to_frame()
            coef_df['std_err'] = model.bse
            coef_df['t'] = model.tvalues
            coef_df['p'] = model.pvalues
            coef_df.to_csv(fr'{out_dir}\regression\{filename_base}_coefficients.csv')

            plt.figure(figsize=(10, 6))
            ax = sns.scatterplot(
                data=group_means, x=x, y=y,
                hue='Group', style='Group', s=120,
                palette=group_palette, legend='full'
            )

            x_min, x_max = group_means[x].min(), group_means[x].max()
            x_grid = np.linspace(x_min, x_max, 200)
            groups = sorted(group_means['Group'].dropna().unique(), key=lambda z: str(z))
            baseline = _infer_baseline_group(model, groups)

            label_offsets = [(12, 10), (12, -10), (14, 16), (14, -16), (18, 0)]
            idx_positions = np.linspace(0.58, 0.9, max(len(groups), 3))[:len(groups)]

            for i, grp in enumerate(groups):
                pred_df = pd.DataFrame({x: x_grid, 'Group': grp})
                y_hat = model.predict(pred_df)
                color = group_palette.get(grp, None)
                ax.plot(x_grid, y_hat, linewidth=2.5, color=color, label=None)
                slope = _group_slope(model, x, grp, baseline)
                idx = int(idx_positions[i] * (len(x_grid) - 1))
                dx_pts, dy_pts = label_offsets[i % len(label_offsets)]
                ax.annotate(
                    f'slope={slope:.3f}',
                    xy=(x_grid[idx], y_hat[idx]),
                    xytext=(dx_pts, dy_pts),
                    textcoords='offset points',
                    ha='left', va='center',
                    fontsize=9, color=color,
                    bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
                    path_effects=[patheffects.withStroke(linewidth=1.2, foreground='white', alpha=0.9)]
                )

            pooled = smf.ols(f"{y} ~ {x}", data=group_means).fit()
            pooled_y = pooled.predict(pd.DataFrame({x: x_grid}))
            pooled_handle = ax.plot(x_grid, pooled_y, linestyle='--', linewidth=2.2, color='black', label='pooled')[0]
            idx_pooled = int(0.66 * (len(x_grid) - 1))
            ax.annotate(
                f'slope={pooled.params.get(x, 0.0):.3f}',
                xy=(x_grid[idx_pooled], pooled_y[idx_pooled]),
                xytext=(14, -12),
                textcoords='offset points',
                ha='left', va='center',
                fontsize=9, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5),
                path_effects=[patheffects.withStroke(linewidth=1.2, foreground='white', alpha=0.9)]
            )

            _legend_groups_plus_pooled(ax, pooled_handle, legend_loc)

            plt.title(f"{title}\nModel: {formula} | R^2 = {model.rsquared:.3f}", fontsize=16)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fr'{out_dir}\{filename_base}.png', dpi=300)
            plt.close()

        # === Run plots with custom legend locations ===
        scatter_reg_and_save(
            x='Stress_S', y='Mean_Accuracy',
            xlabel='Mean Stress Change (Stress_S)', ylabel='Mean Accuracy',
            title='Mean Accuracy vs. Mean Stress Change',
            filename_base='Accuracy_vs_Stress_Scatter',
            legend_loc='upper right'
        )

        scatter_reg_and_save(
            x='Fatigue_S', y='Mean_Accuracy',
            xlabel='Mean Fatigue Change (Fatigue_S)', ylabel='Mean Accuracy',
            title='Mean Accuracy vs. Mean Fatigue Change',
            filename_base='Accuracy_vs_Fatigue_Scatter',
            legend_loc='upper right'
        )

        scatter_reg_and_save(
            x='Fatigue_S', y='Stress_S',
            xlabel='Mean Fatigue Change (Fatigue_S)', ylabel='Mean Stress Change (Stress_S)',
            title='Stress vs. Fatigue by Task',
            filename_base='Stress_vs_Fatigue_Scatter',
            legend_loc='lower right'
        )


