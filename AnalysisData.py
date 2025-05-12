import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,confusion_matrix,
                             mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor

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

    # ──────────────────── main routine ───────────────────────────────
    def ML_models_particapent(self, ID: int | None = None, rangeID: bool = False):

        # ── participants list ────────────────────────────────────────
        pp_csv = f"{self.path}\\Participants\\participation management.csv"
        p_df = (pd.read_csv(pp_csv)
                  .dropna(axis=1, how="all")
                  .dropna(subset=["participant", "Date", "departmant"], how="all"))
        p_df["code"] = pd.to_numeric(p_df["code"], errors="coerce").astype("Int64")

        if ID is not None:
            p_df = p_df[p_df["code"] >= ID] if rangeID else p_df[p_df["code"] == ID]

        win_sizes, overlaps = [5, 10, 30, 60], [0.0, 0.5]

        out_root = Path(r"C:\Users\e3bom\OneDrive - post.bgu.ac.il\מחקר ביורפואי"
                        r"\Participants\Dataset\ML\Per_Particapent")
        out_root.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict] = []

        # ── iterate participants ─────────────────────────────────────
        for _, prow in p_df.iterrows():
            pid, group = int(prow["code"]), prow["Group"]
            out_xlsx = out_root / f"P{pid}_ML_Results.xlsx"
            plot_dir = out_root / f"P{pid}_Importance_Plots"
            plot_dir.mkdir(exist_ok=True)

            cls_metrics, reg_metrics, import_rows = [], [], []

            # ── iterate window × overlap ────────────────────────────
            for win, ov in [(w, o) for w in win_sizes for o in overlaps]:
                fpath = (f"{self.path}\\Participants\\{group}_group\\P_{pid}\\Features"
                         f"\\HRV\\HRV_Time_{win}_{ov}.csv")
                print(f"📁 Processing {fpath}")
                if not os.path.exists(fpath):
                    print("   ❌ file not found"); continue

                df = pd.read_csv(fpath).dropna().reset_index(drop=True)

                df["y_cls"] = df["Class"].map({"test": 1, "music": 0,
                                               "breath": 0, "natural": 0})
                feat_cols = [c for c in df.columns
                             if c not in ["ID", "Group", "Time", "Class",
                                          "Stress", "Fatigue", "y_cls"]]

                tscv = TimeSeriesSplit(n_splits=5)

                # ── classifiers ─────────────────────────────────────
                cls_models = {
                    "LogReg":   LogisticRegression(max_iter=1000, solver="liblinear",
                                                   class_weight="balanced"),
                    "LDA":      LinearDiscriminantAnalysis(),
                    "DT":       DecisionTreeClassifier(random_state=42),
                    "RF":       RandomForestClassifier(n_estimators=300, random_state=42),
                    "XGB":      XGBClassifier(n_estimators=400, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.8,
                                              objective="binary:logistic",
                                              eval_metric="logloss",
                                              use_label_encoder=False,
                                              random_state=42)
                }

                # ── regressors ──────────────────────────────────────
                reg_models = {
                    "LinReg":   LinearRegression(),
                    "DT":       DecisionTreeRegressor(random_state=42),
                    "RF":       RandomForestRegressor(n_estimators=200, random_state=42)
                }

                # ─────────  CLASSIFICATION LOOP  ───────────────────
                for mname, est in cls_models.items():
                    pipe = Pipeline([("sc", StandardScaler()), ("m", est)])

                    fold_probs, fold_true = [], []

                    for tr_idx, val_idx in tscv.split(df):
                        X_tr, y_tr = df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["y_cls"]
                        X_val, y_val = df.iloc[val_idx][feat_cols], df.iloc[val_idx]["y_cls"]

                        pipe.fit(X_tr, y_tr)
                        y_prob = pipe.predict_proba(X_val)[:, 1]   # prob class 1

                        fold_probs.append(y_prob)
                        fold_true.append(y_val.values)

                    y_val_all = np.concatenate(fold_true)
                    y_prob_all = np.concatenate(fold_probs)

                    cutoff, sc = self._best_cutoff(y_val_all, y_prob_all)

                    cls_metrics.append({
                        "Participant": pid, "Group": group,
                        "Window": win, "Overlap": ov, "Model": mname,
                        "Cutoff":     round(cutoff, 3),
                        **sc
                    })

                    # importance from full-data fit
                    pipe.fit(df[feat_cols], df["y_cls"])
                    imp = self._feature_importance(pipe["m"], feat_cols)
                    for feat, val in imp.items():
                        import_rows.append({
                            "Participant": pid, "Group": group,
                            "Window": win, "Overlap": ov,
                            "Model": mname, "Feature": feat,
                            "Importance": val
                        })
                    imp.sort_values().plot(kind="barh", figsize=(6, 3))
                    plt.title(f"P{pid} – {mname}  ({win}s / ov={ov})")
                    plt.tight_layout()
                    plt.savefig(plot_dir / f"P{pid}_{mname}_{win}s_{ov}_CLS.png",
                                dpi=300); plt.close()

                # ─────────  REGRESSION LOOP  ───────────────────────
                for mname, est in reg_models.items():
                    pipe = Pipeline([("sc", StandardScaler()), ("m", est)])

                    maes, rmses, r2s = [], [], []

                    for tr_idx, val_idx in tscv.split(df):
                        X_tr, y_tr = df.iloc[tr_idx][feat_cols], df.iloc[tr_idx]["Stress"]
                        X_val, y_val = df.iloc[val_idx][feat_cols], df.iloc[val_idx]["Stress"]

                        pipe.fit(X_tr, y_tr)
                        y_hat = pipe.predict(X_val)

                        maes.append(mean_absolute_error(y_val, y_hat))
                        rmses.append(np.sqrt(mean_squared_error(y_val, y_hat)))
                        r2s.append(r2_score(y_val, y_hat))

                    reg_metrics.append({
                        "Participant": pid, "Group": group,
                        "Window": win, "Overlap": ov, "Model": mname,
                        "MAE":  np.mean(maes),
                        "RMSE": np.mean(rmses),
                        "R2":   np.mean(r2s)
                    })

                    pipe.fit(df[feat_cols], df["Stress"])
                    imp = self._feature_importance(pipe["m"], feat_cols)
                    for feat, val in imp.items():
                        import_rows.append({
                            "Participant": pid, "Group": group,
                            "Window": win, "Overlap": ov,
                            "Model": mname, "Feature": feat,
                            "Importance": val
                        })
                    imp.sort_values().plot(kind="barh", figsize=(6, 3))
                    plt.title(f"P{pid} – {mname}  ({win}s / ov={ov})")
                    plt.tight_layout()
                    plt.savefig(plot_dir / f"P{pid}_{mname}_{win}s_{ov}_REG.png",
                                dpi=300); plt.close()
            # ── end window/overlap loop ─────────────────────────────

            # build sheets
            cls_df = (pd.DataFrame(cls_metrics).sort_values("F1", ascending=False))

            reg_df = (pd.DataFrame(reg_metrics)
                        .sort_values("RMSE", ascending=True))
            imp_df = (pd.DataFrame(import_rows)
                        .sort_values("Importance", ascending=False))

            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
                cls_df.to_excel(xlw, sheet_name="Classification", index=False)
                reg_df.to_excel(xlw, sheet_name="Regression",    index=False)
                imp_df.to_excel(xlw, sheet_name="Importance",    index=False)

            print(f"✅  Excel + plots finished for P{pid}")

            if not cls_df.empty:
                summary_rows.append({**cls_df.iloc[0].to_dict(), "Sheet": "Classification"})
            if not reg_df.empty:
                summary_rows.append({**reg_df.iloc[0].to_dict(), "Sheet": "Regression"})
            if not imp_df.empty:
                summary_rows.append({**imp_df.iloc[0].to_dict(), "Sheet": "Importance"})

        # ── master summary ──────────────────────────────────────────
        summary_df = pd.DataFrame(summary_rows)
        main_cols = ["Sheet", "Participant", "Group", "Window", "Overlap", "Model"]
        summary_df = summary_df[main_cols + [c for c in summary_df.columns
                                             if c not in main_cols]]
        master_xlsx = out_root / "ML_Best_Summary.xlsx"
        summary_df.to_excel(master_xlsx, index=False)
        print(f"🏆  Overall best summary saved to {master_xlsx}")

    def ML_models_all(self, n_repeats=9, plot=False):
        # Define window sizes and overlap percentages
        window_sizes = [5, 10, 30, 60]
        overlaps = [0.0, 0.5]

        # Define base models without hyperparameter tuning
        base_models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        # Define parameter grids for hyperparameter tuning
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
            }
        }

        # Load participant IDs from management file
        participants_csv = os.path.join(self.path, 'Participants', 'participation management.csv')
        participants = pd.read_csv(participants_csv)
        all_ids = participants['code'].dropna().astype(int).unique()

        results = []

        for repeat in range(n_repeats):
            # Iteration 1: Split 80% train / 20% test by participant IDs
            train_ids, test_ids = train_test_split(
                all_ids, test_size=0.2, random_state=42 + repeat
            )

            # Prepare best window info placeholders for each model
            best_ws = {name: {'window': None, 'overlap': None, 'f1': -np.inf} for name in base_models}

            # Iteration 2: Select optimal window size and overlap via 5-fold GroupKFold
            for ws in window_sizes:
                for ov in overlaps:
                    file_path = fr'{self.path}\Participants\Dataset\Dataset_window{ws}s_{int(ov * 100)}.csv'
                    if not os.path.exists(file_path):
                        continue

                    df = pd.read_csv(file_path).dropna().reset_index(drop=True)
                    df_train = df[df['ID'].isin(train_ids)]
                    feature_cols = [c for c in df.columns if c not in ['Time', 'ID', 'Group', 'Class']]
                    y = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                    groups = df_train['ID']

                    gkf = GroupKFold(n_splits=5)
                    for name, model in base_models.items():
                        f1_scores = []
                        for tr_idx, val_idx in gkf.split(df_train, y, groups):
                            X_tr = df_train.iloc[tr_idx][feature_cols]
                            y_tr = y.iloc[tr_idx]
                            X_val = df_train.iloc[val_idx][feature_cols]
                            y_val = y.iloc[val_idx]

                            model.fit(X_tr, y_tr)
                            y_pred = model.predict(X_val)
                            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

                        mean_f1 = np.mean(f1_scores)
                        if mean_f1 > best_ws[name]['f1']:
                            best_ws[name].update({'window': ws, 'overlap': ov, 'f1': mean_f1})

            # Iteration 3: Tune hyperparameters for selected window and overlap
            tuned_models = {}
            for name, base_model in base_models.items():
                ws = best_ws[name]['window']
                ov = best_ws[name]['overlap']
                file_path = fr'{self.path}\Participants\Dataset\Dataset_window{ws}s_{int(ov * 100)}.csv'
                df = pd.read_csv(file_path).dropna().reset_index(drop=True)
                df_train = df[df['ID'].isin(train_ids)]
                feature_cols = [c for c in df.columns if c not in ['Time', 'ID', 'Group', 'Class']]
                X_tr = df_train[feature_cols]
                y_tr = df_train['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})
                groups = df_train['ID']

                gkf = GroupKFold(n_splits=5)
                grid = GridSearchCV(
                    base_model,
                    param_grid=param_grids[name],
                    cv=gkf,
                    scoring='f1',
                    n_jobs=-1
                )
                grid.fit(X_tr, y_tr, groups=groups)
                tuned_models[name] = grid.best_estimator_

            # Iteration 4: Evaluate tuned models on the external test set
            for name, model in tuned_models.items():
                ws = best_ws[name]['window']
                ov = best_ws[name]['overlap']
                file_path = fr'{self.path}\Participants\Dataset\Dataset_window{ws}s_{int(ov * 100)}.csv'
                df = pd.read_csv(file_path).dropna().reset_index(drop=True)
                df_test = df[df['ID'].isin(test_ids)]
                feature_cols = [c for c in df.columns if c not in ['Time', 'ID', 'Group', 'Class']]
                X_te = df_test[feature_cols]
                y_te = df_test['Class'].map({'test': 1, 'music': 0, 'breath': 0, 'natural': 0})

                y_pred = model.predict(X_te)
                results.append({
                    'Repeat': repeat + 1,
                    'Model': name,
                    'Window (s)': ws,
                    'Overlap (%)': int(ov * 100),
                    'Accuracy': accuracy_score(y_te, y_pred),
                    'Precision': precision_score(y_te, y_pred, zero_division=0),
                    'Recall': recall_score(y_te, y_pred, zero_division=0),
                    'F1': f1_score(y_te, y_pred, zero_division=0)
                })

                # Optional: save feature importance plots
                if plot and hasattr(model, 'feature_importances_'):
                    out_dir = fr"{self.path}\Participants\Dataset\ML\{name}\Repeat{repeat + 1}"
                    os.makedirs(out_dir, exist_ok=True)
                    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    imp.plot.bar(ax=ax)
                    ax.set_title(f"{name} Importances (Repeat {repeat + 1}, {ws}s/{int(ov * 100)}%)")
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, f"{name}_Importance_R{repeat + 1}.png"))
                    plt.close(fig)

        # Save summary of results to CSV
        results_df = pd.DataFrame(results)
        out_path = fr'{self.path}\Participants\Dataset\ML\NestedCV_Results.csv'
        results_df.to_csv(out_path, index=False)
        print(f"Nested CV complete. Results saved to {out_path}")

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




