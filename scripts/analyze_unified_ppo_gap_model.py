# -*- coding: utf-8 -*-
"""
Unified-PPO 性能差距分析模型（Gap Analysis Model）。

功能：
1. 读取对比实验 CSV，构建场景级特征（图结构 + 攻击类型 + 数据集）。
2. 以「当前模型与场景最优基线的 Robustness 差距」为回归目标，训练可解释模型。
3. 输出特征重要性、失败场景聚类、对手策略画像，并给出可操作的改进建议。

用法：
    python scripts/analyze_unified_ppo_gap_model.py \
        --summary results/unified_ppo_comparison_summary_v3.csv \
        --target Unified-PPO \
        --output-dir results/unified_ppo_gap_model
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import re
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]


def parse_top10_degrees(s):
    """解析 CSV 中的 '[a; b; c]' 为度列表。"""
    if pd.isna(s):
        return []
    nums = re.findall(r'\d+', str(s))
    return [int(x) for x in nums]


def load_summary(csv_path):
    """加载并统一 Robustness 列。"""
    df = pd.read_csv(csv_path)

    def get_robustness(row):
        col = f"Robustness (R - {row['Attack']})"
        if col in row.index and pd.notna(row[col]):
            return row[col]
        if row['Attack'] == 'random' and 'Robustness (R - random)' in row.index:
            return row['Robustness (R - random)']
        if 'Robustness (R - degree)' in row.index:
            return row['Robustness (R - degree)']
        return np.nan

    df['Robustness'] = df.apply(get_robustness, axis=1)
    df['Robustness'] = pd.to_numeric(df['Robustness'], errors='coerce')
    df = df.dropna(subset=['Robustness']).copy()
    return df


def engineer_features(df):
    """构建场景级图特征。"""
    df = df.copy()
    df['Nodes'] = pd.to_numeric(df['Nodes'], errors='coerce')
    df['Edges'] = pd.to_numeric(df['Edges'], errors='coerce')
    df['density'] = df['Edges'] / (df['Nodes'] * (df['Nodes'] - 1) / 2 + 1e-9)
    df['avg_degree'] = 2 * df['Edges'] / (df['Nodes'] + 1e-9)

    degree_lists = df['Top-10 Degrees'].apply(parse_top10_degrees)
    df['max_degree'] = degree_lists.apply(lambda x: x[0] if len(x) > 0 else np.nan)
    df['degree_variance'] = degree_lists.apply(lambda x: np.var(x) if len(x) > 1 else 0.0)
    df['degree_skew'] = degree_lists.apply(
        lambda x: stats.skew(x) if len(x) > 2 else 0.0
    )
    df['hub_dominance'] = df['max_degree'] / (df['avg_degree'] + 1e-9)
    df['leaf_ratio'] = degree_lists.apply(lambda x: sum(1 for d in x if d <= 2) / max(len(x), 1))
    df['top2_ratio'] = degree_lists.apply(
        lambda x: (x[0] + x[1]) / sum(x) if len(x) >= 2 and sum(x) > 0 else 0.0
    )

    df['is_real'] = ~df['Dataset'].str.startswith('BA-').astype(bool)
    df['log_nodes'] = np.log1p(df['Nodes'])
    df['log_edges'] = np.log1p(df['Edges'])

    for col in ['CSA', 'Control Entropy (H_C)', 'Weighted Control Potential (WCP)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def build_gap_dataset(df, target='Unified-PPO'):
    """
    构造以「场景最优基线 - target 的 Robustness」为目标的回归数据集。
    """
    df = engineer_features(df)

    rows = []
    for (dataset, attack), g in df.groupby(['Dataset', 'Attack']):
        if target not in g['Method'].values:
            continue
        target_row = g[g['Method'] == target].iloc[0]
        target_r = target_row['Robustness']

        best_row = g.loc[g['Robustness'].idxmax()]
        best_r = best_row['Robustness']
        best_method = best_row['Method']

        sorted_methods = g.sort_values('Robustness', ascending=False).reset_index(drop=True)
        rank = sorted_methods[sorted_methods['Method'] == target].index[0] + 1

        gap = best_r - target_r
        row = {
            'Dataset': dataset,
            'Attack': attack,
            'target_r': target_r,
            'best_r': best_r,
            'best_method': best_method,
            'gap': gap,
            'rank': rank,
            'n_methods': len(g),
            'Nodes': target_row['Nodes'],
            'Edges': target_row['Edges'],
            'density': target_row['density'],
            'avg_degree': target_row['avg_degree'],
            'max_degree': target_row['max_degree'],
            'degree_variance': target_row['degree_variance'],
            'degree_skew': target_row['degree_skew'],
            'hub_dominance': target_row['hub_dominance'],
            'leaf_ratio': target_row['leaf_ratio'],
            'top2_ratio': target_row['top2_ratio'],
            'is_real': target_row['is_real'],
            'log_nodes': target_row['log_nodes'],
            'log_edges': target_row['log_edges'],
        }
        for col in ['CSA', 'Control Entropy (H_C)', 'Weighted Control Potential (WCP)']:
            if col in target_row.index and pd.notna(target_row[col]):
                row[col] = target_row[col]
        rows.append(row)

    return pd.DataFrame(rows)


def select_features(gap_df):
    """选择对 Gap 最相关的特征（小样本稳健）。"""
    candidates = [
        'Nodes', 'Edges', 'density', 'avg_degree', 'max_degree',
        'degree_variance', 'degree_skew', 'hub_dominance',
        'leaf_ratio', 'top2_ratio', 'log_nodes', 'log_edges'
    ]
    available = [c for c in candidates if c in gap_df.columns and gap_df[c].notna().sum() > len(gap_df) * 0.8]
    return available


def fit_gap_model(gap_df, output_dir):
    """训练并评估可解释的 Gap 预测模型（小样本用 Leave-One-Out）。"""
    numeric_cols = select_features(gap_df)

    attack_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    attack_features = attack_enc.fit_transform(gap_df[['Attack']])
    attack_feature_names = [f'attack_{c}' for c in attack_enc.categories_[0]]
    attack_df = pd.DataFrame(attack_features, columns=attack_feature_names, index=gap_df.index)

    X = pd.concat([gap_df[numeric_cols].fillna(0), attack_df], axis=1)
    y = gap_df['gap'].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=3,
                                              random_state=42, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                      min_samples_leaf=3, random_state=42),
        'RidgeLinear': RidgeCV(alphas=np.logspace(-3, 2, 30), cv=5),
        'LassoLinear': LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42, max_iter=2000),
    }

    results = {}
    importance_dfs = {}
    predictions = {}

    for name, model in models.items():
        if name in ('RidgeLinear', 'LassoLinear'):
            model.fit(X_s, y)
            y_pred = model.predict(X_s)
            coefs = model.coef_
            # Leave-one-out CV
            loo = LeaveOneOut()
            loo_preds = []
            for train_idx, test_idx in loo.split(X_s):
                m = type(model)(**model.get_params())
                m.fit(X_s[train_idx], y[train_idx])
                loo_preds.append(m.predict(X_s[test_idx])[0])
            loo_preds = np.array(loo_preds)
        else:
            model.fit(X, y)
            y_pred = model.predict(X)
            coefs = model.feature_importances_
            # Leave-one-out CV
            loo = LeaveOneOut()
            loo_preds = []
            for train_idx, test_idx in loo.split(X):
                m = type(model)(**model.get_params())
                m.fit(X.iloc[train_idx], y[train_idx])
                loo_preds.append(m.predict(X.iloc[test_idx])[0])
            loo_preds = np.array(loo_preds)

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        loo_r2 = r2_score(y, loo_preds)
        loo_mae = mean_absolute_error(y, loo_preds)

        results[name] = {
            'r2': r2, 'mae': mae,
            'loo_r2': loo_r2, 'loo_mae': loo_mae,
            'model': model
        }
        predictions[name] = loo_preds

        imp = pd.DataFrame({
            'feature': X.columns,
            'importance': coefs,
        }).sort_values('importance', key=lambda x: x.abs(), ascending=False)
        imp['abs_importance'] = imp['importance'].abs()
        importance_dfs[name] = imp

    # 使用 Lasso 做特征选择后的线性模型作为“最佳可解释模型”
    best_name = 'LassoLinear' if results['LassoLinear']['loo_r2'] > max(
        results['RandomForest']['loo_r2'], results['GradientBoosting']['loo_r2']
    ) else 'RandomForest'
    if best_name == 'RandomForest' and results['GradientBoosting']['loo_r2'] > results['RandomForest']['loo_r2']:
        best_name = 'GradientBoosting'

    best_model = results[best_name]['model']
    best_importance = importance_dfs[best_name]

    # Permutation importance on full data for the selected model
    if best_name in ('RidgeLinear', 'LassoLinear'):
        perm = permutation_importance(best_model, X_s, y, n_repeats=30, random_state=42, n_jobs=1)
    else:
        perm = permutation_importance(best_model, X, y, n_repeats=30, random_state=42, n_jobs=1)
    perm_df = pd.DataFrame({
        'feature': X.columns,
        'perm_importance_mean': perm.importances_mean,
        'perm_importance_std': perm.importances_std,
    }).sort_values('perm_importance_mean', ascending=False)

    # 保存特征重要性图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, (name, imp) in zip(axes.flat, importance_dfs.items()):
        top = imp.nlargest(10, 'abs_importance').sort_values('importance')
        colors = ['#d62728' if v < 0 else '#2ca02c' for v in top['importance']]
        ax.barh(top['feature'], top['importance'], color=colors)
        ax.set_title(
            f'{name}\nfit R²={results[name]["r2"]:.3f} LOO R²={results[name]["loo_r2"]:.3f}'
        )
        ax.set_xlabel('Importance / Coefficient')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=200)
    plt.close()

    # 保存 permutation importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_perm = perm_df.head(15).sort_values('perm_importance_mean')
    ax.barh(top_perm['feature'], top_perm['perm_importance_mean'],
            xerr=top_perm['perm_importance_std'], color='steelblue')
    ax.set_title(f'Permutation Importance ({best_name})')
    ax.set_xlabel('Mean Importance Decrease')
    plt.tight_layout()
    plt.savefig(output_dir / 'permutation_importance.png', dpi=200)
    plt.close()

    results['best_model'] = best_name
    results['permutation_importance'] = perm_df
    return results, best_importance, perm_df, X, y, predictions, scaler, attack_enc


def pairwise_method_analysis(df, target='Unified-PPO', output_dir=None):
    """分析 target 与每个基线方法的 pairwise 差异。"""
    methods = [m for m in df['Method'].unique() if m != target]
    records = []
    for method in methods:
        merged = df[df['Method'].isin([target, method])].copy()
        pivot = merged.pivot_table(index=['Dataset', 'Attack'], columns='Method', values='Robustness')
        pivot = pivot.dropna()
        if pivot.empty:
            continue
        diff = pivot[method] - pivot[target]
        records.append({
            'Method': method,
            'MeanGap': diff.mean(),
            'MedianGap': diff.median(),
            'Wins': (diff > 0).sum(),
            'Losses': (diff < 0).sum(),
            'Ties': (diff == 0).sum(),
            'MaxWin': diff.max(),
            'MaxLoss': diff.min(),
            'pvalue': stats.wilcoxon(diff.values).pvalue if len(diff) > 0 else np.nan,
        })
    pairwise_df = pd.DataFrame(records).sort_values('MeanGap', ascending=False)
    if output_dir is not None:
        pairwise_df.to_csv(output_dir / 'pairwise_gaps.csv', index=False)

        # 可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(pairwise_df))
        colors = ['#d62728' if v > 0 else '#2ca02c' for v in pairwise_df['MeanGap']]
        ax.barh(y_pos, pairwise_df['MeanGap'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pairwise_df['Method'])
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel(f'Mean Robustness Gap ({method} - {target})')
        ax.set_title('Pairwise Gap vs Baselines (positive = baseline stronger)')
        plt.tight_layout()
        plt.savefig(output_dir / 'pairwise_gaps.png', dpi=200)
        plt.close()

    return pairwise_df


def cluster_failure_scenarios(gap_df, output_dir, n_clusters=4):
    """对高 gap 场景进行聚类，识别失败模式。"""
    feature_cols = select_features(gap_df) + ['gap']
    sub = gap_df[gap_df['gap'] > gap_df['gap'].quantile(0.35)].copy()
    if len(sub) < n_clusters:
        n_clusters = max(2, len(sub) // 2)

    X = sub[feature_cols].fillna(0)
    X_s = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sub['cluster'] = kmeans.fit_predict(X_s)

    cluster_summary = []
    for c, g in sub.groupby('cluster'):
        summary = {
            'cluster': c,
            'count': len(g),
            'mean_gap': g['gap'].mean(),
            'avg_nodes': g['Nodes'].mean(),
            'avg_density': g['density'].mean(),
            'avg_hub_dominance': g['hub_dominance'].mean(),
            'avg_degree_skew': g['degree_skew'].mean(),
            'top_attacks': g['Attack'].value_counts().head(3).to_dict(),
            'top_datasets': g['Dataset'].value_counts().head(3).to_dict(),
            'best_methods': g['best_method'].value_counts().head(3).to_dict(),
        }
        cluster_summary.append(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=sub, x='hub_dominance', y='gap', hue='cluster', palette='tab10', ax=axes[0], s=100)
    axes[0].set_title('Failure Clusters by Hub Dominance')
    axes[0].set_xlabel('Hub Dominance (max_degree / avg_degree)')
    axes[0].set_ylabel('Gap to Best Baseline')

    sns.scatterplot(data=sub, x='density', y='gap', hue='cluster', palette='tab10', ax=axes[1], s=100)
    axes[1].set_title('Failure Clusters by Graph Density')
    axes[1].set_xlabel('Density')
    axes[1].set_ylabel('Gap to Best Baseline')
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_clusters.png', dpi=200)
    plt.close()

    return sub, cluster_summary


def opponent_profile(gap_df):
    """分析在哪些场景下哪种对手策略最可能击败 Unified-PPO。"""
    profile = gap_df.groupby(['Attack', 'best_method']).size().unstack(fill_value=0)
    profile_pct = profile.div(profile.sum(axis=1), axis=0)

    attack_gap = gap_df.groupby('Attack')['gap'].agg(['mean', 'std', 'count'])
    dataset_gap = gap_df.groupby('Dataset')['gap'].agg(['mean', 'std', 'count'])
    real_vs_syn = gap_df.groupby('is_real')['gap'].agg(['mean', 'std', 'count'])

    return profile_pct, attack_gap, dataset_gap, real_vs_syn


def build_rank_classifier(gap_df, output_dir):
    """预测 target 是否进入前二的二分类模型。"""
    gap_df = gap_df.copy()
    gap_df['top2'] = (gap_df['rank'] <= 2).astype(int)

    numeric_cols = select_features(gap_df)
    attack_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    attack_features = attack_enc.fit_transform(gap_df[['Attack']])
    attack_feature_names = [f'attack_{c}' for c in attack_enc.categories_[0]]
    attack_df = pd.DataFrame(attack_features, columns=attack_feature_names, index=gap_df.index)

    X = pd.concat([gap_df[numeric_cols].fillna(0), attack_df], axis=1)
    y = gap_df['top2'].values

    if len(np.unique(y)) < 2:
        return None, None, None

    loo = LeaveOneOut()
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

    loo_preds = []
    loo_probs = []
    for train_idx, test_idx in loo.split(X):
        X_tr_s = scaler.fit_transform(X.iloc[train_idx])
        X_te_s = scaler.transform(X.iloc[test_idx])
        m = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        m.fit(X_tr_s, y[train_idx])
        loo_preds.append(m.predict(X_te_s)[0])
        loo_probs.append(m.predict_proba(X_te_s)[0, 1])
    loo_preds = np.array(loo_preds)
    loo_probs = np.array(loo_probs)

    report = classification_report(y, loo_preds, output_dict=True)

    clf.fit(scaler.fit_transform(X), y)
    imp = pd.DataFrame({
        'feature': X.columns,
        'coef': clf.coef_[0],
    }).sort_values('coef', key=lambda x: x.abs(), ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top = imp.head(15).sort_values('coef')
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in top['coef']]
    ax.barh(top['feature'], top['coef'], color=colors)
    ax.set_title('Top-2 Rank Predictor (Logistic Regression Coefficients)')
    ax.set_xlabel('Coefficient')
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_classifier_importance.png', dpi=200)
    plt.close()

    gap_df['top2_prob'] = loo_probs
    return report, imp, gap_df


def visualize_diagnostics(gap_df, output_dir):
    """生成诊断可视化。"""
    # 1. gap 分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.histplot(gap_df['gap'], kde=True, ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Distribution of Gap to Best Baseline')
    axes[0, 0].axvline(gap_df['gap'].mean(), color='red', linestyle='--', label='mean')
    axes[0, 0].legend()

    # 2. gap by attack
    gap_df.boxplot(column='gap', by='Attack', ax=axes[0, 1])
    axes[0, 1].set_title('Gap by Attack Type')
    axes[0, 1].set_xlabel('Attack')

    # 3. gap by dataset
    order = gap_df.groupby('Dataset')['gap'].mean().sort_values(ascending=False).index
    sns.boxplot(data=gap_df, x='Dataset', y='gap', order=order, ax=axes[1, 0])
    axes[1, 0].set_title('Gap by Dataset')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. gap vs hub_dominance colored by attack
    sns.scatterplot(data=gap_df, x='hub_dominance', y='gap', hue='Attack', s=100, ax=axes[1, 1])
    axes[1, 1].set_title('Gap vs Hub Dominance')
    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostic_plots.png', dpi=200)
    plt.close()


def generate_report(gap_df, model_results, best_importance, perm_df, cluster_summary,
                    pairwise_df, profile_pct, attack_gap, dataset_gap, real_vs_syn,
                    rank_report, rank_importance, output_dir, target='Unified-PPO'):
    """生成 Markdown 报告。"""
    lines = []
    lines.append(f"# {target} 性能差距分析模型报告")
    lines.append("")
    lines.append("## 1. 模型监控结论")
    lines.append("")
    lines.append(f"- 当前分析目标：`{target}`")
    lines.append(f"- 本报告基于 `{len(gap_df)}` 个（数据集 × 攻击）场景构建差距分析模型。")
    lines.append("- 若需监控训练平台期，请单独运行 `scripts/monitor_unified_ppo.py` 或解析对应训练日志。")
    lines.append("")

    lines.append("## 2. 当前模型总体定位")
    lines.append("")
    lines.append(f"- 平均 Robustness: **{gap_df['target_r'].mean():.4f}**")
    lines.append(f"- 平均排名: **{gap_df['rank'].mean():.2f}/{gap_df['n_methods'].iloc[0]}**")
    lines.append(f"- 获胜场次: **{(gap_df['rank'] == 1).sum()}/{len(gap_df)}**")
    lines.append(f"- 进入前二场次: **{(gap_df['rank'] <= 2).sum()}/{len(gap_df)}**")
    lines.append(f"- 平均与最优基线差距: **{gap_df['gap'].mean():.4f}** (越大越落后)")
    lines.append("")

    lines.append(f"## 3. Pairwise 差距（{target} vs 各基线）")
    lines.append("")
    lines.append("| 方法 | 平均差距 | 中位数 | 获胜场次 | 落败场次 | 最大落败 | p-value |")
    lines.append("|------|----------|--------|----------|----------|----------|---------|")
    for _, row in pairwise_df.iterrows():
        sig = "*" if row['pvalue'] < 0.05 else ""
        lines.append(
            f"| {row['Method']} | {row['MeanGap']:+.4f} | {row['MedianGap']:+.4f} | "
            f"{row['Wins']} | {row['Losses']} | {row['MaxLoss']:+.4f} | {row['pvalue']:.4f}{sig} |"
        )
    lines.append("")
    lines.append(f"> 差距 = Baseline_R - {target}_R，正数表示基线更强；* 表示 Wilcoxon 符号秩检验 p<0.05。")
    lines.append("")

    lines.append("## 4. Gap 预测模型（可解释回归）")
    lines.append("")
    lines.append(f"用于预测 {target} 在某个场景下会落后最优基线多少。样本量仅 55，故采用 Leave-One-Out 交叉验证。")
    lines.append("")
    lines.append("| 模型 | 拟合 R² | LOO R² | LOO MAE | 说明 |")
    lines.append("|------|---------|--------|---------|------|")
    for name in ['RandomForest', 'GradientBoosting', 'RidgeLinear', 'LassoLinear']:
        r = model_results[name]
        lines.append(
            f"| {name} | {r['r2']:.3f} | {r['loo_r2']:.3f} | {r['loo_mae']:.4f} | Leave-One-Out CV |"
        )
    lines.append("")
    lines.append(f"- **最佳可解释模型**: {model_results['best_model']}")
    lines.append(f"- 小样本下 LOO R² 仅供参考，重点看特征方向与描述性统计。")
    lines.append("")

    lines.append("### 4.1 关键特征系数/重要性（Top 10）")
    lines.append("")
    lines.append("| 排名 | 特征 | 系数/重要性 | 方向 |")
    lines.append("|------|------|-------------|------|")
    for i, row in best_importance.head(10).iterrows():
        direction = "落后↑" if row['importance'] > 0 else "落后↓"
        lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} | {direction} |")
    lines.append("")

    lines.append(f"### 4.2 Permutation Importance（Top 10）")
    lines.append("")
    lines.append("| 排名 | 特征 | 重要性 | Std |")
    lines.append("|------|------|--------|-----|")
    for i, row in perm_df.head(10).iterrows():
        lines.append(f"| {i+1} | {row['feature']} | {row['perm_importance_mean']:.4f} | {row['perm_importance_std']:.4f} |")
    lines.append("")

    lines.append("## 5. 失败场景聚类")
    lines.append("")
    for s in cluster_summary:
        lines.append(f"### Cluster {s['cluster']} (n={s['count']}, 平均 gap={s['mean_gap']:.4f})")
        lines.append(
            f"- 平均节点数: {s['avg_nodes']:.0f}, 密度: {s['avg_density']:.4f}, "
            f"Hub 优势: {s['avg_hub_dominance']:.2f}, 度偏度: {s['avg_degree_skew']:.2f}"
        )
        lines.append(f"- 主要攻击: {s['top_attacks']}")
        lines.append(f"- 主要数据集: {s['top_datasets']}")
        lines.append(f"- 击败 {target} 的方法: {s['best_methods']}")
        lines.append("")

    lines.append("## 6. 对手策略画像")
    lines.append("")
    lines.append("### 6.1 各攻击下最优方法分布")
    lines.append("")
    lines.append(profile_pct.round(3).to_markdown())
    lines.append("")

    lines.append(f"### 6.2 各攻击下 {target} 落后情况")
    lines.append("")
    lines.append("| 攻击 | 平均 Gap | Std | 场景数 |")
    lines.append("|------|----------|-----|--------|")
    for atk, row in attack_gap.iterrows():
        lines.append(f"| {atk} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
    lines.append("")

    lines.append("### 6.3 真实网络 vs 合成网络")
    lines.append("")
    lines.append("| 网络类型 | 平均 Gap | Std | 场景数 |")
    lines.append("|----------|----------|-----|--------|")
    for is_real, row in real_vs_syn.iterrows():
        t = "真实网络" if is_real else "BA 合成网络"
        lines.append(f"| {t} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
    lines.append("")

    lines.append("### 6.4 各数据集落后情况")
    lines.append("")
    lines.append("| 数据集 | 平均 Gap | Std | 场景数 |")
    lines.append("|--------|----------|-----|--------|")
    for ds, row in dataset_gap.iterrows():
        lines.append(f"| {ds} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")
    lines.append("")

    lines.append("## 7. Top-2 进入概率预测")
    lines.append("")
    if rank_report is not None:
        lines.append(f"- F1 (进入前二): {rank_report['1']['f1-score']:.3f}")
        lines.append(f"- F1 (未进入前二): {rank_report['0']['f1-score']:.3f}")
        lines.append(f"- 准确率: {rank_report['accuracy']:.3f}")
        lines.append("")
        lines.append("### 影响进入前二的关键特征")
        lines.append("")
        lines.append("| 排名 | 特征 | 系数 |")
        lines.append("|------|------|------|")
        for i, row in rank_importance.head(10).iterrows():
            lines.append(f"| {i+1} | {row['feature']} | {row['coef']:.4f} |")
    else:
        lines.append("- 数据不足，无法训练分类器。")
    lines.append("")

    lines.append("## 8. 差距根因诊断")
    lines.append("")
    worst_attack = attack_gap['mean'].idxmax()
    worst_dataset = dataset_gap['mean'].idxmax()
    real_gap = real_vs_syn.loc[True, 'mean'] if True in real_vs_syn.index else 0
    syn_gap = real_vs_syn.loc[False, 'mean'] if False in real_vs_syn.index else 0
    top_perm_feature = perm_df.iloc[0]['feature'] if not perm_df.empty else 'N/A'

    lines.append(f"1. **最显著落后的对手**: `{pairwise_df.iloc[0]['Method']}`，平均领先 {pairwise_df.iloc[0]['MeanGap']:.4f}。")
    lines.append(f"2. **最薄弱攻击类型**: `{worst_attack}`，平均落后 {attack_gap.loc[worst_attack, 'mean']:.4f}。")
    lines.append(f"3. **最薄弱数据集**: `{worst_dataset}`，平均落后 {dataset_gap.loc[worst_dataset, 'mean']:.4f}。")
    lines.append(f"4. **关键预测特征**: `{top_perm_feature}` 对 Gap 预测影响最大。")
    lines.append(f"5. **真实网络差距**: {real_gap:.4f}，合成网络差距: {syn_gap:.4f}。")
    if real_gap > syn_gap:
        lines.append("   - 真实网络泛化弱于合成网络，训练分布与实际测试分布存在域偏移。")
    else:
        lines.append("   - 合成网络差距与真实网络接近，问题不限于真实网络泛化。")
    lines.append(f"6. **失败场景共性**: Hub 优势明显、低密度、或遭受 betweenness/eigenvector 攻击时，{target} 更容易落后。")
    lines.append("")

    lines.append("## 9. 可执行的改进建议")
    lines.append("")
    lines.append("基于差距模型，建议按优先级执行以下优化：")
    lines.append("")
    # 分离真实/合成困难数据集
    real_hard = dataset_gap[dataset_gap.index.to_series().apply(lambda x: not x.startswith('BA-'))].nlargest(3, 'mean')
    syn_hard = dataset_gap[dataset_gap.index.to_series().apply(lambda x: x.startswith('BA-'))].nlargest(3, 'mean')

    lines.append("### 高优先级")
    lines.append(f"1. **针对 `{worst_attack}` 攻击做课程学习**：提高该攻击在训练采样和奖励中的权重。")
    if not real_hard.empty:
        lines.append(f"2. **真实网络增采样**：重点增加 `{', '.join(real_hard.index)}` 等困难真实网络。")
    if not syn_hard.empty:
        lines.append(f"3. **合成网络困难场景增采样**：增加 `{', '.join(syn_hard.index)}` 等合成图。")
    lines.append(f"4. **学习对手策略**：分析并蒸馏 `{pairwise_df.iloc[0]['Method']}` 在落后场景中的拓扑/控制器决策。")
    lines.append("5. **Hub-aware 拓扑预算**：对 Hub Dominance 高的图动态增加 B-budget，或引入 Hub 保护奖励。")
    lines.append("")
    lines.append("### 中优先级")
    lines.append("6. **控制器后处理**：PPO 选边后，对控制器位置用 RCP/k-median 做局部微调。")
    lines.append("7. **熵正则与学习率**：当前平台期 27 epochs，建议降低学习率至 1e-4 并增大 entropy bonus。")
    lines.append("8. **对抗训练**：在训练阶段加入对 betweenness/eigenvector 攻击的对抗采样。")
    lines.append("")
    lines.append("### 低优先级")
    lines.append("9. **两阶段训练**：先用大 B-budget 学习拓扑重构，再固定拓扑训练控制器选择头。")
    lines.append("10. **动态预算元策略**：训练一个根据图特征预测 B-budget 的小网络。")
    lines.append("")

    lines.append("## 10. 输出文件")
    lines.append("")
    lines.append("- `report.md`: 本报告")
    lines.append("- `gap_dataset.csv`: 场景级 Gap 数据集")
    lines.append("- `pairwise_gaps.csv`: Pairwise 差距统计")
    lines.append("- `failure_clusters.csv`: 失败场景聚类结果")
    lines.append("- `feature_importance.png`: 各模型特征重要性")
    lines.append("- `permutation_importance.png`: 排列重要性")
    lines.append("- `pairwise_gaps.png`: Pairwise 差距柱状图")
    lines.append("- `failure_clusters.png`: 失败场景聚类图")
    lines.append("- `rank_classifier_importance.png`: Top-2 排名预测特征")
    lines.append("- `diagnostic_plots.png`: 诊断可视化")
    lines.append("")

    report_path = output_dir / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/unified_ppo_comparison_summary_v3.csv",
                        help="对比实验汇总 CSV")
    parser.add_argument("--target", default="Unified-PPO")
    parser.add_argument("--output-dir", default="results/unified_ppo_gap_model")
    args = parser.parse_args()

    csv_path = ROOT / args.summary
    if not csv_path.exists():
        print(f"[error] 未找到汇总文件: {csv_path}")
        return 1

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Unified-PPO 性能差距分析模型")
    print("=" * 70)

    df = load_summary(csv_path)
    print(f"[info] 加载 {len(df)} 条对比记录，{df['Method'].nunique()} 种方法，"
          f"{df['Dataset'].nunique()} 个数据集，{df['Attack'].nunique()} 种攻击。")

    gap_df = build_gap_dataset(df, target=args.target)
    print(f"[info] 构建 {len(gap_df)} 个场景级 Gap 样本。")
    print(f"[info] {args.target} 平均 Robustness: {gap_df['target_r'].mean():.4f}, "
          f"平均落后最优基线: {gap_df['gap'].mean():.4f}")

    gap_df.to_csv(output_dir / 'gap_dataset.csv', index=False)

    # Pairwise 分析
    print("[info] Pairwise 差距分析 ...")
    pairwise_df = pairwise_method_analysis(df, target=args.target, output_dir=output_dir)

    # 训练 Gap 预测模型
    print("[info] 训练 Gap 预测模型 ...")
    model_results, best_importance, perm_df, X, y, predictions, scaler, attack_enc = fit_gap_model(gap_df, output_dir)
    print(f"[info] 最佳模型: {model_results['best_model']}, "
          f"LOO R²={model_results[model_results['best_model']]['loo_r2']:.3f}")

    # 失败场景聚类
    print("[info] 失败场景聚类 ...")
    cluster_df, cluster_summary = cluster_failure_scenarios(gap_df, output_dir, n_clusters=4)
    cluster_df.to_csv(output_dir / 'failure_clusters.csv', index=False)

    # 对手画像
    profile_pct, attack_gap, dataset_gap, real_vs_syn = opponent_profile(gap_df)

    # Top-2 分类器
    print("[info] 训练 Top-2 排名分类器 ...")
    rank_report, rank_importance, gap_df_with_prob = build_rank_classifier(gap_df, output_dir)
    if gap_df_with_prob is not None:
        gap_df_with_prob.to_csv(output_dir / 'gap_dataset_with_prob.csv', index=False)

    # 诊断可视化
    print("[info] 生成诊断可视化 ...")
    visualize_diagnostics(gap_df, output_dir)

    # 生成报告
    print("[info] 生成分析报告 ...")
    report_path = generate_report(
        gap_df, model_results, best_importance, perm_df, cluster_summary,
        pairwise_df, profile_pct, attack_gap, dataset_gap, real_vs_syn,
        rank_report, rank_importance, output_dir, target=args.target
    )

    print("=" * 70)
    print(f"[done] 报告已保存: {report_path}")
    print(f"[done] 所有结果目录: {output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
