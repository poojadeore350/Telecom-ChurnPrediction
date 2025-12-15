"""
Exploratory Data Analysis Utilities for Telecom Customer Churn Prediction

This module contains advanced EDA functions with interactive visualizations
and statistical analysis specifically designed for churn analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ChurnEDA:
    """
    Advanced Exploratory Data Analysis class for telecom churn prediction.
    
    This class provides comprehensive EDA capabilities including statistical analysis,
    interactive visualizations, and churn-specific insights.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def dataset_overview(self, df: pd.DataFrame) -> None:
        """
        Provide a comprehensive overview of the dataset.
        
        Args:
            df: Input DataFrame
        """
        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        
        print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types summary
        print(f"\n📋 Data Types Summary:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n❌ Missing Values:")
            missing_percent = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing_data[missing_data > 0],
                'Percentage': missing_percent[missing_data > 0]
            }).sort_values('Missing Count', ascending=False)
            print(missing_df)
        else:
            print(f"\n✅ No missing values found!")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        print(f"\n🔄 Duplicate Rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
        
        # Churn distribution if available
        if 'Churn' in df.columns:
            churn_dist = df['Churn'].value_counts()
            churn_rate = churn_dist[True] / len(df) * 100 if True in churn_dist else 0
            print(f"\n🎯 Churn Distribution:")
            print(f"   Non-Churn: {churn_dist.get(False, 0):,} ({100-churn_rate:.1f}%)")
            print(f"   Churn: {churn_dist.get(True, 0):,} ({churn_rate:.1f}%)")
    
    def numerical_features_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze numerical features with comprehensive statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with statistical summary
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        print(f"\n📈 NUMERICAL FEATURES ANALYSIS ({len(numerical_cols)} features)")
        print("=" * 60)
        
        stats_summary = []
        
        for col in numerical_cols:
            col_stats = {
                'Feature': col,
                'Count': df[col].count(),
                'Mean': df[col].mean(),
                'Std': df[col].std(),
                'Min': df[col].min(),
                'Q1': df[col].quantile(0.25),
                'Median': df[col].median(),
                'Q3': df[col].quantile(0.75),
                'Max': df[col].max(),
                'Skewness': df[col].skew(),
                'Kurtosis': df[col].kurtosis(),
                'Outliers_IQR': self._count_outliers_iqr(df[col]),
                'Zeros': (df[col] == 0).sum(),
                'Unique_Values': df[col].nunique()
            }
            stats_summary.append(col_stats)
        
        stats_df = pd.DataFrame(stats_summary)
        return stats_df
    
    def categorical_features_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical analysis
        """
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        print(f"\n📊 CATEGORICAL FEATURES ANALYSIS ({len(categorical_cols)} features)")
        print("=" * 60)
        
        cat_summary = []
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            most_frequent = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
            most_frequent_count = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            
            cat_stats = {
                'Feature': col,
                'Unique_Values': unique_values,
                'Most_Frequent': most_frequent,
                'Most_Frequent_Count': most_frequent_count,
                'Most_Frequent_Pct': (most_frequent_count / len(df)) * 100,
                'Cardinality': 'High' if unique_values > 10 else 'Low'
            }
            cat_summary.append(cat_stats)
        
        cat_df = pd.DataFrame(cat_summary)
        return cat_df
    
    def plot_churn_distribution(self, df: pd.DataFrame) -> None:
        """
        Create comprehensive churn distribution visualizations.
        
        Args:
            df: Input DataFrame
        """
        if 'Churn' not in df.columns:
            print("Churn column not found in dataset")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        churn_counts = df['Churn'].value_counts()
        axes[0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
        
        # Bar plot with percentages
        churn_pct = df['Churn'].value_counts(normalize=True) * 100
        bars = axes[1].bar(['No Churn', 'Churn'], churn_pct.values, 
                          color=['#2ecc71', '#e74c3c'], alpha=0.8)
        axes[1].set_title('Churn Rate (%)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Percentage')
        
        # Add value labels on bars
        for bar, value in zip(bars, churn_pct.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n🎯 Churn Analysis Summary:")
        print(f"   Total Customers: {len(df):,}")
        print(f"   Churned Customers: {churn_counts.get(True, 0):,}")
        print(f"   Churn Rate: {churn_pct.get(True, 0):.2f}%")
        print(f"   Class Balance Ratio: 1:{churn_counts.get(False, 0)/churn_counts.get(True, 1):.1f}")
    
    def plot_numerical_distributions(self, df: pd.DataFrame, cols_per_row: int = 3) -> None:
        """
        Plot distributions of numerical features.
        
        Args:
            df: Input DataFrame
            cols_per_row: Number of columns per row in subplot grid
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        n_cols = len(numerical_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5*cols_per_row, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color=self.colors[i % len(self.colors)], density=True)
                
                # Add KDE line
                try:
                    df[col].dropna().plot.kde(ax=axes[i], color='red', linewidth=2)
                except:
                    pass
                
                axes[i].set_title(f'{col}\n(Skew: {df[col].skew():.2f})', fontsize=10)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_churn_by_categorical(self, df: pd.DataFrame) -> None:
        """
        Analyze churn rates by categorical features.
        
        Args:
            df: Input DataFrame
        """
        if 'Churn' not in df.columns:
            print("Churn column not found in dataset")
            return
        
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        # Filter out high cardinality features (like State)
        categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
        
        if not categorical_cols:
            print("No suitable categorical features found for analysis")
            return
        
        n_cols = len(categorical_cols)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Calculate churn rates
                churn_rates = df.groupby(col)['Churn'].agg(['count', 'sum', 'mean']).reset_index()
                churn_rates['churn_rate'] = churn_rates['mean'] * 100
                
                # Create grouped bar plot
                x_pos = np.arange(len(churn_rates))
                width = 0.35
                
                bars1 = axes[i].bar(x_pos - width/2, churn_rates['count'] - churn_rates['sum'], 
                                   width, label='No Churn', color='#2ecc71', alpha=0.8)
                bars2 = axes[i].bar(x_pos + width/2, churn_rates['sum'], 
                                   width, label='Churn', color='#e74c3c', alpha=0.8)
                
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].set_title(f'Churn Distribution by {col}')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(churn_rates[col])
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Add churn rate labels
                for j, (bar, rate) in enumerate(zip(bars2, churn_rates['churn_rate'])):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Churn Analysis by Categorical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_numerical_vs_churn(self, df: pd.DataFrame, top_n: int = 8) -> None:
        """
        Compare numerical features between churned and non-churned customers.
        
        Args:
            df: Input DataFrame
            top_n: Number of top features to plot
        """
        if 'Churn' not in df.columns:
            print("Churn column not found in dataset")
            return
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        # Select top features based on correlation with churn
        correlations = []
        for col in numerical_cols:
            corr = abs(df[col].corr(df['Churn'].astype(int)))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:top_n]]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(top_features):
            if i < len(axes):
                # Box plot
                churn_data = df[df['Churn'] == True][col].dropna()
                no_churn_data = df[df['Churn'] == False][col].dropna()
                
                axes[i].boxplot([no_churn_data, churn_data], labels=['No Churn', 'Churn'],
                               patch_artist=True, 
                               boxprops=dict(facecolor='lightblue', alpha=0.7),
                               medianprops=dict(color='red', linewidth=2))
                
                axes[i].set_title(f'{col}\n(Corr: {correlations[i][1]:.3f})', fontsize=10)
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Top Numerical Features vs Churn', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Statistical significance tests
        print(f"\n📊 Statistical Significance Tests (Top {top_n} Features):")
        print("-" * 60)
        
        for col, corr in correlations[:top_n]:
            churn_data = df[df['Churn'] == True][col].dropna()
            no_churn_data = df[df['Churn'] == False][col].dropna()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(churn_data, no_churn_data)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"{col:25} | Corr: {corr:6.3f} | p-value: {p_value:8.6f} {significance}")
    
    def correlation_analysis(self, df: pd.DataFrame) -> None:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            df: Input DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            print("Not enough numerical features for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"\n🔗 Highly Correlated Feature Pairs (|r| > 0.8):")
            print("-" * 60)
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"{feat1:25} ↔ {feat2:25} | r = {corr:6.3f}")
        else:
            print(f"\n✅ No highly correlated feature pairs found (|r| > 0.8)")
    
    def create_interactive_churn_dashboard(self, df: pd.DataFrame) -> None:
        """
        Create an interactive dashboard for churn analysis using Plotly.
        
        Args:
            df: Input DataFrame
        """
        if 'Churn' not in df.columns:
            print("Churn column not found in dataset")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Distribution', 'Customer Service Calls vs Churn',
                          'Total Charges Distribution', 'International Plan vs Churn'),
            specs=[[{"type": "pie"}, {"type": "box"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Churn Distribution Pie Chart
        churn_counts = df['Churn'].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Churn', 'Churn'], values=churn_counts.values,
                  name="Churn Distribution", hole=0.3),
            row=1, col=1
        )
        
        # 2. Customer Service Calls Box Plot
        for churn_val in [False, True]:
            fig.add_trace(
                go.Box(y=df[df['Churn'] == churn_val]['Customer service calls'],
                      name=f'Churn: {churn_val}', showlegend=False),
                row=1, col=2
            )
        
        # 3. Total Charges Histogram
        if 'Total day charge' in df.columns:
            total_charges = (df['Total day charge'] + df['Total eve charge'] + 
                           df['Total night charge'] + df['Total intl charge'])
            
            for churn_val in [False, True]:
                fig.add_trace(
                    go.Histogram(x=total_charges[df['Churn'] == churn_val],
                               name=f'Churn: {churn_val}', opacity=0.7, showlegend=False),
                    row=2, col=1
                )
        
        # 4. International Plan vs Churn
        if 'International plan' in df.columns:
            intl_churn = df.groupby('International plan')['Churn'].agg(['count', 'sum']).reset_index()
            intl_churn['churn_rate'] = (intl_churn['sum'] / intl_churn['count']) * 100
            
            fig.add_trace(
                go.Bar(x=intl_churn['International plan'], y=intl_churn['churn_rate'],
                      name='Churn Rate (%)', showlegend=False),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive Churn Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        fig.show()
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """
        Count outliers using IQR method.
        
        Args:
            series: Pandas Series
            
        Returns:
            Number of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        return outliers
    
    def generate_eda_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive EDA report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing EDA insights
        """
        print("🔍 Generating Comprehensive EDA Report...")
        print("=" * 60)
        
        # Dataset overview
        self.dataset_overview(df)
        
        # Numerical analysis
        numerical_stats = self.numerical_features_analysis(df)
        
        # Categorical analysis
        categorical_stats = self.categorical_features_analysis(df)
        
        # Visualizations
        print(f"\n📊 Creating Visualizations...")
        self.plot_churn_distribution(df)
        self.plot_numerical_distributions(df)
        self.plot_churn_by_categorical(df)
        self.plot_numerical_vs_churn(df)
        self.correlation_analysis(df)
        
        # Interactive dashboard
        print(f"\n🎛️ Creating Interactive Dashboard...")
        self.create_interactive_churn_dashboard(df)
        
        report = {
            'numerical_stats': numerical_stats,
            'categorical_stats': categorical_stats,
            'dataset_shape': df.shape,
            'churn_rate': df['Churn'].mean() * 100 if 'Churn' in df.columns else None
        }
        
        print(f"\n✅ EDA Report Generated Successfully!")
        return report