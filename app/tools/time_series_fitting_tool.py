# -*- coding: utf-8 -*-
"""
时间序列拟合模型 - 基于Loss最小化的景观指数补全工具
功能：基于不连续的时间序列数据，训练回归模型补全缺失时间点的数值
支持多种拟合方法：随机森林、梯度提升树、样条插值
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QProgressBar,
    QTextEdit, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QFrame, QSlider, QDialog, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGB_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CatBoostRegressor = None
    CAT_AVAILABLE = False
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import UnivariateSpline, CubicSpline, interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class ModelWorker(QThread):
    progress_signal = pyqtSignal(str)
    progress_value = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.config = config
        self._is_running = True
        
    def stop(self):
        self._is_running = False
        
    def run(self):
        try:
            results = self.process_data()
            if self._is_running:
                self.finished_signal.emit(results)
            else:
                self.progress_signal.emit("任务已取消")
        except Exception as e:
            self.progress_signal.emit(f"错误: {str(e)}")
            
    def process_data(self):
        df = self.data.copy()
        model_type = self.config['model_type']
        time_granularity = self.config['time_granularity']
        smooth_factor = self.config.get('smooth_factor', 0.5)
        
        self.progress_signal.emit("开始数据预处理...")
        self.progress_value.emit(5)
        
        df, target_cols = self.preprocess_data(df)
        
        if not self._is_running:
            return None
        
        self.progress_signal.emit(f"数据预处理完成，共 {len(df)} 条有效记录")
        self.progress_signal.emit(f"发现 {len(target_cols)} 个指标: {', '.join(target_cols)}")
        self.progress_value.emit(10)
        
        watersheds = df['watershed'].unique()
        self.progress_signal.emit(f"发现 {len(watersheds)} 个流域")
        
        all_results = []
        all_metrics = []
        
        total = len(watersheds)
        for idx, watershed in enumerate(watersheds):
            if not self._is_running:
                return None
                
            self.progress_signal.emit(f"\n处理流域: {watershed} ({idx+1}/{total})")
            
            watershed_df = df[df['watershed'] == watershed].copy()
            
            result, metrics = self.process_watershed(
                watershed_df, watershed, model_type, time_granularity, smooth_factor, target_cols
            )
            
            if result is not None:
                all_results.append(result)
                all_metrics.append(metrics)
            
            progress = 10 + int((idx + 1) / total * 80)
            self.progress_value.emit(progress)
        
        if not self._is_running:
            return None
        
        self.progress_signal.emit("\n合并所有流域结果...")
        self.progress_value.emit(95)
        
        final_df = pd.concat(all_results, ignore_index=True)
        metrics_df = pd.DataFrame(all_metrics)
        
        self.progress_signal.emit("处理完成！")
        self.progress_value.emit(100)
        
        return {
            'data': final_df,
            'metrics': metrics_df
        }
    
    def preprocess_data(self, df):
        df = df.copy()
        
        df = df.rename(columns={
            'time': 'date',
            'std': 'NDVI_std',
            'mean': 'NDVI_mean'
        })
        
        df['date'] = df['date'].astype(str)
        
        df['year'] = df['date'].str[:4].astype(int)
        df['month'] = df['date'].str[4:6].astype(int)
        df['day'] = df['date'].str[6:8].astype(int)
        
        filter_month = self.config.get('filter_month', {})
        
        if filter_month.get('enabled', False):
            start_month = filter_month['start_month']
            end_month = filter_month['end_month']
            
            if start_month <= end_month:
                mask = (df['month'] >= start_month) & (df['month'] <= end_month)
            else:
                mask = (df['month'] >= start_month) | (df['month'] <= end_month)
            
            original_count = len(df)
            df = df[mask]
            self.progress_signal.emit(f"月份过滤: 保留 {start_month}-{end_month}月的数据，从 {original_count} 条减少到 {len(df)} 条")
        
        df['date_obj'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['days_from_start'] = (df['date_obj'] - pd.Timestamp('2016-01-01')).dt.days
        
        df['day_of_year'] = df['date_obj'].dt.dayofyear
        df['sin_season'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_season'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        if 'NDVI_mean' in df.columns:
            df = df[(df['NDVI_mean'] >= 0) & (df['NDVI_mean'] <= 1)]
        if 'CONTAG' in df.columns:
            df = df[(df['CONTAG'] >= 0) & (df['CONTAG'] <= 100)]
        if 'SIDI' in df.columns:
            df = df[(df['SIDI'] >= 0) & (df['SIDI'] <= 1)]
        
        exclude_cols = ['date', 'year', 'month', 'day', 'date_obj', 'days_from_start', 
                       'day_of_year', 'sin_season', 'cos_season', 'watershed']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if target_cols:
            df = df.dropna(subset=target_cols)
        
        return df, target_cols
    
    def process_watershed(self, df, watershed, model_type, time_granularity, smooth_factor, target_cols):
        if len(df) < 5:
            self.progress_signal.emit(f"  流域 {watershed} 数据点不足，跳过")
            return None, None
        
        df = df.sort_values('days_from_start')
        filter_month = self.config.get('filter_month', {})
        full_time_range = self.generate_time_range(df, time_granularity, filter_month)
        
        metrics = {'watershed': watershed}
        
        feature_cols = ['days_from_start', 'sin_season', 'cos_season']
        
        for col in target_cols:
            if not self._is_running:
                return None, None
                
            if col not in df.columns:
                continue
                
            self.progress_signal.emit(f"  拟合 {col}...")
            
            X_train = df['days_from_start'].values
            y_train = df[col].values
            X_full = full_time_range['days_from_start'].values
            
            if model_type == 'Spline':
                y_full_pred, r2, mse = self.fit_spline(
                    X_train, y_train, X_full, smooth_factor
                )
            elif model_type == 'CubicSpline':
                y_full_pred, r2, mse = self.fit_cubic_spline(
                    X_train, y_train, X_full
                )
            elif model_type == 'PLS':
                y_full_pred, r2, mse = self.fit_pls(
                    X_train, y_train, X_full, df, feature_cols, 
                    full_time_range, smooth_factor
                )
            elif model_type == 'SVR':
                y_full_pred, r2, mse = self.fit_svr(
                    X_train, y_train, X_full, df, feature_cols, 
                    full_time_range, smooth_factor
                )
            elif model_type == 'XGBoost':
                y_full_pred, r2, mse = self.fit_xgboost(
                    X_train, y_train, X_full, df, feature_cols, 
                    full_time_range, smooth_factor
                )
            elif model_type == 'CatBoost':
                y_full_pred, r2, mse = self.fit_catboost(
                    X_train, y_train, X_full, df, feature_cols, 
                    full_time_range, smooth_factor
                )
            else:
                y_full_pred, r2, mse = self.fit_tree_model(
                    X_train, y_train, X_full, df, feature_cols, 
                    full_time_range, model_type, smooth_factor
                )
            
            metrics[f'{col}_R2'] = r2
            metrics[f'{col}_MSE'] = mse
            
            self.progress_signal.emit(f"    {col}: R2={r2:.4f}, MSE={mse:.4f}")
            
            if col == 'CONTAG':
                y_full_pred = np.clip(y_full_pred, 0, 100)
            elif col in ['SIDI', 'NDVI_mean', 'NDVI_std']:
                y_full_pred = np.clip(y_full_pred, 0, 1)
            
            full_time_range[f'{col}_filled'] = y_full_pred
        
        full_time_range['watershed'] = watershed
        
        filled_cols = [f'{col}_filled' for col in target_cols if f'{col}_filled' in full_time_range.columns]
        result_df = full_time_range[['watershed', 'date', 'year', 'month'] + filled_cols]
        
        return result_df, metrics
    
    def fit_spline(self, X_train, y_train, X_full, smooth_factor):
        unique_mask = np.concatenate([[True], np.diff(X_train) > 0])
        X_unique = X_train[unique_mask]
        y_unique = y_train[unique_mask]
        
        if len(X_unique) < 4:
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
        
        y_var = np.var(y_unique)
        if y_var == 0:
            y_var = 1
        
        s = smooth_factor * y_var * len(X_unique) * 0.1
        
        try:
            spline = UnivariateSpline(X_unique, y_unique, s=s, k=3)
            y_full_pred = spline(X_full)
            y_train_pred = spline(X_train)
        except:
            y_full_pred = np.interp(X_full, X_unique, y_unique)
            y_train_pred = np.interp(X_train, X_unique, y_unique)
        
        r2 = r2_score(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)
        
        return y_full_pred, r2, mse
    
    def fit_svr(self, X_train, y_train, X_full, df, feature_cols, 
                full_time_range, smooth_factor):
        X_feat = df[feature_cols].values
        X_full_feat = full_time_range[feature_cols].values
        
        model_params = self.config.get('model_params', {})
        params = model_params.get('SVR', {})
        
        try:
            model = SVR(
                kernel='rbf', 
                C=params.get('C', 100), 
                gamma=params.get('gamma', 'scale'), 
                epsilon=params.get('epsilon', 0.1)
            )
            model.fit(X_feat, y_train)
            
            y_pred = model.predict(X_feat)
            y_full_pred = model.predict(X_full_feat)
            
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            
            if len(y_full_pred) >= 10:
                sigma = max(1, smooth_factor * 2)
                y_full_pred = gaussian_filter1d(y_full_pred, sigma=sigma)
            
            return y_full_pred, r2, mse
        except Exception as e:
            self.progress_signal.emit(f"    SVR拟合失败: {str(e)}")
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
    
    def fit_xgboost(self, X_train, y_train, X_full, df, feature_cols, 
                   full_time_range, smooth_factor):
        if XGBRegressor is None:
            self.progress_signal.emit(f"    XGBoost未安装，使用GradientBoosting代替")
            return self.fit_tree_model(X_train, y_train, X_full, df, feature_cols, 
                                    full_time_range, 'GradientBoosting', smooth_factor)
        
        X_feat = df[feature_cols].values
        X_full_feat = full_time_range[feature_cols].values
        
        model_params = self.config.get('model_params', {})
        params = model_params.get('XGBoost', {})
        
        try:
            model = XGBRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_feat, y_train)
            
            y_pred = model.predict(X_feat)
            y_full_pred = model.predict(X_full_feat)
            
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            
            if len(y_full_pred) >= 10:
                sigma = max(1, smooth_factor * 3)
                y_full_pred = gaussian_filter1d(y_full_pred, sigma=sigma)
            
            return y_full_pred, r2, mse
        except Exception as e:
            self.progress_signal.emit(f"    XGBoost拟合失败: {str(e)}")
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
    
    def fit_catboost(self, X_train, y_train, X_full, df, feature_cols, 
                   full_time_range, smooth_factor):
        if CatBoostRegressor is None:
            self.progress_signal.emit(f"    CatBoost未安装，使用GradientBoosting代替")
            return self.fit_tree_model(X_train, y_train, X_full, df, feature_cols, 
                                    full_time_range, 'GradientBoosting', smooth_factor)
        
        X_feat = df[feature_cols].values
        X_full_feat = full_time_range[feature_cols].values
        
        model_params = self.config.get('model_params', {})
        params = model_params.get('CatBoost', {})
        
        try:
            model = CatBoostRegressor(
                iterations=params.get('iterations', 200),
                depth=params.get('depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                random_seed=42,
                verbose=False
            )
            model.fit(X_feat, y_train)
            
            y_pred = model.predict(X_feat)
            y_full_pred = model.predict(X_full_feat)
            
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            
            if len(y_full_pred) >= 10:
                sigma = max(1, smooth_factor * 3)
                y_full_pred = gaussian_filter1d(y_full_pred, sigma=sigma)
            
            return y_full_pred, r2, mse
        except Exception as e:
            self.progress_signal.emit(f"    CatBoost拟合失败: {str(e)}")
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
    
    def fit_cubic_spline(self, X_train, y_train, X_full):
        unique_mask = np.concatenate([[True], np.diff(X_train) > 0])
        X_unique = X_train[unique_mask]
        y_unique = y_train[unique_mask]
        
        if len(X_unique) < 4:
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
        
        try:
            cs = CubicSpline(X_unique, y_unique, bc_type='natural')
            y_full_pred = cs(X_full)
            y_train_pred = cs(X_train)
        except:
            y_full_pred = np.interp(X_full, X_unique, y_unique)
            y_train_pred = np.interp(X_train, X_unique, y_unique)
        
        r2 = r2_score(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)
        
        return y_full_pred, r2, mse
    
    def fit_tree_model(self, X_train, y_train, X_full, df, feature_cols, 
                       full_time_range, model_type, smooth_factor):
        X_feat = df[feature_cols].values
        X_full_feat = full_time_range[feature_cols].values
        
        model_params = self.config.get('model_params', {})
        
        if model_type == 'RandomForest':
            params = model_params.get('RandomForest', {})
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 15),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
        else:
            params = model_params.get('GradientBoosting', {})
            model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=42
            )
        
        model.fit(X_feat, y_train)
        
        y_pred = model.predict(X_feat)
        y_full_pred = model.predict(X_full_feat)
        
        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        
        if len(y_full_pred) >= 10:
            sigma = max(1, smooth_factor * 5)
            y_full_pred = gaussian_filter1d(y_full_pred, sigma=sigma)
        
        return y_full_pred, r2, mse
    
    def fit_pls(self, X_train, y_train, X_full, df, feature_cols, 
                 full_time_range, smooth_factor):
        X_feat = df[feature_cols].values
        X_full_feat = full_time_range[feature_cols].values
        
        model_params = self.config.get('model_params', {})
        params = model_params.get('PLS', {})
        
        n_components = params.get('n_components', 5)
        n_components = min(len(X_feat) - 1, n_components)
        n_components = max(n_components, 2)
        
        try:
            model = PLSRegression(n_components=n_components, scale=True)
            model.fit(X_feat, y_train)
            
            y_pred = model.predict(X_feat).ravel()
            y_full_pred = model.predict(X_full_feat).ravel()
            
            r2 = r2_score(y_train, y_pred)
            mse = mean_squared_error(y_train, y_pred)
            
            if len(y_full_pred) >= 10:
                sigma = max(1, smooth_factor * 3)
                y_full_pred = gaussian_filter1d(y_full_pred, sigma=sigma)
            
            return y_full_pred, r2, mse
        except Exception as e:
            self.progress_signal.emit(f"    PLS拟合失败: {str(e)}")
            y_pred = np.interp(X_full, X_train, y_train)
            r2 = r2_score(y_train, np.interp(X_train, X_train, y_train))
            mse = mean_squared_error(y_train, np.interp(X_train, X_train, y_train))
            return y_pred, r2, mse
    
    def generate_time_range(self, df, granularity, filter_month=None):
        min_date = df['date_obj'].min()
        max_date = df['date_obj'].max()
        
        if granularity == 'yearly':
            years = range(min_date.year, max_date.year + 1)
            dates = [datetime(y, 7, 1) for y in years]
        elif granularity == 'quarterly':
            dates = []
            current = datetime(min_date.year, 1, 1)
            while current <= max_date:
                dates.append(current)
                month = current.month + 3
                year = current.year
                if month > 12:
                    month = 1
                    year += 1
                current = datetime(year, month, 15)
        elif granularity == 'monthly':
            dates = []
            current = datetime(min_date.year, min_date.month, 15)
            while current <= max_date:
                dates.append(current)
                month = current.month + 1
                year = current.year
                if month > 12:
                    month = 1
                    year += 1
                current = datetime(year, month, 15)
        elif granularity == 'monthly_4':
            dates = []
            current = datetime(min_date.year, min_date.month, 1)
            while current <= max_date:
                dates.append(current)
                day = current.day + 7
                month = current.month
                year = current.year
                if day > 28:
                    day = 1
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                current = datetime(year, month, day)
        else:
            dates = []
            current = min_date
            while current <= max_date:
                dates.append(current)
                current += timedelta(days=1)
        
        result = pd.DataFrame({
            'date_obj': dates,
            'days_from_start': [(d - pd.Timestamp('2016-01-01')).days for d in dates]
        })
        result['date'] = result['date_obj'].dt.strftime('%Y%m%d')
        result['year'] = result['date_obj'].dt.year
        result['month'] = result['date_obj'].dt.month
        
        result['day_of_year'] = result['date_obj'].dt.dayofyear
        result['sin_season'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['cos_season'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
        
        if filter_month and filter_month.get('enabled', False):
            start_month = filter_month['start_month']
            end_month = filter_month['end_month']
            
            if start_month <= end_month:
                mask = (result['month'] >= start_month) & (result['month'] <= end_month)
            else:
                mask = (result['month'] >= start_month) | (result['month'] <= end_month)
            
            original_count = len(result)
            result = result[mask].reset_index(drop=True)
        
        return result


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.set_facecolor('white')
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False


class PaginatedTable(QWidget):
    def __init__(self, parent=None, rows_per_page=500, filter_column=None):
        super().__init__(parent)
        
        self.rows_per_page = rows_per_page
        self.current_page = 1
        self.total_pages = 1
        self.df = None
        self.df_filtered = None
        self.filter_column = filter_column
        self.filter_value = None
    
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if filter_column:
            filter_layout = QHBoxLayout()
            filter_layout.addWidget(QLabel(f"{filter_column}筛选:"))
            self.combo_filter = QComboBox()
            self.combo_filter.addItem("全部")
            self.combo_filter.currentTextChanged.connect(self.on_filter_changed)
            filter_layout.addWidget(self.combo_filter)
            filter_layout.addStretch()
            layout.addLayout(filter_layout)
        
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        page_layout = QHBoxLayout()
        
        self.btn_first = QPushButton("首页")
        self.btn_first.clicked.connect(self.go_first)
        page_layout.addWidget(self.btn_first)
        
        self.btn_prev = QPushButton("上一页")
        self.btn_prev.clicked.connect(self.go_prev)
        page_layout.addWidget(self.btn_prev)
        
        self.label_page = QLabel("第 1 页 / 共 1 页")
        page_layout.addWidget(self.label_page)
        
        self.btn_next = QPushButton("下一页")
        self.btn_next.clicked.connect(self.go_next)
        page_layout.addWidget(self.btn_next)
        
        self.btn_last = QPushButton("末页")
        self.btn_last.clicked.connect(self.go_last)
        page_layout.addWidget(self.btn_last)
        
        page_layout.addWidget(QLabel("  每页显示:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setMinimum(100)
        self.spin_rows.setMaximum(2000)
        self.spin_rows.setValue(rows_per_page)
        self.spin_rows.setSingleStep(100)
        self.spin_rows.valueChanged.connect(self.on_rows_changed)
        page_layout.addWidget(self.spin_rows)
        page_layout.addWidget(QLabel("行"))
        
        page_layout.addStretch()
        
        self.label_total = QLabel("共 0 行")
        page_layout.addWidget(self.label_total)
        
        if filter_column:
            btn_export = QPushButton("导出")
            btn_export.clicked.connect(self.export_data)
            page_layout.addWidget(btn_export)
        
        layout.addLayout(page_layout)
        
        self.update_buttons()
    
    def on_filter_changed(self, value):
        self.filter_value = value if value != "全部" else None
        self.current_page = 1
        self.apply_filter()
        self.update_display()
        self.update_buttons()
    
    def apply_filter(self):
        if self.df is None:
            self.df_filtered = None
            return
        
        if self.filter_value is None or self.filter_column not in self.df.columns:
            self.df_filtered = self.df.copy()
        else:
            self.df_filtered = self.df[self.df[self.filter_column] == self.filter_value].copy()
        
        self.total_pages = max(1, (len(self.df_filtered) + self.rows_per_page - 1) // self.rows_per_page) if len(self.df_filtered) > 0 else 1
    
    def set_data(self, df):
        self.df = df
        self.current_page = 1
        self.filter_value = None
        
        if self.filter_column and df is not None and self.filter_column in df.columns:
            unique_values = sorted(df[self.filter_column].unique())
            self.combo_filter.blockSignals(True)
            self.combo_filter.clear()
            self.combo_filter.addItem("全部")
            self.combo_filter.addItems([str(v) for v in unique_values])
            self.combo_filter.blockSignals(False)
        
        self.apply_filter()
        self.update_display()
        self.update_buttons()
    
    def update_display(self):
        df_to_show = self.df_filtered
        
        if df_to_show is None or len(df_to_show) == 0:
            self.table.clear()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.label_page.setText("第 1 页 / 共 1 页")
            self.label_total.setText("共 0 行")
            return
        
        start_row = (self.current_page - 1) * self.rows_per_page
        end_row = min(start_row + self.rows_per_page, len(df_to_show))
        
        page_df = df_to_show.iloc[start_row:end_row]
        
        self.table.clear()
        self.table.setRowCount(len(page_df))
        self.table.setColumnCount(len(page_df.columns))
        self.table.setHorizontalHeaderLabels(page_df.columns.tolist())
        
        for i in range(len(page_df)):
            for j in range(len(page_df.columns)):
                item = QTableWidgetItem(str(page_df.iloc[i, j]))
                self.table.setItem(i, j, item)
        
        self.table.resizeColumnsToContents()
        
        total_count = len(self.df_filtered) if self.df_filtered is not None else 0
        self.label_page.setText(f"第 {self.current_page} 页 / 共 {self.total_pages} 页")
        self.label_total.setText(f"共 {total_count} 行")
    
    def update_buttons(self):
        self.btn_first.setEnabled(self.current_page > 1)
        self.btn_prev.setEnabled(self.current_page > 1)
        self.btn_next.setEnabled(self.current_page < self.total_pages)
        self.btn_last.setEnabled(self.current_page < self.total_pages)
    
    def go_first(self):
        self.current_page = 1
        self.update_display()
        self.update_buttons()
    
    def go_prev(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.update_display()
            self.update_buttons()
    
    def go_next(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.update_display()
            self.update_buttons()
    
    def go_last(self):
        self.current_page = self.total_pages
        self.update_display()
        self.update_buttons()
    
    def on_rows_changed(self, value):
        self.rows_per_page = value
        if self.df_filtered is not None and len(self.df_filtered) > 0:
            self.total_pages = max(1, (len(self.df_filtered) + self.rows_per_page - 1) // self.rows_per_page)
            self.current_page = min(self.current_page, self.total_pages)
            self.update_display()
            self.update_buttons()
    
    def export_data(self):
        if self.df_filtered is None or len(self.df_filtered) == 0:
            return
        
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent(), "保存数据", "", "Excel文件 (*.xlsx)"
        )
        
        if file_path:
            try:
                self.df_filtered.to_excel(file_path, index=False, engine='openpyxl')
                QMessageBox.information(self.parent(), "成功", f"数据已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self.parent(), "错误", f"导出失败: {str(e)}")
    
    def clear(self):
        self.df = None
        self.df_filtered = None
        self.current_page = 1
        self.total_pages = 1
        self.filter_value = None
        if self.filter_column:
            self.combo_filter.blockSignals(True)
            self.combo_filter.clear()
            self.combo_filter.addItem("全部")
            self.combo_filter.blockSignals(False)
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.label_page.setText("第 1 页 / 共 1 页")
        self.label_total.setText("共 0 行")
        self.update_buttons()


class ScrollableCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt5.QtWidgets import QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.canvas_widget = QWidget()
        self.canvas_layout = QVBoxLayout(self.canvas_widget)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        self.canvas = MplCanvas(self, width=12, height=10)
        self.canvas_layout.addWidget(self.canvas)
        
        self.scroll_area.setWidget(self.canvas_widget)
        layout.addWidget(self.scroll_area)
        
        self.setMinimumHeight(400)
    
    def set_canvas_height(self, n_rows, row_height=3):
        height = max(10, n_rows * row_height)
        self.canvas.fig.set_figheight(height)
        self.canvas_widget.setMinimumHeight(int(height * 100))
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("时间序列拟合模型 - 景观指数补全工具")
        self.setGeometry(100, 100, 1400, 900)
        
        self.data = None
        self.results = None
        self.worker = None
        self.doy_results = None
        
        self.model_params = {
            'RandomForest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'GradientBoosting': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'min_samples_leaf': 2
            },
            'XGBoost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'CatBoost': {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.05
            },
            'SVR': {
                'C': 100,
                'epsilon': 0.1,
                'gamma': 'scale'
            },
            'PLS': {
                'n_components': 5
            }
        }
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_group = QGroupBox("控制面板")
        control_layout = QGridLayout()
        
        btn_load = QPushButton("加载数据")
        btn_load.clicked.connect(self.load_data)
        control_layout.addWidget(btn_load, 0, 0)
        
        self.btn_clear = QPushButton("清除数据")
        self.btn_clear.clicked.connect(self.clear_data)
        control_layout.addWidget(self.btn_clear, 0, 1)
        
        self.label_file = QLabel("未选择文件")
        control_layout.addWidget(self.label_file, 0, 2, 1, 3)
        
        control_layout.addWidget(QLabel("拟合方法:"), 1, 0)
        self.combo_model = QComboBox()
        model_options = ["Spline", "CubicSpline", "RandomForest", "GradientBoosting", "PLS", "SVR"]
        if XGB_AVAILABLE:
            model_options.append("XGBoost")
        if CAT_AVAILABLE:
            model_options.append("CatBoost")
        self.combo_model.addItems(model_options)
        self.combo_model.currentTextChanged.connect(self.on_model_changed)
        control_layout.addWidget(self.combo_model, 1, 1)
        
        control_layout.addWidget(QLabel("时间粒度:"), 1, 2)
        self.combo_granularity = QComboBox()
        self.combo_granularity.addItems(["daily", "quarterly", "yearly", "monthly", "monthly_4"])
        control_layout.addWidget(self.combo_granularity, 1, 3)
        
        control_layout.addWidget(QLabel("平滑程度:"), 2, 0)
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setMinimum(0)
        self.slider_smooth.setMaximum(100)
        self.slider_smooth.setValue(10)
        self.slider_smooth.valueChanged.connect(self.on_smooth_changed)
        control_layout.addWidget(self.slider_smooth, 2, 1)
        self.label_smooth = QLabel("10%")
        control_layout.addWidget(self.label_smooth, 2, 2)
        
        self.btn_params = QPushButton("参数设置")
        self.btn_params.clicked.connect(self.show_params_dialog)
        control_layout.addWidget(self.btn_params, 2, 3)
        
        control_layout.addWidget(QLabel("月份过滤:"), 3, 0)
        month_filter_layout = QHBoxLayout()
        self.spin_month_start = QSpinBox()
        self.spin_month_start.setRange(1, 12)
        self.spin_month_start.setValue(6)
        self.spin_month_start.setPrefix("起始: ")
        self.spin_month_start.setSuffix("月")
        month_filter_layout.addWidget(self.spin_month_start)
        
        self.spin_month_end = QSpinBox()
        self.spin_month_end.setRange(1, 12)
        self.spin_month_end.setValue(10)
        self.spin_month_end.setPrefix("结束: ")
        self.spin_month_end.setSuffix("月")
        month_filter_layout.addWidget(self.spin_month_end)
        
        self.check_filter_month = QCheckBox("启用月份过滤")
        self.check_filter_month.setChecked(True)
        self.check_filter_month.setToolTip("勾选后只使用指定月份的数据进行拟合，例如6-10月为生长季")
        month_filter_layout.addWidget(self.check_filter_month)
        month_filter_layout.addStretch()
        control_layout.addLayout(month_filter_layout, 3, 1, 1, 3)
        
        self.btn_run = QPushButton("全时序拟合")
        self.btn_run.clicked.connect(self.run_model)
        self.btn_run.setEnabled(False)
        control_layout.addWidget(self.btn_run, 4, 0)
        
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_model)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop, 4, 1)
        
        self.btn_export = QPushButton("导出结果")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        control_layout.addWidget(self.btn_export, 4, 2)
        
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar, 4, 3)
        
        hint_text = "Spline/CubicSpline适合平滑插值，RandomForest/GradientBoosting适合复杂非线性关系"
        if not XGB_AVAILABLE:
            hint_text += " (XGBoost未安装)"
        if not CAT_AVAILABLE:
            hint_text += " (CatBoost未安装)"
        self.label_hint = QLabel(f"提示: {hint_text}")
        self.label_hint.setStyleSheet("color: gray; font-size: 10px;")
        control_layout.addWidget(self.label_hint, 5, 0, 1, 4)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        tabs = QTabWidget()
        
        tab_log = QWidget()
        log_layout = QVBoxLayout(tab_log)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.text_log)
        tabs.addTab(tab_log, "运行日志")
        
        tab_data = QWidget()
        data_layout = QVBoxLayout(tab_data)
        self.paginated_data = PaginatedTable(self, rows_per_page=500, filter_column='watershed')
        data_layout.addWidget(self.paginated_data)
        tabs.addTab(tab_data, "原始数据")
        
        tab_results = QWidget()
        results_layout = QVBoxLayout(tab_results)
        self.paginated_results = PaginatedTable(self, rows_per_page=500, filter_column='watershed')
        results_layout.addWidget(self.paginated_results)
        tabs.addTab(tab_results, "补全结果")
        
        tab_metrics = QWidget()
        metrics_layout = QVBoxLayout(tab_metrics)
        self.table_metrics = QTableWidget()
        metrics_layout.addWidget(self.table_metrics)
        tabs.addTab(tab_metrics, "拟合指标")
        
        tab_plot = QWidget()
        plot_layout = QVBoxLayout(tab_plot)
        
        plot_control = QHBoxLayout()
        plot_control.addWidget(QLabel("选择流域:"))
        self.combo_watershed_plot = QComboBox()
        self.combo_watershed_plot.currentTextChanged.connect(self.update_plot)
        plot_control.addWidget(self.combo_watershed_plot)
        
        plot_control.addWidget(QLabel("选择年份:"))
        self.combo_year_plot = QComboBox()
        self.combo_year_plot.addItems(["所有时序"])
        self.combo_year_plot.currentTextChanged.connect(self.update_plot)
        plot_control.addWidget(self.combo_year_plot)
        
        plot_control.addStretch()
        plot_layout.addLayout(plot_control)
        
        self.scrollable_canvas = ScrollableCanvas(self)
        plot_layout.addWidget(self.scrollable_canvas)
        tabs.addTab(tab_plot, "可视化")
        
        tab_doy = QWidget()
        doy_layout = QVBoxLayout(tab_doy)
        
        doy_control = QHBoxLayout()
        doy_control.addWidget(QLabel("查看流域:"))
        self.combo_watershed_doy = QComboBox()
        self.combo_watershed_doy.currentTextChanged.connect(self.show_doy_result)
        doy_control.addWidget(self.combo_watershed_doy)
        
        doy_control.addWidget(QLabel("拟合方法:"))
        self.combo_doy_method = QComboBox()
        doy_model_options = ["Spline", "CubicSpline", "RandomForest", "GradientBoosting", "PLS", "SVR"]
        if XGB_AVAILABLE:
            doy_model_options.append("XGBoost")
        if CAT_AVAILABLE:
            doy_model_options.append("CatBoost")
        self.combo_doy_method.addItems(doy_model_options)
        doy_control.addWidget(self.combo_doy_method)
        
        self.btn_doy_fit = QPushButton("开始拟合所有流域")
        self.btn_doy_fit.clicked.connect(self.run_doy_fit_all)
        doy_control.addWidget(self.btn_doy_fit)
        
        doy_control.addStretch()
        doy_layout.addLayout(doy_control)
        
        self.scrollable_canvas_doy = ScrollableCanvas(self)
        doy_layout.addWidget(self.scrollable_canvas_doy)
        
        tabs.addTab(tab_doy, "DOY拟合")
        
        main_layout.addWidget(tabs)
        
        self.statusBar().showMessage("就绪")
    
    def clear_data(self):
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '确认', '有任务正在运行，确定要清除吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_model()
            else:
                return
        
        self.data = None
        self.results = None
        
        self.label_file.setText("未选择文件")
        self.text_log.clear()
        self.paginated_data.clear()
        self.paginated_results.clear()
        self.table_metrics.clearContents()
        self.table_metrics.setRowCount(0)
        self.combo_watershed_plot.clear()
        self.combo_year_plot.clear()
        self.combo_year_plot.addItems(["所有时序"])
        
        self.scrollable_canvas.canvas.fig.clear()
        self.scrollable_canvas.canvas.axes = self.scrollable_canvas.canvas.fig.add_subplot(111)
        self.scrollable_canvas.canvas.draw()
        
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_stop.setEnabled(False)
        
        self.statusBar().showMessage("数据已清除")
        self.log("数据已清除")
    
    def on_model_changed(self, model_name):
        pass
    
    def on_smooth_changed(self, value):
        self.label_smooth.setText(f"{value}%")
    
    def show_params_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("模型参数设置")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        tabs = QTabWidget()
        
        rf_widget = self.create_rf_params_widget()
        tabs.addTab(rf_widget, "RandomForest")
        
        gb_widget = self.create_gb_params_widget()
        tabs.addTab(gb_widget, "GradientBoosting")
        
        if XGB_AVAILABLE:
            xgb_widget = self.create_xgb_params_widget()
            tabs.addTab(xgb_widget, "XGBoost")
        
        if CAT_AVAILABLE:
            cat_widget = self.create_cat_params_widget()
            tabs.addTab(cat_widget, "CatBoost")
        
        svr_widget = self.create_svr_params_widget()
        tabs.addTab(svr_widget, "SVR")
        
        pls_widget = self.create_pls_params_widget()
        tabs.addTab(pls_widget, "PLS")
        
        layout.addWidget(tabs)
        
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("恢复默认")
        btn_reset.clicked.connect(lambda: self.reset_params_to_default(tabs))
        btn_layout.addWidget(btn_reset)
        
        btn_layout.addStretch()
        
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(lambda: self.save_params_from_dialog(tabs, dialog))
        btn_layout.addWidget(btn_ok)
        
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def create_rf_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['RandomForest']
        
        self.spin_rf_n_estimators = QSpinBox()
        self.spin_rf_n_estimators.setRange(10, 1000)
        self.spin_rf_n_estimators.setValue(params['n_estimators'])
        layout.addRow("n_estimators (树的数量):", self.spin_rf_n_estimators)
        
        self.spin_rf_max_depth = QSpinBox()
        self.spin_rf_max_depth.setRange(1, 50)
        self.spin_rf_max_depth.setValue(params['max_depth'])
        layout.addRow("max_depth (最大深度):", self.spin_rf_max_depth)
        
        self.spin_rf_min_samples_split = QSpinBox()
        self.spin_rf_min_samples_split.setRange(2, 100)
        self.spin_rf_min_samples_split.setValue(params['min_samples_split'])
        layout.addRow("min_samples_split (分裂最小样本数):", self.spin_rf_min_samples_split)
        
        self.spin_rf_min_samples_leaf = QSpinBox()
        self.spin_rf_min_samples_leaf.setRange(1, 100)
        self.spin_rf_min_samples_leaf.setValue(params['min_samples_leaf'])
        layout.addRow("min_samples_leaf (叶节点最小样本数):", self.spin_rf_min_samples_leaf)
        
        return widget
    
    def create_gb_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['GradientBoosting']
        
        self.spin_gb_n_estimators = QSpinBox()
        self.spin_gb_n_estimators.setRange(10, 1000)
        self.spin_gb_n_estimators.setValue(params['n_estimators'])
        layout.addRow("n_estimators (树的数量):", self.spin_gb_n_estimators)
        
        self.spin_gb_max_depth = QSpinBox()
        self.spin_gb_max_depth.setRange(1, 50)
        self.spin_gb_max_depth.setValue(params['max_depth'])
        layout.addRow("max_depth (最大深度):", self.spin_gb_max_depth)
        
        self.spin_gb_learning_rate = QDoubleSpinBox()
        self.spin_gb_learning_rate.setRange(0.001, 1.0)
        self.spin_gb_learning_rate.setDecimals(3)
        self.spin_gb_learning_rate.setSingleStep(0.01)
        self.spin_gb_learning_rate.setValue(params['learning_rate'])
        layout.addRow("learning_rate (学习率):", self.spin_gb_learning_rate)
        
        self.spin_gb_min_samples_leaf = QSpinBox()
        self.spin_gb_min_samples_leaf.setRange(1, 100)
        self.spin_gb_min_samples_leaf.setValue(params['min_samples_leaf'])
        layout.addRow("min_samples_leaf (叶节点最小样本数):", self.spin_gb_min_samples_leaf)
        
        return widget
    
    def create_xgb_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['XGBoost']
        
        self.spin_xgb_n_estimators = QSpinBox()
        self.spin_xgb_n_estimators.setRange(10, 1000)
        self.spin_xgb_n_estimators.setValue(params['n_estimators'])
        layout.addRow("n_estimators (树的数量):", self.spin_xgb_n_estimators)
        
        self.spin_xgb_max_depth = QSpinBox()
        self.spin_xgb_max_depth.setRange(1, 50)
        self.spin_xgb_max_depth.setValue(params['max_depth'])
        layout.addRow("max_depth (最大深度):", self.spin_xgb_max_depth)
        
        self.spin_xgb_learning_rate = QDoubleSpinBox()
        self.spin_xgb_learning_rate.setRange(0.001, 1.0)
        self.spin_xgb_learning_rate.setDecimals(3)
        self.spin_xgb_learning_rate.setSingleStep(0.01)
        self.spin_xgb_learning_rate.setValue(params['learning_rate'])
        layout.addRow("learning_rate (学习率):", self.spin_xgb_learning_rate)
        
        self.spin_xgb_subsample = QDoubleSpinBox()
        self.spin_xgb_subsample.setRange(0.1, 1.0)
        self.spin_xgb_subsample.setDecimals(2)
        self.spin_xgb_subsample.setSingleStep(0.1)
        self.spin_xgb_subsample.setValue(params['subsample'])
        layout.addRow("subsample (子采样比例):", self.spin_xgb_subsample)
        
        self.spin_xgb_colsample_bytree = QDoubleSpinBox()
        self.spin_xgb_colsample_bytree.setRange(0.1, 1.0)
        self.spin_xgb_colsample_bytree.setDecimals(2)
        self.spin_xgb_colsample_bytree.setSingleStep(0.1)
        self.spin_xgb_colsample_bytree.setValue(params['colsample_bytree'])
        layout.addRow("colsample_bytree (列采样比例):", self.spin_xgb_colsample_bytree)
        
        return widget
    
    def create_cat_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['CatBoost']
        
        self.spin_cat_iterations = QSpinBox()
        self.spin_cat_iterations.setRange(10, 1000)
        self.spin_cat_iterations.setValue(params['iterations'])
        layout.addRow("iterations (迭代次数):", self.spin_cat_iterations)
        
        self.spin_cat_depth = QSpinBox()
        self.spin_cat_depth.setRange(1, 16)
        self.spin_cat_depth.setValue(params['depth'])
        layout.addRow("depth (树的深度):", self.spin_cat_depth)
        
        self.spin_cat_learning_rate = QDoubleSpinBox()
        self.spin_cat_learning_rate.setRange(0.001, 1.0)
        self.spin_cat_learning_rate.setDecimals(3)
        self.spin_cat_learning_rate.setSingleStep(0.01)
        self.spin_cat_learning_rate.setValue(params['learning_rate'])
        layout.addRow("learning_rate (学习率):", self.spin_cat_learning_rate)
        
        return widget
    
    def create_svr_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['SVR']
        
        self.spin_svr_C = QDoubleSpinBox()
        self.spin_svr_C.setRange(0.1, 10000)
        self.spin_svr_C.setDecimals(1)
        self.spin_svr_C.setSingleStep(10)
        self.spin_svr_C.setValue(params['C'])
        layout.addRow("C (正则化参数):", self.spin_svr_C)
        
        self.spin_svr_epsilon = QDoubleSpinBox()
        self.spin_svr_epsilon.setRange(0.001, 1.0)
        self.spin_svr_epsilon.setDecimals(3)
        self.spin_svr_epsilon.setSingleStep(0.01)
        self.spin_svr_epsilon.setValue(params['epsilon'])
        layout.addRow("epsilon (容忍边界):", self.spin_svr_epsilon)
        
        self.combo_svr_gamma = QComboBox()
        self.combo_svr_gamma.addItems(['scale', 'auto'])
        self.combo_svr_gamma.setCurrentText(params['gamma'])
        layout.addRow("gamma (核系数):", self.combo_svr_gamma)
        
        return widget
    
    def create_pls_params_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        
        params = self.model_params['PLS']
        
        self.spin_pls_n_components = QSpinBox()
        self.spin_pls_n_components.setRange(1, 50)
        self.spin_pls_n_components.setValue(params['n_components'])
        layout.addRow("n_components (主成分数量):", self.spin_pls_n_components)
        
        return widget
    
    def save_params_from_dialog(self, tabs, dialog):
        self.model_params['RandomForest'] = {
            'n_estimators': self.spin_rf_n_estimators.value(),
            'max_depth': self.spin_rf_max_depth.value(),
            'min_samples_split': self.spin_rf_min_samples_split.value(),
            'min_samples_leaf': self.spin_rf_min_samples_leaf.value()
        }
        
        self.model_params['GradientBoosting'] = {
            'n_estimators': self.spin_gb_n_estimators.value(),
            'max_depth': self.spin_gb_max_depth.value(),
            'learning_rate': self.spin_gb_learning_rate.value(),
            'min_samples_leaf': self.spin_gb_min_samples_leaf.value()
        }
        
        if XGB_AVAILABLE:
            self.model_params['XGBoost'] = {
                'n_estimators': self.spin_xgb_n_estimators.value(),
                'max_depth': self.spin_xgb_max_depth.value(),
                'learning_rate': self.spin_xgb_learning_rate.value(),
                'subsample': self.spin_xgb_subsample.value(),
                'colsample_bytree': self.spin_xgb_colsample_bytree.value()
            }
        
        if CAT_AVAILABLE:
            self.model_params['CatBoost'] = {
                'iterations': self.spin_cat_iterations.value(),
                'depth': self.spin_cat_depth.value(),
                'learning_rate': self.spin_cat_learning_rate.value()
            }
        
        self.model_params['SVR'] = {
            'C': self.spin_svr_C.value(),
            'epsilon': self.spin_svr_epsilon.value(),
            'gamma': self.combo_svr_gamma.currentText()
        }
        
        self.model_params['PLS'] = {
            'n_components': self.spin_pls_n_components.value()
        }
        
        self.log("模型参数已更新")
        dialog.accept()
    
    def reset_params_to_default(self, tabs):
        self.model_params = {
            'RandomForest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'GradientBoosting': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'min_samples_leaf': 2
            },
            'XGBoost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'CatBoost': {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.05
            },
            'SVR': {
                'C': 100,
                'epsilon': 0.1,
                'gamma': 'scale'
            },
            'PLS': {
                'n_components': 5
            }
        }
        
        self.spin_rf_n_estimators.setValue(200)
        self.spin_rf_max_depth.setValue(15)
        self.spin_rf_min_samples_split.setValue(2)
        self.spin_rf_min_samples_leaf.setValue(1)
        
        self.spin_gb_n_estimators.setValue(200)
        self.spin_gb_max_depth.setValue(6)
        self.spin_gb_learning_rate.setValue(0.05)
        self.spin_gb_min_samples_leaf.setValue(2)
        
        if XGB_AVAILABLE:
            self.spin_xgb_n_estimators.setValue(200)
            self.spin_xgb_max_depth.setValue(6)
            self.spin_xgb_learning_rate.setValue(0.05)
            self.spin_xgb_subsample.setValue(0.8)
            self.spin_xgb_colsample_bytree.setValue(0.8)
        
        if CAT_AVAILABLE:
            self.spin_cat_iterations.setValue(200)
            self.spin_cat_depth.setValue(6)
            self.spin_cat_learning_rate.setValue(0.05)
        
        self.spin_svr_C.setValue(100)
        self.spin_svr_epsilon.setValue(0.1)
        self.combo_svr_gamma.setCurrentText('scale')
        
        self.spin_pls_n_components.setValue(5)
        
        self.log("参数已恢复为默认值")
        
    def load_data(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "请等待当前任务完成或停止后再加载新数据")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "Excel文件 (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                self.clear_data()
                
                self.data = pd.read_excel(file_path)
                self.label_file.setText(f"已加载: {os.path.basename(file_path)}")
                self.btn_run.setEnabled(True)
                
                self.paginated_data.set_data(self.data)
                self.log(f"成功加载数据: {file_path}")
                self.log(f"数据维度: {self.data.shape[0]} 行, {self.data.shape[1]} 列")
                self.log(f"列名: {list(self.data.columns)}")
                
                if 'watershed' in self.data.columns:
                    watersheds = self.data['watershed'].unique()
                    self.combo_watershed_doy.clear()
                    self.combo_watershed_doy.addItems(sorted(watersheds))
                
                self.statusBar().showMessage(f"已加载: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")
                
    def log(self, message):
        self.text_log.append(message)
        
    def display_data(self, df, table):
        table.setRowCount(min(len(df), 1000))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns)
        
        for i in range(min(len(df), 1000)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                table.setItem(i, j, item)
        
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
    def run_model(self):
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "有任务正在运行，请等待完成或停止")
            return
            
        config = {
            'model_type': self.combo_model.currentText(),
            'time_granularity': self.combo_granularity.currentText(),
            'smooth_factor': self.slider_smooth.value() / 100.0,
            'model_params': self.model_params,
            'filter_month': {
                'enabled': self.check_filter_month.isChecked(),
                'start_month': self.spin_month_start.value(),
                'end_month': self.spin_month_end.value()
            }
        }
        
        self.log(f"\n开始拟合模型...")
        self.log(f"拟合方法: {config['model_type']}")
        self.log(f"时间粒度: {config['time_granularity']}")
        self.log(f"平滑程度: {config['smooth_factor']:.2f}")
        
        if config['filter_month']['enabled']:
            start_m = config['filter_month']['start_month']
            end_m = config['filter_month']['end_month']
            self.log(f"月份过滤: 已启用 ({start_m}月 - {end_m}月)")
        else:
            self.log(f"月份过滤: 未启用 (使用全部数据)")
        
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.worker = ModelWorker(self.data, config)
        self.worker.progress_signal.connect(self.log)
        self.worker.progress_value.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_model_finished)
        self.worker.start()
    
    def stop_model(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)
            
            self.log("任务已停止")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("任务已停止")
        
    def on_model_finished(self, results):
        if results is None:
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return
            
        self.results = results
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(True)
        
        self.paginated_results.set_data(results['data'])
        self.display_data(results['metrics'], self.table_metrics)
        
        watersheds = results['data']['watershed'].unique()
        self.combo_watershed_plot.clear()
        self.combo_watershed_plot.addItems(sorted(watersheds))
        
        years = sorted(results['data']['year'].unique())
        self.combo_year_plot.clear()
        self.combo_year_plot.addItems(["所有时序"] + [str(y) for y in years])
        
        self.combo_watershed_doy.clear()
        self.combo_watershed_doy.addItems(sorted(watersheds))
        
        self.log("\n" + "="*50)
        self.log("拟合完成！")
        self.log(f"补全后数据: {len(results['data'])} 条记录")
        
        r2_cols = [col for col in results['metrics'].columns if col.endswith('_R2')]
        for col in r2_cols:
            metric_name = col.replace('_R2', '')
            if col in results['metrics'].columns:
                self.log(f"平均R2 ({metric_name}): {results['metrics'][col].mean():.4f}")
        
        self.statusBar().showMessage("拟合完成")
        
    def update_plot(self, *args):
        if self.results is None:
            return
        
        watershed = self.combo_watershed_plot.currentText()
        if not watershed:
            return
        
        selected_year = self.combo_year_plot.currentText()
        
        df = self.results['data']
        watershed_df = df[df['watershed'] == watershed].copy()
        
        if len(watershed_df) == 0:
            return
        
        if selected_year != "所有时序" and selected_year:
            year = int(selected_year)
            watershed_df = watershed_df[watershed_df['year'] == year]
            if len(watershed_df) == 0:
                return
        
        watershed_df = watershed_df.sort_values('date')
        watershed_df['date_obj'] = pd.to_datetime(watershed_df['date'], format='%Y%m%d')
        
        original_df = None
        if self.data is not None:
            orig = self.data.copy()
            if 'watershed' in orig.columns:
                original_df = orig[orig['watershed'] == watershed].copy()
                if len(original_df) > 0:
                    if 'time' in original_df.columns:
                        original_df['date'] = original_df['time'].astype(str)
                    if 'date' in original_df.columns:
                        original_df['date_obj'] = pd.to_datetime(original_df['date'].astype(str), format='%Y%m%d')
                    if selected_year != "所有时序" and selected_year:
                        year = int(selected_year)
                        original_df['year'] = original_df['date_obj'].dt.year
                        original_df = original_df[original_df['year'] == year]
        
        filled_cols = [col for col in watershed_df.columns if col.endswith('_filled')]
        titles = [col.replace('_filled', '') for col in filled_cols]
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        orig_cols = [col.replace('_filled', '') for col in filled_cols]
        
        n_points = len(watershed_df)
        line_width = 0.8 if n_points > 365 else 1.2
        
        self.scrollable_canvas.canvas.fig.clear()
        
        n_metrics = len(filled_cols)
        if n_metrics == 0:
            return
        
        n_cols = 2
        n_rows = (n_metrics + 1) // 2 + 1
        
        gs = self.scrollable_canvas.canvas.fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.3)
        
        for idx, (col, title, orig_col) in enumerate(zip(filled_cols, titles, orig_cols)):
            row = idx // 2
            col_idx = idx % 2
            ax = self.scrollable_canvas.canvas.fig.add_subplot(gs[row, col_idx])
            
            color = colors[idx % len(colors)]
            
            ax.plot(watershed_df['date_obj'], watershed_df[col], 
                   linewidth=line_width, color=color, label='拟合曲线', alpha=0.8)
            
            if original_df is not None and len(original_df) > 0 and orig_col in original_df.columns:
                orig_sorted = original_df.sort_values('date_obj')
                ax.scatter(orig_sorted['date_obj'], orig_sorted[orig_col], 
                          color='red', s=15, marker='o', label='原始数据', zorder=5, alpha=0.7)
            
            ax.set_title(title, fontsize=10, pad=15)
            ax.set_xlabel('时间', fontsize=9, labelpad=8)
            ax.set_ylabel(title, fontsize=9, labelpad=8)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=6, framealpha=0.8)
            
            for label in ax.get_xticklabels():
                label.set_horizontalalignment('right')
        
        ax_loss = self.scrollable_canvas.canvas.fig.add_subplot(gs[n_rows-1, :])
        metrics_df = self.results['metrics']
        current_metrics = metrics_df[metrics_df['watershed'] == watershed]
        
        if len(current_metrics) > 0:
            mse_cols = [col for col in current_metrics.columns if col.endswith('_MSE')]
            mse_labels = [col.replace('_MSE', '') for col in mse_cols]
            mse_values = [current_metrics[col].values[0] if col in current_metrics.columns else 0 for col in mse_cols]
            
            x_pos = np.arange(len(mse_labels))
            bar_colors = ['steelblue', 'forestgreen', 'indianred', 'darkorange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            bars = ax_loss.bar(x_pos, mse_values, color=bar_colors[:len(mse_labels)], alpha=0.7)
            
            ax_loss.set_xticks(x_pos)
            ax_loss.set_xticklabels(mse_labels, fontsize=8)
            ax_loss.set_ylabel('MSE (Loss)', fontsize=9, labelpad=8)
            ax_loss.set_title('拟合损失 (MSE)', fontsize=10, pad=15)
            ax_loss.tick_params(axis='y', labelsize=7)
            ax_loss.grid(True, alpha=0.3, axis='y')
            
            max_val = max(mse_values) if mse_values and max(mse_values) > 0 else 1
            for bar, val in zip(bars, mse_values):
                height = bar.get_height()
                if val > 0:
                    y_pos = height + max_val * 0.03
                    ax_loss.annotate(f'{val:.6f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=7,
                                   rotation=20)
            
            ax_loss.set_ylim(0, max_val * 1.2)
        
        self.scrollable_canvas.canvas.fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.15)
        self.scrollable_canvas.set_canvas_height(n_rows)
        self.scrollable_canvas.canvas.draw()
    
    def run_doy_fit_all(self):
        if self.data is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        
        method = self.combo_doy_method.currentText()
        
        self.log(f"\n开始DOY拟合所有流域...")
        self.log(f"拟合方法: {method}")
        
        self.progress_bar.setValue(0)
        self.btn_doy_fit.setEnabled(False)
        
        try:
            df = self.data.copy()
            
            if 'time' in df.columns:
                df['date'] = df['time'].astype(str)
            
            df['date_obj'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
            df['doy'] = df['date_obj'].dt.dayofyear
            df['year'] = df['date_obj'].dt.year
            
            if 'mean' in df.columns:
                df['NDVI_mean'] = df['mean']
            if 'std' in df.columns:
                df['NDVI_std'] = df['std']
            
            watersheds = df['watershed'].unique() if 'watershed' in df.columns else []
            
            if len(watersheds) == 0:
                QMessageBox.warning(self, "警告", "未找到流域数据")
                self.btn_doy_fit.setEnabled(True)
                return
            
            self.doy_results = {}
            
            exclude_cols = ['date', 'year', 'month', 'day', 'date_obj', 'days_from_start', 
                           'day_of_year', 'sin_season', 'cos_season', 'watershed', 'time']
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = [col for col in numeric_cols if col not in exclude_cols]
            
            if not metrics:
                QMessageBox.warning(self, "警告", "未找到可拟合的指标")
                self.btn_doy_fit.setEnabled(True)
                return
            
            self.log(f"发现 {len(metrics)} 个指标: {', '.join(metrics)}")
            
            total_watersheds = len(watersheds)
            
            for idx, watershed in enumerate(watersheds):
                self.log(f"\n处理流域: {watershed} ({idx+1}/{total_watersheds})")
                
                watershed_df = df[df['watershed'] == watershed].copy()
                
                results = self.fit_doy_watershed(watershed_df, watershed, method, metrics)
                
                if results:
                    self.doy_results[watershed] = results
                
                progress = int((idx + 1) / total_watersheds * 100)
                self.progress_bar.setValue(progress)
            
            self.combo_watershed_doy.blockSignals(True)
            self.combo_watershed_doy.clear()
            self.combo_watershed_doy.addItems(sorted(self.doy_results.keys()))
            self.combo_watershed_doy.blockSignals(False)
            
            if self.doy_results:
                first_watershed = sorted(self.doy_results.keys())[0]
                self.show_doy_result(first_watershed)
                self.log(f"\nDOY拟合完成！共处理 {len(self.doy_results)} 个流域")
            else:
                self.log("\nDOY拟合完成，但没有有效结果")
            
        except Exception as e:
            self.log(f"DOY拟合错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"DOY拟合失败: {str(e)}")
        finally:
            self.btn_doy_fit.setEnabled(True)
    
    def fit_doy_watershed(self, df, watershed, method, metrics):
        doy_range = np.arange(1, 366)
        results = {}
        
        for metric in metrics:
            df_metric = df.dropna(subset=['doy', metric])
            
            if len(df_metric) < 10:
                self.log(f"  {metric}: 数据点不足 ({len(df_metric)} 个)")
                results[metric] = None
                continue
            
            if method in ['Spline', 'CubicSpline']:
                df_agg = df_metric.groupby('doy')[metric].agg(['mean', 'std']).reset_index()
                df_agg = df_agg.dropna()
                
                if len(df_agg) < 4:
                    self.log(f"  {metric}: 唯一DOY点不足 ({len(df_agg)} 个)")
                    results[metric] = None
                    continue
                
                X_doy = df_agg['doy'].values
                y = df_agg['mean'].values
            else:
                X_doy = df_metric['doy'].values
                y = df_metric[metric].values
            
            try:
                if method == 'Spline':
                    if len(X_doy) >= 4:
                        y_var = np.var(y) if np.var(y) > 0 else 1
                        s = 0.5 * y_var * len(X_doy) * 0.5
                        spline = UnivariateSpline(X_doy, y, s=s, k=3)
                        y_pred = spline(doy_range)
                        y_train_pred = spline(X_doy)
                        
                        y_min = np.min(y)
                        y_max = np.max(y)
                        y_pred = np.clip(y_pred, y_min * 0.9, y_max * 1.1)
                        y_train_pred = np.clip(y_train_pred, y_min * 0.9, y_max * 1.1)
                    else:
                        y_pred = np.interp(doy_range, X_doy, y)
                        y_train_pred = np.interp(X_doy, X_doy, y)
                        
                elif method == 'CubicSpline':
                    if len(X_doy) >= 4:
                        cs = CubicSpline(X_doy, y, bc_type='natural')
                        y_pred = cs(doy_range)
                        y_train_pred = cs(X_doy)
                        
                        y_min = np.min(y)
                        y_max = np.max(y)
                        y_pred = np.clip(y_pred, y_min * 0.9, y_max * 1.1)
                        y_train_pred = np.clip(y_train_pred, y_min * 0.9, y_max * 1.1)
                    else:
                        y_pred = np.interp(doy_range, X_doy, y)
                        y_train_pred = np.interp(X_doy, X_doy, y)
                        
                else:
                    sin_doy = np.sin(2 * np.pi * X_doy / 365)
                    cos_doy = np.cos(2 * np.pi * X_doy / 365)
                    X_feat = np.column_stack([X_doy, sin_doy, cos_doy])
                    
                    if method == 'RandomForest':
                        params = self.model_params.get('RandomForest', {})
                        model = RandomForestRegressor(
                            n_estimators=params.get('n_estimators', 200),
                            max_depth=params.get('max_depth', 15),
                            min_samples_split=params.get('min_samples_split', 2),
                            min_samples_leaf=params.get('min_samples_leaf', 1),
                            random_state=42, n_jobs=-1
                        )
                    elif method == 'GradientBoosting':
                        params = self.model_params.get('GradientBoosting', {})
                        model = GradientBoostingRegressor(
                            n_estimators=params.get('n_estimators', 200),
                            max_depth=params.get('max_depth', 6),
                            learning_rate=params.get('learning_rate', 0.05),
                            min_samples_leaf=params.get('min_samples_leaf', 2),
                            random_state=42
                        )
                    elif method == 'PLS':
                        params = self.model_params.get('PLS', {})
                        n_components = params.get('n_components', 5)
                        n_components = min(len(X_doy) - 1, n_components)
                        n_components = max(n_components, 2)
                        model = PLSRegression(n_components=n_components, scale=True)
                    elif method == 'SVR':
                        params = self.model_params.get('SVR', {})
                        model = SVR(
                            kernel='rbf',
                            C=params.get('C', 100),
                            gamma=params.get('gamma', 'scale'),
                            epsilon=params.get('epsilon', 0.1)
                        )
                    elif method == 'XGBoost':
                        if XGBRegressor is None:
                            params = self.model_params.get('GradientBoosting', {})
                            model = GradientBoostingRegressor(
                                n_estimators=params.get('n_estimators', 200),
                                max_depth=params.get('max_depth', 6),
                                learning_rate=params.get('learning_rate', 0.05),
                                random_state=42
                            )
                        else:
                            params = self.model_params.get('XGBoost', {})
                            model = XGBRegressor(
                                n_estimators=params.get('n_estimators', 200),
                                max_depth=params.get('max_depth', 6),
                                learning_rate=params.get('learning_rate', 0.05),
                                subsample=params.get('subsample', 0.8),
                                colsample_bytree=params.get('colsample_bytree', 0.8),
                                random_state=42, n_jobs=-1
                            )
                    elif method == 'CatBoost':
                        if CatBoostRegressor is None:
                            params = self.model_params.get('GradientBoosting', {})
                            model = GradientBoostingRegressor(
                                n_estimators=params.get('n_estimators', 200),
                                max_depth=params.get('max_depth', 6),
                                learning_rate=params.get('learning_rate', 0.05),
                                random_state=42
                            )
                        else:
                            params = self.model_params.get('CatBoost', {})
                            model = CatBoostRegressor(
                                iterations=params.get('iterations', 200),
                                depth=params.get('depth', 6),
                                learning_rate=params.get('learning_rate', 0.05),
                                random_seed=42, verbose=False
                            )
                    
                    model.fit(X_feat, y)
                    y_train_pred = model.predict(X_feat).ravel()
                    
                    sin_doy_full = np.sin(2 * np.pi * doy_range / 365)
                    cos_doy_full = np.cos(2 * np.pi * doy_range / 365)
                    X_full = np.column_stack([doy_range, sin_doy_full, cos_doy_full])
                    y_pred = model.predict(X_full).ravel()
                    
                    if method in ['RandomForest', 'GradientBoosting', 'SVR', 'XGBoost', 'CatBoost'] and len(y_pred) >= 10:
                        y_pred = gaussian_filter1d(y_pred, sigma=3)
                
                r2 = r2_score(y, y_train_pred)
                mse = mean_squared_error(y, y_train_pred)
                
                self.log(f"  {metric}: R2={r2:.4f}, MSE={mse:.6f}")
                
                results[metric] = {
                    'X_doy': X_doy,
                    'y': y,
                    'y_pred': y_pred,
                    'r2': r2,
                    'mse': mse,
                    'df_metric': df_metric,
                    'method': method
                }
            except Exception as e:
                self.log(f"  {metric}: 拟合失败 - {str(e)}")
                results[metric] = None
        
        return results
    
    def show_doy_result(self, watershed):
        if not watershed or self.doy_results is None or watershed not in self.doy_results:
            return
        
        results = self.doy_results[watershed]
        method = self.combo_doy_method.currentText()
        
        metrics = [k for k in results.keys() if results[k] is not None]
        
        self.plot_doy_result(results, metrics, watershed, method)
    
    def run_doy_fit(self):
        pass
    
    def plot_doy_result(self, results, metrics, watershed, method):
        self.scrollable_canvas_doy.canvas.fig.clear()
        
        colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        n_metrics = len(metrics)
        if n_metrics == 0:
            self.scrollable_canvas_doy.canvas.fig.text(0.5, 0.5, '无有效数据', ha='center', va='center', fontsize=14)
            self.scrollable_canvas_doy.canvas.draw()
            return
        
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        gs = self.scrollable_canvas_doy.canvas.fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.4)
        
        for idx, metric in enumerate(metrics):
            row = idx // 2
            col_idx = idx % 2
            ax = self.scrollable_canvas_doy.canvas.fig.add_subplot(gs[row, col_idx])
            
            result = results[metric]
            df_metric = result['df_metric']
            method_used = result['method']
            color = colors[idx % len(colors)]
            
            if method_used in ['Spline', 'CubicSpline']:
                df_agg = df_metric.groupby('doy')[metric].agg(['mean', 'std']).reset_index()
                df_agg = df_agg.dropna()
                ax.scatter(df_agg['doy'], df_agg['mean'], 
                          c='red', s=20, alpha=0.7, label='平均值')
            else:
                for i, year in enumerate(sorted(df_metric['year'].unique())):
                    year_data = df_metric[df_metric['year'] == year]
                    plot_color = colors_map[i % 10]
                    ax.scatter(year_data['doy'], year_data[metric], 
                              c=[plot_color], s=15, alpha=0.6, label=f'{year}年')
            
            doy_range = np.arange(1, 366)
            ax.plot(doy_range, result['y_pred'], color=color, linewidth=2, label='拟合曲线')
            
            ax.set_xlabel('DOY', fontsize=9, labelpad=8)
            ax.set_ylabel(metric, fontsize=9, labelpad=8)
            ax.set_title(f'{metric}\nR²={result["r2"]:.4f}, MSE={result["mse"]:.6f}', fontsize=10, pad=15)
            ax.set_xlim(1, 365)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=6, ncol=2)
        
        self.scrollable_canvas_doy.canvas.fig.suptitle(f'{watershed} - DOY拟合 ({method})', fontsize=12, y=0.98)
        
        self.scrollable_canvas_doy.set_canvas_height(n_rows)
        self.scrollable_canvas_doy.canvas.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.1)
        self.scrollable_canvas_doy.canvas.draw()
        
    def export_results(self):
        if self.results is None:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "Excel文件 (*.xlsx)"
        )
        
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    self.results['data'].to_excel(writer, sheet_name='补全结果', index=False)
                    self.results['metrics'].to_excel(writer, sheet_name='拟合指标', index=False)
                    
                self.log(f"\n结果已导出: {file_path}")
                QMessageBox.information(self, "成功", f"结果已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '确认退出',
                '有任务正在运行，确定要退出吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait(3000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


def launch_tool_window(parent=None):
    import matplotlib
    matplotlib.use('Qt5Agg')

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        app.setStyle('Fusion')
        font = QFont("Microsoft YaHei", 9)
        app.setFont(font)

    window = MainWindow()
    window.setWindowTitle("时间序列拟合工具")

    if parent is not None:
        window.setWindowModality(Qt.NonModal)
        window.destroyed.connect(lambda: None)

    window.show()
    window.raise_()
    window.activateWindow()

    return window


if __name__ == '__main__':
    main()
