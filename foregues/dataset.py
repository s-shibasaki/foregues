import ta.volatility
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, List, Tuple
import ta.trend, ta.momentum, ta.volatility
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class ForeguesDataset(Dataset):
    def __init__(self, data, th1=0.0005, th2=0.0010, sequence_length=1440, prediction_period=288, preprocessing_params: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: 5分足のOHLCデータフレーム (インデックスがDatetimeIndexでソート済みであること)
            th1: 小さい閾値 (例: 0.0005 = 5pips)
            th2: 大きい閾値 (例: 0.0010 = 10pips)
            sequence_length: 約5日間の5分足本数 (5日 × 24時間 × 12本/時間 = 1440本)
            prediction_period: 約24時間の予測期間 (288本)
        """
        self.data = data.copy()
        self.th1 = th1
        self.th2 = th2
        self.sequence_length = sequence_length
        self.prediction_period = prediction_period

        assert th2 > th1, "th2 must be greater than th1"

        # 前処理パラメータの初期化
        self.preprocessing_params = preprocessing_params

        # データの前処理
        self._prepare_data()

        # 前処理パラメータがない場合はフィットを実行
        if self.preprocessing_params is None:
            self._fit_preprocessing()

        # 前処理の適用
        self._apply_preprocessing()

        # 有効なタイムスタンプの生成
        self._generate_valid_timestamps()

    def _prepare_data(self):
        """データの前処理（テクニカル指標の計算など）"""
        # 数値データをfloat32に変換
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(np.float32)
        
        # テクニカル指標の計算
        self._calculate_technical_indicators()

        # ラベル生成
        self._generate_labels()

    def _calculate_technical_indicators(self):
        """テクニカル指標の計算"""
        # 価格変化率
        self.data['price_change'] = self.data['close'].pct_change().astype(np.float32)
        self.data['price_change_5'] = self.data['close'].pct_change(5).astype(np.float32)
        self.data['price_change_20'] = self.data['close'].pct_change(20).astype(np.float32)

        # 移動平均
        self.data['sma_5'] = ta.trend.sma_indicator(self.data['close'], window=5).astype(np.float32)
        self.data['sma_20'] = ta.trend.sma_indicator(self.data['close'], window=20).astype(np.float32)

        # 移動平均からの乖離率
        self.data['close_sma5_ratio'] = (self.data['close'] / self.data['sma_5'] - 1).astype(np.float32)
        self.data['close_sma20_ratio'] = (self.data['close'] / self.data['sma_20'] - 1).astype(np.float32)

        # RSI
        self.data['rsi_14'] = ta.momentum.rsi(self.data['close'], window=14).astype(np.float32)

        # MACD
        macd = ta.trend.MACD(self.data['close'])
        self.data['macd'] = macd.macd().astype(np.float32)
        self.data['macd_signal'] = macd.macd_signal().astype(np.float32)
        self.data['macd_diff'] = macd.macd_diff().astype(np.float32)

        # ボリンジャーバンド
        bb = ta.volatility.BollingerBands(self.data['close'], window=20)
        self.data['bb_upper'] = bb.bollinger_hband().astype(np.float32)
        self.data['bb_lower'] = bb.bollinger_lband().astype(np.float32)
        self.data['bb_position'] = ((self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])).astype(np.float32)

        # ATR (ボラティリティ)
        self.data['atr'] = ta.volatility.average_true_range(self.data['high'], self.data['low'], self.data['close'], window=14).astype(np.float32)

        # 時間関連の特徴量 (カテゴリ)
        self.data['hour'] = self.data.index.hour
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month

        # ローソク足パターン (高値・安値・始値・終値の関係)
        self.data['body_size'] = (abs(self.data['close'] - self.data['open'])).astype(np.float32)
        self.data['upper_shadow'] = (self.data['high'] - np.maximum(self.data['open'], self.data['close'])).astype(np.float32)
        self.data['lower_shadow'] = (np.minimum(self.data['open'], self.data['close']) - self.data['low']).astype(np.float32)

        # ボラティリティ (過去N期間の価格変動の標準偏差)
        self.data['volatility_5'] = self.data['price_change'].rolling(window=5).std().astype(np.float32)
        self.data['volatility_20'] = self.data['price_change'].rolling(window=20).std().astype(np.float32)

        # 特徴量リストの定義
        self.numerical_features = [
            'open', 'high', 'low', 'close',
            'price_change', 'price_change_5', 'price_change_20',
            'sma_5', 'sma_20', 'close_sma5_ratio', 'close_sma20_ratio',
            'rsi_14', 'macd', 'macd_signal', 'macd_diff',
            'bb_upper', 'bb_lower', 'bb_position', 'atr',
            'body_size', 'upper_shadow', 'lower_shadow',
            'volatility_5', 'volatility_20'
        ]

        self.categorical_features = ['hour', 'dayofweek', 'month']

    def _generate_labels(self):
        """ラベルの生成"""
        labels = []

        for i in range(len(self.data)):
            # 現在のタイムスタンプ (t)
            current_idx = i

            # t-1のclose価格を取得
            if current_idx == 0:
                labels.append(-1)  # 最初の行はラベル生成不可
                continue

            prev_close = self.data['close'].iloc[current_idx - 1]

            # t+1から24時間後までの範囲を確認
            future_start_idx = current_idx + 1
            future_end_idx = current_idx + 1 + self.prediction_period

            if future_end_idx >= len(self.data):
                labels.append(-1) # 予測期間のデータが不足
                continue

            # 予測期間の高値・安値を取得
            future_data = self.data.iloc[future_start_idx:future_end_idx]
            future_high = future_data['high'].max()
            future_low = future_data['low'].min()

            # ラベル判定
            if future_low > prev_close - self.th1 and future_high >= prev_close + self.th2:
                label = 1  # 買いエントリー
            elif future_high < prev_close + self.th1 and future_low <= prev_close - self.th2:
                label = 2  # 売りエントリー
            else:
                label = 0  # 何もしない

            labels.append(label)

        self.data['label'] = np.array(labels, dtype=np.int64)

    def _fit_preprocessing(self):
        """前処理パラメータをフィット"""
        self.preprocessing_params = {
            'numerical_scalers': {},
            'categorical_encoders': {},
            'categorical_vocab_sizes': {}
        }

        # 数値特徴量の標準化パラメータをフィット
        for feature in self.numerical_features:
            scaler = StandardScaler()
            # NaNを除いてフィット
            valid_data = self.data[feature].dropna().values.reshape(-1, 1)
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.preprocessing_params['numerical_scalers'][feature] = {
                    'mean': float(scaler.mean_[0]) if scaler.mean_ is not None else 0.0,
                    'scale': float(scaler.scale_[0]) if scaler.scale_ is not None else 1.0,
                }
            else:
                # データがない場合のデフォルト
                self.preprocessing_params['numerical_scalers'][feature] = {
                    'mean': 0.0,
                    'scale': 1.0,
                }

        # カテゴリ特徴量のエンコーディングパラメータをフィット
        for feature in self.categorical_features:
            encoder = LabelEncoder()
            # 欠損値を<NULL>で置換してフィット
            feature_data = self.data[feature].fillna('<NULL>')
            unique_values = feature_data.unique()

            # <NULL>を0に、その他を1から割り当て
            if "<NULL>" in unique_values:
                label_mapping = {'<NULL>': 0}
                other_values = [v for v in unique_values if v != '<NULL>']
                for i, value in enumerate(sorted(other_values), 1):
                    label_mapping[value] = i
            else:
                label_mapping = {value: i for i, value in enumerate(sorted(unique_values))}
                label_mapping['<NULL>'] = 0  # 未知の値用
            
            self.preprocessing_params['categorical_encoders'][feature] = label_mapping
            self.preprocessing_params['categorical_vocab_sizes'][feature] = len(label_mapping)

    def _apply_preprocessing(self):
        """前処理の適用"""
        if self.preprocessing_params is None:
            raise ValueError("Preprocessing parameters are not set. Call _fit_preprocessing() first.")

        # 数値特徴量の標準化
        for feature in self.numerical_features:
            if feature in self.data.columns and feature in self.preprocessing_params['numerical_scalers']:
                params = self.preprocessing_params['numerical_scalers'][feature]
                # NaNはそのまま残す
                mask = ~self.data[feature].isna()
                self.data.loc[mask, feature] = ((self.data.loc[mask, feature] - params['mean']) / params['scale']).astype(np.float32)

        # カテゴリ特徴量のエンコーディング
        for feature in self.categorical_features:
            if feature in self.data.columns and feature in self.preprocessing_params['categorical_encoders']:
                mapping = self.preprocessing_params['categorical_encoders'][feature]
                # NaNは<NULL>としてエンコード
                self.data[feature] = self.data[feature].fillna('<NULL>').map(mapping).fillna(0).astype(np.int64)

    def _generate_valid_timestamps(self):
        """有効なタイムスタンプの生成"""
        valid_indices = []

        for i in range(len(self.data)):
            # ラベルが有効(-1でない)かチェック
            if self.data['label'].iloc[i] == -1:
                continue

            # 過去のsequence_length分のデータがあるかチェック
            if i < self.sequence_length:
                continue

            valid_indices.append(i)

        # 有効なタイムスタンプを保存
        self.valid_timestamps = [self.data.index[i] for i in valid_indices]
        self.valid_indices = valid_indices

    def _get_sequence_data(self, timestamp):
        """指定されたタイムスタンプの時系列データを取得"""
        # タイムスタンプのインデックスを取得
        timestamp_idx = self.data.index.get_loc(timestamp)

        # 過去sequence_length分のデータを取得
        start_idx = timestamp_idx - self.sequence_length
        end_idx = timestamp_idx  # tは含まない (t-1まで)

        sequence_data = self.data.iloc[start_idx:end_idx].copy()

        # 数値特徴量とカテゴリ特徴量を分離
        x_num = {}
        x_cat = {}

        # 数値特徴量
        for feature in self.numerical_features:
            if feature in sequence_data.columns:
                x_num[feature] = sequence_data[feature].values.astype(np.float32)

        # カテゴリ特徴量
        for feature in self.categorical_features:
            if feature in sequence_data.columns:
                x_cat[feature] = sequence_data[feature].values.astype(np.int64)
                
        # マスク生成 (NaNでない箇所を1とする)
        mask = np.ones(len(sequence_data), dtype=bool)

        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'mask': mask
        }
    
    def get_preprocessing_params(self):
        """前処理パラメータを取得"""
        if self.preprocessing_params is None:
            raise ValueError("Preprocessing parameters are not set. Call _fit_preprocessing() first.")
        return self.preprocessing_params.copy()
    
    def get_categorical_vocab_sizes(self):
        """カテゴリ特徴量のボキャブラリサイズを取得"""
        if self.preprocessing_params is None:
            raise ValueError("Preprocessing parameters are not set. Call _fit_preprocessing() first.")
        return self.preprocessing_params.get('categorical_vocab_sizes', {})

    def __len__(self):
        return len(self.valid_timestamps)
    
    def __getitem__(self, idx):
        timestamp = self.valid_timestamps[idx]
        timestamp_idx = self.valid_indices[idx]

        # ラベルを取得
        label = self.data['label'].iloc[timestamp_idx]

        # 現在時点 (t) の特徴量を取得
        current_data = self.data.iloc[timestamp_idx]

        # 一般的な数値・カテゴリ特徴量
        x_num_general = {}
        x_cat_general = {}

        for feature in self.numerical_features:
            if feature in current_data:
                x_num_general[feature] = np.float32(current_data[feature])
        for feature in self.categorical_features:
            if feature in current_data:
                x_cat_general[feature] = np.int64(current_data[feature])
                
        sequence_data = self._get_sequence_data(timestamp)

        return {
            'timestamp': str(timestamp),
            'x_num': x_num_general,
            'x_cat': x_cat_general,
            'sequence_data': {'price_history': sequence_data},
            'labels': label
        }