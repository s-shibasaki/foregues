import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict, Any, cast
from itertools import permutations
import logging


logger = logging.getLogger(__name__)


class SoftBinning(nn.Module):
    """
    標準化済み数値特徴量をソフトビニングするモジュール
    
    Args:
        num_bins: ビンの数
        temperature: ソフトマックスの温度パラメータ（低いほどハード、高いほどソフト）
        init_range: ビン中心の初期化範囲 [-init_range, init_range]
    """
    
    def __init__(self, num_bins: int = 10, temperature: float = 1.0, init_range: float = 3.0):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        
        # 学習可能なビン中心（標準化されたデータを想定して初期化）
        self.bin_centers = nn.Parameter(
            torch.linspace(-init_range, init_range, num_bins)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 標準化済み数値特徴量 (...,) 任意の形状
        Returns:
            binned: ソフトビニング結果 (..., num_bins)
        """
        # 入力の形状を保存
        original_shape = x.shape
        
        # xを展開して計算しやすくする
        x_flat = x.view(-1, 1)  # (N, 1)
        bin_centers = self.bin_centers.view(1, -1)  # (1, num_bins)
        
        # 各ビン中心との距離を計算（負の二乗距離）
        distances = -(x_flat - bin_centers) ** 2  # (N, num_bins)
        
        # 温度でスケールしてソフトマックス
        soft_assignments = F.softmax(distances / self.temperature, dim=-1)  # (N, num_bins)
        
        # 元の形状に戻す
        output_shape = original_shape + (self.num_bins,)
        return soft_assignments.view(output_shape)




class SoftBinnedLinear(nn.Module):
    """
    数値特徴量をSoftBinningしてからLinear変換するモジュール
    
    Args:
        num_bins: ビンの数
        d_token: 出力次元
        temperature: ソフトマックスの温度パラメータ
        init_range: ビン中心の初期化範囲
        dropout: ドロップアウト率
    """
    
    def __init__(self, num_bins: int = 10, d_token: int = 192, temperature: float = 1.0, 
                 init_range: float = 3.0):
        super().__init__()
        self.soft_binning = SoftBinning(
            num_bins=num_bins,
            temperature=temperature,
            init_range=init_range
        )
        self.linear = nn.Linear(num_bins, d_token)

        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        # Linearレイヤーの重み初期化 (Xavier uniform)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 標準化済み数値特徴量 (...,) 任意の形状
        Returns:
            output: (..., d_token) Linear変換後の特徴量
        """
        # SoftBinning適用
        binned = self.soft_binning(x)  # (..., num_bins)
        
        # Linear変換
        output = self.linear(binned)  # (..., d_token)
        
        return output


class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化（共通利用）"""

    def __init__(self, numerical_features: List[str], categorical_features: Dict[str, int], feature_aliases: Dict[str, str],
                 d_token: int = 192, num_bins: int = 10, binning_temperature: float = 1.0,
                 binning_init_range: float = 3.0, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_aliases = feature_aliases

        # 数値特徴量用のSoftBinnedLinear（特徴量ごとに個別）
        self.numerical_tokenizers = nn.ModuleDict()
        for feature in numerical_features:
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.numerical_tokenizers:
                self.numerical_tokenizers[tokenizer_name] = SoftBinnedLinear(
                    num_bins=num_bins,
                    d_token=d_token,
                    temperature=binning_temperature,
                    init_range=binning_init_range,
                )

        # カテゴリ特徴量用の埋め込み
        self.categorical_tokenizers = nn.ModuleDict()
        for feature, vocab_size in categorical_features.items():
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.categorical_tokenizers:
                self.categorical_tokenizers[tokenizer_name] = nn.Embedding(vocab_size, d_token, padding_idx=0) 

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # カテゴリ埋め込みの初期化
        for embedding in self.categorical_tokenizers.values():
            embedding = cast(nn.Embedding, embedding)
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
            # padding_idxは0で固定
            if embedding.padding_idx is not None:
                nn.init.constant_(embedding.weight[embedding.padding_idx], 0)
        
        # LayerNormの初期化（通常はデフォルトで適切）
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, x_num: Optional[Dict[str, torch.Tensor]] = None, 
                x_cat: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            x_num: 数値特徴量 - 各値は任意の形状のテンソル
            x_cat: カテゴリ特徴量 - 各値は任意の形状のテンソル
        Returns:
            tokens: (..., num_tokens, d_token) - トークン化された特徴量
        """
        tokens = []

        # 数値特徴量のトークン化
        if x_num is not None:
            for name, feature_values in x_num.items():
                tokenizer_name = self.feature_aliases.get(name, name)
                tokenizer = self.numerical_tokenizers[tokenizer_name]
                
                # NaN処理
                nan_mask = torch.isnan(feature_values)
                clean_values = torch.where(nan_mask, torch.zeros_like(feature_values), feature_values)
                
                # SoftBinnedLinearで変換
                token = tokenizer(clean_values)  # (..., d_token)

                # NaN位置のトークンをゼロにする
                nan_mask = nan_mask.unsqueeze(-1)  # (..., 1)
                token = torch.where(nan_mask, torch.zeros_like(token), token)
                tokens.append(token)

        # カテゴリ特徴量のトークン化
        if x_cat is not None:
            for name, feature_values in x_cat.items():
                tokenizer_name = self.feature_aliases.get(name, name)
                tokenizer = self.categorical_tokenizers[tokenizer_name]
                token = tokenizer(feature_values)  # (..., d_token)
                tokens.append(token)

        # トークンを結合
        result = torch.stack(tokens, dim=-2)  # (..., num_tokens, d_token)

        # 正規化とドロップアウト
        result = self.norm(result)
        result = self.dropout(result)
        
        return result
    

class AttentionHead(nn.Module):
    def __init__(self, d_token: int = 192, d_head: int = 8):
        super().__init__()
        self.q = nn.Linear(d_token, d_head)
        self.k = nn.Linear(d_token, d_head)
        self.v = nn.Linear(d_token, d_head)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # Attention用の線形層の初期化 (Xavier uniform)
        for linear in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)

    def forward(self, x, mask=None):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        dim_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)

        # マスクを適用
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, seq_len) -> (batch_size, seq_len, seq_len) にブロードキャスト
            scores = scores.masked_fill(~mask.bool(), -1e9)

        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)
    
class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション"""

    def __init__(self, d_token: int = 192, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_token % n_heads == 0

        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_token, self.d_head) for _ in range(n_heads)]
        )
        self.output_linear = nn.Linear(d_token, d_token)

    def _init_weights(self):
        """重みの初期化"""
        # 出力層の初期化
        nn.init.xavier_uniform_(self.output_linear.weight)
        if self.output_linear.bias is not None:
            nn.init.constant_(self.output_linear.bias, 0)

    def forward(self, x, mask=None):
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""

    def __init__(self, d_token: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_token, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_token)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # He初期化 (GELU活性化関数に適している)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x  # (batch_size, seq_len, d_token)

class TransformerBlock(nn.Module):
    """Transformerブロック"""

    def __init__(self, d_token: int, n_heads: int = 8, d_ffn: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_token * 4

        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.attention = MultiHeadAttention(d_token, n_heads, dropout)
        self.feed_forward = FeedForward(d_token, d_ffn, dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # LayerNormの初期化
        for norm in [self.norm1, self.norm2]:
            nn.init.constant_(norm.weight, 1.0)
            nn.init.constant_(norm.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden_state = self.norm1(x)
        x = x + self.attention(hidden_state, mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class SequenceTransformer(nn.Module):
    """時系列データ処理用のTransformer (CLSトークンを出力)"""

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: Optional[int] = None, 
                 dropout: float = 0.1, max_seq_len: int = 512):
        super().__init__()
        self.d_token = d_token
        self.max_seq_len = max_seq_len

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # 学習可能な位置エンコーディング
        # CLSトークン + 最大シーケンス長分を用意
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len + 1, d_token))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # CLSトークンの初期化
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
        
        # LayerNormの初期化
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, sequence_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            sequence_tokens: (..., seq_len, d_token) - 時系列トークン
            mask: (..., seq_len) - シーケンスマスク (1=有効, 0=無効/パディング)
        Returns:
            cls_output: (..., d_token) - CLSトークンの出力
        """
        # 入力の形状を取得
        *batch_dims, seq_len, d_token = sequence_tokens.shape
        batch_size = int(np.prod(batch_dims))
        
        # シーケンス長のチェック
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. Truncating.")
            sequence_tokens = sequence_tokens[..., :self.max_seq_len, :]
            if mask is not None:
                mask = mask[..., :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # バッチ次元をまとめる
        sequence_tokens_flat = sequence_tokens.view(batch_size, seq_len, d_token)

        # [CLS]トークンを先頭に追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_token)
        tokens_with_cls = torch.cat([cls_tokens, sequence_tokens_flat], dim=1)  # (batch_size, seq_len + 1, d_token)

        # 位置エンコーディングを追加
        seq_len_with_cls = seq_len + 1
        pos_embeddings = self.position_embeddings[:, :seq_len_with_cls, :]  # (1, seq_len + 1, d_token)
        tokens_with_cls = tokens_with_cls + pos_embeddings

        # マスクも[CLS]トークン分を拡張
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len)
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask_with_cls = torch.cat([cls_mask, mask_flat], dim=1)  # (batch_size, seq_len + 1)
        else:
            mask_with_cls = None

        # Transformerブロックを通す
        x = tokens_with_cls
        for block in self.transformer_blocks:
            x = block(x, mask_with_cls)

        # [CLS]トークン（最初のトークン）を抽出
        cls_token = x[:, 0]
        cls_token = self.norm(cls_token)
        cls_token = self.dropout(cls_token)

        # 元の形状に戻す
        cls_output = cls_token.view(*batch_dims, d_token)

        return cls_output
    

class ForeguesModel(nn.Module):
    """統合されたモデル（多クラス分類用）"""

    def __init__(self,
                 sequence_names: List[str],
                 feature_aliases: Dict[str, str],
                 numerical_features: List[str],
                 categorical_features: Dict[str, int],

                 # 分類設定
                 num_classes: int = 3,  # クラス数（例：下降、横ばい、上昇）

                 # 次元数
                 d_token: int = 192,

                 # SoftBinning 設定
                 num_bins: int = 8,  # 計算効率とモデル複雑性のバランス
                 binning_temperature: float = 0.8,  # やや鋭い分布で特徴量の境界を明確化
                 binning_init_range: float = 2.5,  # 標準化データの99%をカバーする範囲

                 # 特徴量統合Transformer (軽量化)
                 ft_n_layers: int = 3,
                 ft_n_heads: int = 8,
                 ft_d_ffn: Optional[int] = None,

                 # 時系列統合Transformer (中程度の複雑性)
                 seq_n_layers: int = 3,
                 seq_n_heads: int = 8,
                 seq_d_ffn: Optional[int] = None,  # d_token * 3

                 # 過学習防止
                 dropout: float = 0.1):
        super().__init__()
        self.sequence_names = sequence_names
        self.feature_aliases = feature_aliases
        self.d_token = d_token
        self.num_classes = num_classes

        # dataset_params から必要な情報を抽出
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 共通のFeatureTokenizer（SoftBinning対応）
        self.tokenizer = FeatureTokenizer(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            feature_aliases=self.feature_aliases,
            d_token=d_token,
            num_bins=num_bins,
            binning_temperature=binning_temperature,
            binning_init_range=binning_init_range,
            dropout=dropout
        )

        # 時系列用のFTTransformer (特徴量統合用)
        self.sequence_ft_transformers = nn.ModuleDict()
        for seq_name in self.sequence_names:
            self.sequence_ft_transformers[seq_name] = SequenceTransformer(
                d_token=d_token,
                n_layers=ft_n_layers,
                n_heads=ft_n_heads,
                d_ffn=ft_d_ffn,
                dropout=dropout
            )

        # 時系列用のSequenceTransformer (時系列処理用)
        self.sequence_transformers = nn.ModuleDict()
        for seq_name in self.sequence_names:
            self.sequence_transformers[seq_name] = SequenceTransformer(
                d_token=d_token,
                n_layers=seq_n_layers,
                n_heads=seq_n_heads,
                d_ffn=seq_d_ffn,
                dropout=dropout
            )

        self.general_ft_transformer = SequenceTransformer(
            d_token=d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn=ft_d_ffn,
            dropout=dropout
        )

        # 最終統合用のFTTransformer
        self.final_ft_transformer = SequenceTransformer(
            d_token=d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn=ft_d_ffn,
            dropout=dropout
        )

        # 多クラス分類用のヘッド
        self.classifier_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, d_token // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 4, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # classifier_headの初期化
        for module in self.classifier_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self,
                x_num: Optional[Dict[str, torch.Tensor]] = None,
                x_cat: Optional[Dict[str, torch.Tensor]] = None,
                sequence_data: Optional[Dict[str, Dict]] = None):
        """
        Args:
            x_num: Dict - 一般数値特徴量 (batch_size,)
            x_cat: Dict - 一般カテゴリ特徴量 (batch_size,)
            sequence_data: Dict - 時系列データ
                例: {
                    'price_history': {
                        'x_num': {...},  # (batch_size, seq_len)
                        'x_cat': {...},  # (batch_size, seq_len)
                        'mask': ...      # (batch_size, seq_len)
                    }
                }
        Returns:
            output: (batch_size, num_classes) - クラス確率またはlogits
        """
        # バッチサイズを取得
        if x_num:
            batch_size = list(x_num.values())[0].shape[0]
        elif x_cat:
            batch_size = list(x_cat.values())[0].shape[0]
        elif sequence_data:
            first_seq = next(iter(sequence_data.values()))
            if 'x_num' in first_seq:
                batch_size = list(first_seq['x_num'].values())[0].shape[0]
            elif 'x_cat' in first_seq:
                batch_size = list(first_seq['x_cat'].values())[0].shape[0]
            else:
                raise ValueError("Cannot determine batch size from inputs")
        else:
            raise ValueError("No input data provided")
        
        # 1. 時系列データの処理
        sequence_features = []
        
        for seq_name, seq_data in (sequence_data or {}).items():
            # 時系列データから特徴量を抽出
            seq_x_num = seq_data.get('x_num', {})
            seq_x_cat = seq_data.get('x_cat', {})
            seq_mask = seq_data.get('mask', None)

            # FeatureTokenizer -> FTTransformer -> SequenceTransformer
            seq_tokens = self.tokenizer(seq_x_num or None, seq_x_cat or None)  # (batch_size, seq_len, num_features, d_token)
            
            # 各時系列ステップの特徴量を統合
            if seq_tokens.size(-2) > 0:  # 特徴量が存在する場合
                batch_size_seq, seq_len, num_features, d_token = seq_tokens.shape
                seq_tokens_reshaped = seq_tokens.view(batch_size_seq * seq_len, num_features, d_token)
                
                # FTTransformerで各時系列ステップの特徴量を統合
                step_features = self.sequence_ft_transformers[seq_name](seq_tokens_reshaped)  # (batch_size * seq_len, d_token)
                step_features = step_features.view(batch_size_seq, seq_len, d_token)  # (batch_size, seq_len, d_token)
                
                # SequenceTransformerで時系列を処理
                seq_feature = self.sequence_transformers[seq_name](step_features, seq_mask)  # (batch_size, d_token)
                sequence_features.append(seq_feature)

        # 2. 一般データの処理
        if x_num or x_cat:
            # 一般データをトークン化
            general_tokens = self.tokenizer(x_num, x_cat)  # (batch_size, num_features, d_token)
            
            # 一般データの特徴量を統合
            if general_tokens.size(-2) > 0:  # 特徴量が存在する場合
                general_feature = self.general_ft_transformer(general_tokens)  # (batch_size, d_token)
                sequence_features.append(general_feature)

        # 3. 最終統合
        if sequence_features:
            # 全ての特徴量を統合
            all_features = torch.stack(sequence_features, dim=1)  # (batch_size, num_feature_types, d_token)
            feature = self.final_ft_transformer(all_features)  # (batch_size, d_token)
        else:
            # 特徴量がない場合はゼロベクトル
            feature = torch.zeros(batch_size, self.d_token, device=next(self.parameters()).device)

        # 4. 最終的なクラス分類
        logits = self.classifier_head(feature)  # (batch_size, num_classes)
        
        return logits
