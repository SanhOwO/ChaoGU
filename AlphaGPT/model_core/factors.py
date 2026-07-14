import torch
import torch.nn as nn

from .vocab import FEATURE_NAMES


class AShareFactorEngineer:
    """A 股特征工程：12 维因子空间"""

    @staticmethod
    def compute_features(raw_dict):
        """
        raw_dict: dict with keys
            'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover',
            'market_cap', optional 'macd_hist'
        All tensors: [Stocks, Time]
        Returns: features [Stocks, Features, Time]
        """
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        amt = raw_dict['amount']       # 成交额
        turn = raw_dict['turnover']    # 换手率
        mc = raw_dict['market_cap']
        
        # 检测缺失列：turnover 和 market_cap 是否为0（CSV导入时的默认值）
        has_turnover = torch.abs(turn).sum() > 1e-3
        has_market_cap = torch.abs(mc).sum() > 1e-3
        
        if not has_turnover:
            logger.info(" turnover data missing, using volume change rate as proxy")
        if not has_market_cap:
            logger.info(" market_cap data missing, using amount as proxy")
        
        # 1. RET: 对数收益率
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
        
        # 2. TURN: 换手率标准化（换手率 / 20日MA换手率）
        # 如果 turnover 缺失，用 volume 变化率替代
        if has_turnover:
            turn_ma = AShareFactorEngineer._rolling_mean(turn, window=20)
            turn_norm = turn / (turn_ma + 1e-9)
        else:
            v_ma20 = AShareFactorEngineer._rolling_mean(v, window=20)
            turn_norm = v / (v_ma20 + 1e-9)  # volume 标准化替代
        
        # 3. VOLAT: 20日波动率
        volat = AShareFactorEngineer._rolling_std(ret, window=20)
        
        # 4. MOM: 20日动量（累积对数收益）
        mom = AShareFactorEngineer._rolling_sum(ret, window=20)
        
        # 5. LOG_AMT: 对数成交额
        log_amt = torch.log1p(amt + 1e-9)
        
        # 6. RSI: 14日 RSI
        rsi = AShareFactorEngineer._compute_rsi(ret, window=14)
        
        # 7. DEV: 偏离 20 日均线
        ma20 = AShareFactorEngineer._rolling_mean(c, window=20)
        dev = (c - ma20) / (ma20 + 1e-9)
        
        # 8. VR: 量比（成交量 / MA5成交量）
        v_ma5 = AShareFactorEngineer._rolling_mean(v, window=5)
        vr = v / (v_ma5 + 1e-9)
        
        # 9. LOG_MC: 对数流通市值
        # 如果 market_cap 缺失，用 amount（成交额）的截面排名替代
        if has_market_cap:
            log_mc = torch.log1p(mc + 1e-9)
        else:
            # 替代方案：用 amount 作为活跃度和规模的代理
            amt_proxy = torch.log1p(amt + 1e-9)
            log_mc = AShareFactorEngineer._cross_sectional_rank(amt_proxy) * 5.0  # 缩放到合理范围
        log_mc = torch.log1p(mc + 1e-9)
        
        # 10. HL_RANGE: 振幅 (high - low) / close
        hl_range = (h - l) / (c + 1e-9)
        
        # 11. CLOSE_POS: 收盘价在高低区间位置
        close_pos = (c - l) / (h - l + 1e-9)
        
        # 12. MACD_HIST: MACD 柱状图（近似用 price_momentum 替代，如果原始数据中没有）
        macd_hist = raw_dict.get('macd_hist')
        if macd_hist is None:
            # 近似：12日EMA - 26日EMA 的差分趋势
            ema12 = AShareFactorEngineer._rolling_mean(c, window=12)
            ema26 = AShareFactorEngineer._rolling_mean(c, window=26)
            macd_hist = (ema12 - ema26) / (ema26 + 1e-9)
        
        # 鲁棒归一化
        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)
        
        features = torch.stack([
            robust_norm(ret),
            robust_norm(turn_norm),
            robust_norm(volat),
            robust_norm(mom),
            robust_norm(log_amt),
            rsi,                      # RSI 已经在 0-100 范围，需额外处理
            robust_norm(dev),
            robust_norm(vr),
            log_mc,                   # 市值因子用原始值，截面排名更有效（后续处理）
            robust_norm(hl_range),
            close_pos,                # 0-1 范围，不需要归一化
            robust_norm(macd_hist)
        ], dim=1)
        
        # 特殊处理：RSI 映射到 [-1, 1]
        rsi_idx = 5
        features[:, rsi_idx, :] = (features[:, rsi_idx, :] - 50) / 50
        
        # 特殊处理：市值因子用截面排名（rank）更有预测力
        mc_idx = 8
        features[:, mc_idx, :] = AShareFactorEngineer._cross_sectional_rank(features[:, mc_idx, :])
        
        # 清理 NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return features
    
    @staticmethod
    def _rolling_mean(x, window):
        """时序 rolling mean，使用 unfold"""
        pad = torch.zeros((x.shape[0], window - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, window, 1).mean(dim=-1)
    
    @staticmethod
    def _rolling_std(x, window):
        """时序 rolling std"""
        pad = torch.zeros((x.shape[0], window - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, window, 1).std(dim=-1) + 1e-9
    
    @staticmethod
    def _rolling_sum(x, window):
        """时序 rolling sum"""
        pad = torch.zeros((x.shape[0], window - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, window, 1).sum(dim=-1)
    
    @staticmethod
    def _compute_rsi(ret, window=14):
        """计算 RSI 指标"""
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window - 1), device=gains.device, dtype=gains.dtype)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _cross_sectional_rank(x):
        """截面排名（rank 到 [-1, 1]）"""
        # x: [Stocks, Time]
        ranks = torch.zeros_like(x)
        for t in range(x.shape[1]):
            col = x[:, t]
            # argsort  twice gives rank
            sorted_idx = torch.argsort(col)
            rank = torch.argsort(sorted_idx).float()
            n = len(rank)
            if n > 1:
                rank = (rank - n / 2.0) / (n / 2.0)  # 映射到 [-1, 1]
            ranks[:, t] = rank
        return ranks
