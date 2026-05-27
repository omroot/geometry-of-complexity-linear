"""Data loading, cleaning, and preparation utilities for IPCA workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class IPCAMatrices:
    """Container for matrices required by InstrumentedPCA."""

    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    char_cols: List[str]


class DataPipeline:
    """Download, clean, and prepare characteristic panel data for IPCA."""

    def __init__(
        self,
        id_col: str = "permno",
        time_col: str = "yyyymm",
        ret_col: str = "excess_ret",
    ) -> None:
        self.id_col = id_col
        self.time_col = time_col
        self.ret_col = ret_col

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------

    def download_openap_data(
        self,
        start_yyyymm: str,
        end_yyyymm: str,
        backend: str = "pandas",
    ) -> pd.DataFrame:
        """Download OpenAP signals and keep the requested date window."""
        try:
            import openassetpricing as oap
        except ImportError as exc:
            raise ImportError(
                "openassetpricing is required. Install with: pip install openassetpricing"
            ) from exc

        openap = oap.OpenAP()
        df = openap.dl_all_signals(backend)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        df[self.time_col] = pd.to_datetime(
            df[self.time_col].astype(str), format="%Y%m", errors="coerce"
        )
        df = df.dropna(subset=[self.id_col, self.time_col])
        return df.sort_values([self.id_col, self.time_col]).reset_index(drop=True)

    def download_sp500_returns_wrds(
        self,
        start_date: str = "1980-01-01",
        end_date: Optional[str] = None,
        include_delist_adjustment: bool = True,
    ) -> pd.DataFrame:
        """Download monthly returns for S&P 500 constituents only, point-in-time.

        Joins ``crsp.msf`` with ``crsp.msp500list`` in SQL so only stocks that
        were *actually in the index* at each observation date are returned —
        no survivorship bias, no post-hoc filtering required.

        Parameters
        ----------
        start_date : str
            First month to include (``YYYY-MM-DD``).
        end_date : str, optional
            Last month to include. Defaults to today.
        include_delist_adjustment : bool
            If True, compound raw return with delisting return (``dlret``) from
            ``crsp.msedelist`` and expose as ``ret_adj``.
        """
        try:
            import wrds
        except ImportError as exc:
            raise ImportError("wrds package is required. Install with: pip install wrds") from exc

        db = wrds.Connection()

        where_parts = ["msf.date >= %(start_date)s"]
        params: dict[str, object] = {"start_date": start_date}
        if end_date is not None:
            where_parts.append("msf.date <= %(end_date)s")
            params["end_date"] = end_date

        where_sql = " AND ".join(where_parts)

        # The INNER JOIN on msp500list enforces point-in-time S&P 500 membership:
        # a row is included only when msf.date falls within the stock's index spell.
        if include_delist_adjustment:
            sql = f"""
                SELECT
                    msf.permno,
                    msf.date,
                    msf.ret,
                    dl.dlret,
                    msf.prc,
                    msf.shrout,
                    msf.cfacpr,
                    msf.cfacshr
                FROM crsp.msf AS msf
                INNER JOIN crsp.msp500list AS sp
                    ON  msf.permno = sp.permno
                    AND msf.date  >= sp.start
                    AND (msf.date <= sp.ending OR sp.ending IS NULL)
                LEFT JOIN crsp.msedelist AS dl
                    ON  msf.permno = dl.permno
                    AND msf.date   = dl.dlstdt
                WHERE {where_sql}
                ORDER BY msf.permno, msf.date
            """
        else:
            sql = f"""
                SELECT
                    msf.permno,
                    msf.date,
                    msf.ret,
                    msf.prc,
                    msf.shrout,
                    msf.cfacpr,
                    msf.cfacshr
                FROM crsp.msf AS msf
                INNER JOIN crsp.msp500list AS sp
                    ON  msf.permno = sp.permno
                    AND msf.date  >= sp.start
                    AND (msf.date <= sp.ending OR sp.ending IS NULL)
                WHERE {where_sql}
                ORDER BY msf.permno, msf.date
            """

        try:
            out = db.raw_sql(sql, params=params, date_cols=["date"])
        finally:
            db.close()

        out = out.copy()
        out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
        for col in ["prc", "shrout", "cfacpr", "cfacshr"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        if "dlret" in out.columns:
            out["dlret"] = pd.to_numeric(out["dlret"], errors="coerce")
            out["ret_adj"] = (1.0 + out["ret"].fillna(0.0)) * (
                1.0 + out["dlret"].fillna(0.0)
            ) - 1.0
            both_missing = out["ret"].isna() & out["dlret"].isna()
            out.loc[both_missing, "ret_adj"] = np.nan

        out[self.time_col] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
        return out

    # Cleaning / merging primitives were moved to :mod:`src.preprocess.panel`.
    # The class retains only the download + model-panel construction methods.

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    def build_model_panel(
        self,
        df: pd.DataFrame,
        shift_target: bool = True,
        target_col: str = "y_ipca",
        extra_exclude: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build model-ready panel with target column and integer entity/time ids."""
        out = df.copy()
        out[self.time_col] = pd.to_datetime(
            out[self.time_col].astype(str), format="%Y%m", errors="coerce"
        )
        out = out.dropna(subset=[self.id_col, self.time_col]).sort_values(
            [self.id_col, self.time_col]
        )

        if shift_target:
            out[target_col] = out.groupby(self.id_col)[self.ret_col].shift(-1)
        else:
            out[target_col] = out[self.ret_col]

        exclude = {
            self.id_col,
            self.time_col,
            "ret",
            "ret_adj",
            "excess_ret",
            "year",
            target_col,
            "prc",
            "shrout",
            "cfacpr",
            "cfacshr",
            "mcap",
        }
        if extra_exclude:
            exclude.update(extra_exclude)
        char_cols = [
            c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
        ]

        out = out[[self.id_col, self.time_col, target_col] + char_cols].replace(
            [np.inf, -np.inf], np.nan
        )
        out = out.dropna(subset=[target_col]).copy()

        out["i_idx"] = out[self.id_col].factorize(sort=True)[0].astype(np.int64)
        out["t_idx"] = out[self.time_col].factorize(sort=True)[0].astype(np.int64)
        return out

    def split_train_test(
        self,
        panel_df: pd.DataFrame,
        cutoff: str | pd.Timestamp = "2015-12-31",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split panel into train and test by calendar date."""
        cutoff_ts = pd.Timestamp(cutoff)
        train_df = panel_df[panel_df[self.time_col] <= cutoff_ts].copy()
        test_df = panel_df[panel_df[self.time_col] > cutoff_ts].copy()
        return train_df, test_df

    @staticmethod
    def normalize_train_test(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        char_cols: List[str],
        clip_quantiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize using train-only statistics to avoid look-ahead leakage."""
        train_out = train_df.copy()
        test_out = test_df.copy()

        mu = train_out[char_cols].mean(axis=0)
        sd = train_out[char_cols].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)

        train_out[char_cols] = (train_out[char_cols] - mu) / sd
        test_out[char_cols] = (test_out[char_cols] - mu) / sd

        if clip_quantiles is not None:
            q_lo, q_hi = clip_quantiles
            lo = train_out[char_cols].quantile(q_lo / 100.0)
            hi = train_out[char_cols].quantile(q_hi / 100.0)
            train_out[char_cols] = train_out[char_cols].clip(lo, hi, axis=1)
            test_out[char_cols] = test_out[char_cols].clip(lo, hi, axis=1)

        return train_out, test_out

    @staticmethod
    def to_ipca_matrices(
        panel_df: pd.DataFrame,
        char_cols: List[str],
        target_col: str = "y_ipca",
    ) -> IPCAMatrices:
        """Convert panel DataFrame into X, y, indices arrays ready for InstrumentedPCA."""
        X = panel_df[char_cols].to_numpy(np.float64)
        y = panel_df[target_col].to_numpy(np.float64)
        indices = panel_df[["i_idx", "t_idx"]].to_numpy(np.int64)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return IPCAMatrices(X=X[mask], y=y[mask], indices=indices[mask], char_cols=char_cols)
