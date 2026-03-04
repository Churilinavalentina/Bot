from datetime import datetime, timezone
from t_tech.invest import CandleInterval

INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
START_DATE = datetime(2025, 11, 24, tzinfo=timezone.utc)
END_DATE = datetime(2026, 2, 24, tzinfo=timezone.utc)
# 99 самых ликвидных
TICKERS = [
    "AFKS", "AFLT", "AKRN", "ALRS", "APTK", "AQUA", "ASTR", "BANEP", "BELU", "BSPB",
    "CBOM", "CHMF", "CNRU", "DATA", "DOMRF", "ELFV", "ENPG", "ETLN", "EUTR",
    "FEES", "FESH", "FIXR", "FLOT", "GAZP", "GEMC", "GMKN", "HEAD", "HNFG", "HYDR",
    "IRAO", "LEAS", "LENT", "LKOH", "LSNGP", "LSRG", "MAGN", "MBNK", "MDMG", "MGNT",
    "MGTSP", "MOEX", "MRKC", "MRKP", "MRKU", "MRKV", "MSNG", "MSRS", "MTLR", "MTLRP", "MTSS",
    "NKHP", "NKNC", "NKNCP", "NLMK", "NMTP", "NVTK", "OGKB", "OZON", "OZPH", "PHOR", "PIKK",
    "PLZL", "POSI", "PRMD", "RAGR", "RENI", "RNFT", "ROSN", "RTKM",
    "RTKMP", "RUAL", "SBER", "SBERP", "SELG", "SFIN", "SGZH", "SMLT", "SNGS", "SNGSP", "SOFL",
    "SPBE", "SVAV", "SVCB", "T", "TATN", "TATNP", "TGKA", "TRMK", "TRNFP", "UGLD", "UPRO",
    "UWGN", "VKCO", "VSEH", "VSMO", "VTBR", "WUSH", "X5", "YDEX"
]

EPOCHS = 25
BATCH_SIZE= 30_000 * 20
# Количество тиков для обучения
DELTA_T= 100
# Количество тиков для предсказания
DELTA_T_FUTURE= 100
# желаемый проент прироста для предсказания
K = 0.15
TRAIN_PART = 0.8
LOSS = 1e-4

NUM_EXECUTOR_READ = 1