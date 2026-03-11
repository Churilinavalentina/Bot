from datetime import datetime, timezone
from t_tech.invest import CandleInterval
from t_tech.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX

#FOLDER = '/content/drive/MyDrive/Colab Notebooks/' # Collab
FOLDER = '' # локальный запуск

#TARGET = INVEST_GRPC_API # Боевой
TARGET = INVEST_GRPC_API_SANDBOX

IS_SANDBOX = 1

INTERVAL = CandleInterval.CANDLE_INTERVAL_1_MIN
START_DATE = datetime(2025, 12, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 3, 10, tzinfo=timezone.utc)
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
BATCH_SIZE= 30_000 * 10
# Количество тиков для обучения
DELTA_T= 100
# Количество тиков для предсказания
DELTA_T_FUTURE= 30
# желаемый проент прироста для предсказания
K = 0.1
TRAIN_PART = 0.8
LOSS = 1e-4

NUM_EXECUTOR_READ = 1