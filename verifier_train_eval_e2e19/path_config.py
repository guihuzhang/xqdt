from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "verifier_train_eval_e2e19" / "data"
ERROR_PREDICTION_E2E19_DIR = DATA_DIR

VERIFIER_TRAIN_EVAL_DIR = PROJECT_ROOT / "verifier_train_eval_e2e19"
PROMPT_BASELINES_DIR = VERIFIER_TRAIN_EVAL_DIR / "prompt_baselines"
PROMPT_BASELINE_EVAL_DIR = PROMPT_BASELINES_DIR / "eval"
PROMPT_BASELINE_EVAL_LOG_DIR = PROMPT_BASELINES_DIR / "eval_logs"
XQDT_OUTPUTS_DIR = VERIFIER_TRAIN_EVAL_DIR / "xqdt_outputs"
XQDT_EVAL_DIR = XQDT_OUTPUTS_DIR / "eval"
XQDT_EVAL_LOG_DIR = XQDT_OUTPUTS_DIR / "eval_logs"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_E2E19_DIR = CHECKPOINTS_DIR / "checkpoints_e2e19"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")
    return path
