"""
Generate game trajectories against env servers and save as an HF DatasetDict
(train / validation splits) ready for train_sft_env.py.

Analogous to tokenize_instruct.py but for environment SFT tasks.

Run from /workspace/scripts/:
  # Single environment (--num_games / --max_turn override the built-in defaults)
  python -m envs.generate_trajectories --environment_names liars_dice \
      --output_path /path/to/dataset --num_games 50000

  # Multiple environments — uses built-in per-env defaults for num_games/max_turn,
  # generates each to a staging path then merges into output_path
  python -m envs.generate_trajectories \
      --environment_names gin_rummy liars_dice leduc_poker \
      --output_path /path/to/dataset

Score-based sampling:
  Some generators (e.g. leduc_poker) return (messages, score) tuples.  When
  --sample-by-score is set, each game is kept with probability
  clamp(score, 0, 1) ** score_power.  --wins-only is a stricter filter that
  discards any game where score <= 0.  For generators that return only
  messages (no score), all games are kept regardless of these flags.
"""

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset, DatasetDict, concatenate_datasets

from envs.shared_env import GAMES_TO_TASK_ID_RANGE, _log, init_env_pool
from envs.sft_env_configs import get_sft_trajectory_generator


# ── Process-pool worker ───────────────────────────────────────────────────────
# Each worker process loads the expert generator once via _worker_init, then
# handles multiple games sequentially. Using processes (not threads) gives each
# worker its own GIL so CPU-bound expert computation runs truly in parallel
# without contention. --num_workers controls how many concurrent env server
# connections are open, letting you tune without overloading either side.

_GENERATE_FN = None


def _worker_init(env_name: str) -> None:
    global _GENERATE_FN
    _GENERATE_FN = get_sft_trajectory_generator(env_name)


def _worker_play(
    game_id: int, endpoint: str, max_turn: int
) -> "list[dict] | tuple[list[dict], float] | None":
    return _GENERATE_FN(game_id, endpoint, max_turn)

# ─────────────────────────────────────────────────────────────────────────────

MIN_ASSISTANT_TURNS = 1


def _sliding_windows(conv: list[dict], window_turns: int, window_step: int) -> list[list[dict]]:
    """
    Split a conversation into overlapping sub-conversations.
    Each window: [system] + window_turns × (user, assistant) pairs.
    Short games (fewer than window_turns pairs) are kept as one window.
    """
    system = [m for m in conv if m["role"] == "system"]
    turns  = [m for m in conv if m["role"] != "system"]

    pairs = []
    i = 0
    while i + 1 < len(turns):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i], turns[i + 1]))
            i += 2
        else:
            i += 1

    if not pairs:
        return []

    windows = []
    for start in range(0, len(pairs), window_step):
        chunk = pairs[start : start + window_turns]
        if not chunk:
            break
        window_conv = system[:]
        for user_msg, asst_msg in chunk:
            window_conv.extend([user_msg, asst_msg])
        windows.append(window_conv)

    return windows


def _clean(messages: "list[dict] | None") -> "list[dict] | None":
    if not messages:
        return None
    messages = [{"role": m["role"], "content": str(m["content"])} for m in messages]
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if not messages:
        return None
    if sum(1 for m in messages if m["role"] == "assistant") < MIN_ASSISTANT_TURNS:
        return None
    return messages


def _stats(conversations: list[list[dict]]) -> dict:
    turn_counts = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    return {
        "total": len(conversations),
        "avg_assistant_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "turn_distribution": dict(sorted(Counter(turn_counts).items())),
    }


# Per-environment generation defaults. Applied for both single and multi-env paths;
# CLI flags (--num_games, --max_turn, --score-power) override when explicitly provided.
_ENV_DEFAULT_ARGS: dict[str, dict] = {
    "gin_rummy":   {"num_games": 4000,   "max_turn": 200, "mcts_simulations": 25},
    "liars_dice":  {"num_games": 50000,  "max_turn": 30,  "mcts_simulations": 225},
    "leduc_poker": {"num_games": 200000, "max_turn": 10,  "mcts_simulations": 50,
                    "sample_by_score": True, "score_power": 3.0},
}
_DEFAULT_ARGS: dict = {"num_games": 50000, "max_turn": 30, "mcts_simulations": 10}


def merge_datasets(per_env_paths: list[str], output_path: str) -> None:
    """Concatenate per-environment DatasetDicts into one and save to output_path."""
    splits: dict[str, list] = {}
    for p in per_env_paths:
        for split_name, ds in DatasetDict.load_from_disk(p).items():
            splits.setdefault(split_name, []).append(ds)
    DatasetDict({k: concatenate_datasets(v) for k, v in splits.items()}).save_to_disk(output_path)
    _log(f"Merged {len(per_env_paths)} env datasets → {output_path}")


def generate_for_env(
    env_name: str,
    output_path: str,
    num_games: int,
    max_turn: int,
    window_turns: int = 10,
    window_step: int = 0,
    num_workers: int = 0,
    seed: int = 42,
    wins_only: bool = False,
    sample_by_score: bool = False,
    score_power: float = 1.0,
    mcts_simulations: int = 10,
    time_limit_seconds: float | None = None,
) -> None:
    """Generate and save a trajectory dataset for a single environment."""
    if window_step == 0:
        window_step = window_turns // 2 or 1

    task_id_min, task_id_max = GAMES_TO_TASK_ID_RANGE[env_name]

    reset_payload = {
        "task_id": task_id_min,
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": mcts_simulations,
        "mcts_num_rollouts": 1,
    }
    _, env_pool, num_servers, _, _ = init_env_pool(reset_payload)

    num_workers = num_workers or max(1, num_servers)

    _log(f"Environment  : {env_name}")
    _log(f"Output       : {output_path}")
    _log(f"Num games    : {num_games}")
    _log(f"Window turns : {window_turns}  step {window_step}")
    _log(f"Env servers  : {num_servers}   Workers: {num_workers}")

    random.seed(seed)
    game_ids = random.sample(range(task_id_min + 1, task_id_max), num_games)
    tasks = [
        (gid, env_pool[i % num_servers]["base_url"], max_turn)
        for i, gid in enumerate(game_ids)
    ]

    use_score_filter = wins_only or sample_by_score
    _log(f"Playing {num_games} games..." + (f" (limit {time_limit_seconds:.0f}s)" if time_limit_seconds else ""))
    if use_score_filter:
        _log(f"Score filter: wins_only={wins_only}  sample_by_score={sample_by_score}"
              f"  score_power={score_power}")
    conversations: list[list[dict]] = []
    skipped = 0
    score_filtered = 0
    all_scores: list[float] = []
    deadline = time.monotonic() + time_limit_seconds if time_limit_seconds else None
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(env_name,),
    ) as pool:
        futures = {pool.submit(_worker_play, gid, ep, mt): gid for gid, ep, mt in tasks}
        completed = 0
        for future in as_completed(futures):
            result = future.result()

            # Unpack score when the generator returns (messages, score)
            if isinstance(result, tuple):
                raw_messages, score = result
                all_scores.append(score)
            else:
                raw_messages, score = result, None

            # Apply score-based filters only when a score is available
            if score is not None and use_score_filter:
                if wins_only and score <= 0:
                    score_filtered += 1
                    completed += 1
                    continue
                if sample_by_score:
                    prob = max(0.0, min(1.0, score)) ** score_power
                    if random.random() >= prob:
                        score_filtered += 1
                        completed += 1
                        continue

            cleaned = _clean(raw_messages)
            if cleaned is None:
                skipped += 1
            else:
                conversations.append(cleaned)
            completed += 1
            if completed % 100 == 0:
                _log(f"  {completed}/{num_games} games done", flush=True)

            if deadline is not None and time.monotonic() >= deadline:
                _log(f"Time limit reached, stopping at {completed}/{num_games} games")
                for f in futures:
                    f.cancel()
                break

    score_summary = ""
    if all_scores:
        wins = sum(1 for s in all_scores if s > 0)
        score_summary = (
            f"   Score stats: min={min(all_scores):.3f}  max={max(all_scores):.3f}"
            f"  wins(>0)={wins}/{len(all_scores)} ({100*wins/len(all_scores):.1f}%)\n"
            f"   Score-filtered: {score_filtered}"
        )
    _log(f"Valid : {len(conversations)}   Skipped : {skipped}{chr(10) + score_summary if score_summary else ''}")

    if not conversations:
        raise RuntimeError("No valid conversations generated. Check ENVIRONMENT_SERVER_URLS.")

    # Raw game length stats — helps diagnose max_turn being hit or unexpectedly long games.
    raw_lengths = [sum(1 for m in c if m["role"] == "assistant") for c in conversations]
    max_turn_hits = sum(1 for l in raw_lengths if l >= max_turn)
    length_buckets = Counter(l // 10 * 10 for l in raw_lengths)
    _log(f"\n  ── Raw game stats (before windowing) ──────────────────")
    _log(f"  Avg turns/game   : {sum(raw_lengths)/len(raw_lengths):.1f}")
    _log(f"  Min/Max turns    : {min(raw_lengths)} / {max(raw_lengths)}")
    _log(f"  Hit max_turn={max_turn}  : {max_turn_hits} / {len(conversations)} games"
          f"  ({100*max_turn_hits/len(conversations):.1f}%)")
    _log(f"  Length buckets   : " +
          "  ".join(f"{k}-{k+9}:{v}" for k, v in sorted(length_buckets.items())))

    # Apply sliding window — expands long games into overlapping sub-conversations.
    # Short games (< window_turns pairs) are kept whole as a single window.
    windowed: list[list[dict]] = []
    for conv in conversations:
        windows = _sliding_windows(conv, window_turns, window_step)
        windowed.extend(windows if windows else [conv])
    conversations = windowed
    _log(f"\n  ── After windowing (turns={window_turns} step={window_step}) ──")
    _log(f"  Total examples   : {len(conversations)}")

    for k, v in _stats(conversations).items():
        _log(f"  {k}: {v}")

    dataset = Dataset.from_list([{"messages": c} for c in conversations])
    dd = DatasetDict({"train": dataset})
    _log(f"Train: {len(dd['train'])}")

    dd.save_to_disk(output_path)
    _log(f"Dataset saved → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--environment_names", nargs="+", required=True,
                   help="One or more environment names. Single entry uses --num_games / "
                        "--max_turn overrides; multiple entries use built-in per-env defaults.")
    p.add_argument("--output_path",      required=True)
    p.add_argument("--num_games",   type=int, default=None,
                   help="Override num_games (single-env only; defaults to per-env built-in).")
    p.add_argument("--max_turn",    type=int, default=None,
                   help="Override max_turn (single-env only; defaults to per-env built-in).")
    p.add_argument("--window_turns", type=int, default=10,
                   help="Split each game into sub-conversations of this many (user,assistant) "
                        "pairs. Games shorter than this are kept whole. Default 10.")
    p.add_argument("--window_step", type=int, default=0,
                   help="Slide window by this many pairs (default: window_turns // 2).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of worker processes. Default 0 = num_servers. "
                        "Each process holds one concurrent env server connection; "
                        "raise to increase throughput, lower to reduce env server load.")
    p.add_argument("--seed", type=int, default=42)
    # Score-based sampling (for generators that return (messages, score) tuples)
    p.add_argument("--wins-only", action="store_true",
                   help="Discard games where score <= 0. Only applies when the "
                        "generator returns a (messages, score) tuple.")
    p.add_argument("--sample-by-score", action="store_true",
                   help="Keep each game with probability clamp(score, 0, 1) ** score-power. "
                        "Only applies when the generator returns a (messages, score) tuple.")
    p.add_argument("--score-power", type=float, default=None,
                   help="Exponent for score-based sampling (single-env override; "
                        "defaults to per-env built-in, or 1.0 if not set).")
    p.add_argument("--time_limit_seconds", type=float, default=None,
                   help="Total generation budget in seconds. For multiple envs the budget "
                        "is divided equally. None = unlimited (generate all num_games).")
    args = p.parse_args()

    if len(args.environment_names) == 1:
        # Single env: built-in per-env defaults apply; CLI flags override when provided
        env_name = args.environment_names[0]
        env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
        generate_for_env(
            env_name, args.output_path,
            num_games=args.num_games if args.num_games is not None else env_cfg["num_games"],
            max_turn=args.max_turn if args.max_turn is not None else env_cfg["max_turn"],
            window_turns=args.window_turns,
            window_step=args.window_step,
            num_workers=args.num_workers,
            seed=args.seed,
            wins_only=args.wins_only or env_cfg.get("wins_only", False),
            sample_by_score=args.sample_by_score or env_cfg.get("sample_by_score", False),
            score_power=args.score_power if args.score_power is not None else env_cfg.get("score_power", 1.0),
            mcts_simulations=env_cfg["mcts_simulations"],
            time_limit_seconds=args.time_limit_seconds,
        )
    else:
        # Multiple envs: generate each to a staging path using built-in defaults, then merge
        per_env_limit = args.time_limit_seconds / len(args.environment_names) if args.time_limit_seconds else None
        per_env_paths = []
        for env_name in args.environment_names:
            env_cfg = {**_DEFAULT_ARGS, **_ENV_DEFAULT_ARGS.get(env_name, {})}
            env_path = f"{args.output_path}_{env_name}"
            per_env_paths.append(env_path)
            generate_for_env(
                env_name, env_path,
                num_games=env_cfg["num_games"],
                max_turn=env_cfg["max_turn"],
                window_turns=args.window_turns,
                window_step=args.window_step,
                num_workers=args.num_workers,
                seed=args.seed,
                wins_only=env_cfg.get("wins_only", False),
                sample_by_score=env_cfg.get("sample_by_score", False),
                score_power=env_cfg.get("score_power", 1.0),
                mcts_simulations=env_cfg["mcts_simulations"],
                time_limit_seconds=per_env_limit,
            )
        merge_datasets(per_env_paths, args.output_path)


if __name__ == "__main__":
    main()
