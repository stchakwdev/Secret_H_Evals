# Running Experiments

This tutorial covers single game execution, batch experiments, and parallel processing.

## Single Game Execution

### Basic Game

Run a single 5-player game with default settings:

```bash
python run_game.py --players 5
```

### With Database Logging

Enable persistent storage for later analysis:

```bash
python run_game.py --players 7 --enable-db-logging
```

### With Specific Model

```bash
python run_game.py --players 5 --model anthropic/claude-3.5-sonnet
```

## Batch Experiments

### Quick Verification Batch

Test configuration with a small batch:

```bash
python run_game.py --batch --games 5 --players 5 \
  --batch-tag "verification" \
  --batch-id "verify-$(date +%Y%m%d-%H%M)"
```

### Production Batch

Run a full experiment with 100+ games:

```bash
python run_game.py --batch --games 100 --players 7 \
  --enable-db-logging \
  --batch-tag "production-deepseek-v3" \
  --model deepseek/deepseek-v3.2-exp
```

### Model Comparison Batch

Compare different models:

```bash
# Run same experiment with different models
for model in "deepseek/deepseek-v3.2-exp" "anthropic/claude-3.5-sonnet"; do
  python run_game.py --batch --games 50 --players 7 \
    --enable-db-logging \
    --batch-tag "comparison-$(echo $model | tr '/' '-')" \
    --model "$model"
done
```

## Monitoring Progress

### Real-Time Tracking

Watch batch progress with live updates:

```bash
python check_batch_progress.py --watch
```

### Manual Check

Check progress at a specific point:

```bash
python check_batch_progress.py
```

### With Custom Interval

```bash
python check_batch_progress.py --watch --interval 10
```

## Batch Metadata

Each batch creates a metadata file at `logs/.current_batch`:

```json
{
  "batch_id": "batch-20251101-143000-a1b2c3d4",
  "batch_tag": "production-deepseek-v3",
  "start_time": "2025-11-01 14:30:00",
  "target_games": 100,
  "players": 7,
  "model": "deepseek/deepseek-v3.2-exp"
}
```

## Output Locations

| Output Type | Location |
|-------------|----------|
| Game logs | `logs/<game-uuid>/game.log` |
| Batch metadata | `logs/.current_batch` |
| SQLite database | `data/games.db` |
| Visualizations | `visualizations/` |
| Reports | `reports/` |

## Best Practices

1. **Always verify first**: Run a 5-game verification batch before production runs
2. **Enable database logging**: Use `--enable-db-logging` for any analysis
3. **Use descriptive tags**: `--batch-tag` helps identify experiments later
4. **Monitor costs**: Watch API usage during initial experiments
5. **Document configuration**: Record model, player count, and parameters

## Troubleshooting

### Games Not Progressing

Check logs for JSON parsing errors:

```bash
grep -i "parse" logs/*/game.log | tail -20
```

### High API Costs

Use cost-effective models for initial testing:

```bash
python run_game.py --model deepseek/deepseek-v3.2-exp
```

### Database Not Found

Ensure `data/` directory exists:

```bash
mkdir -p data
```

## Next Steps

- [Analyzing Results](analyzing-results.md) - How to analyze your data
- [Custom Models](custom-models.md) - Adding new LLM models
