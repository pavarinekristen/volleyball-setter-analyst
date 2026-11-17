# volleyball-setter-analyst

Tracks a setter's own training session from a phone that stays still. It finds
each set from the ball's own path — not the pose alone, since a set, a block and
an attack all raise the hands the same way — and measures the contact height
against the setter's own best reach, plus how consistent the contact point and
the trajectory are from one repetition to the next.

Adapted from [jeremyipark/vision-demos (basketball)](https://github.com/jeremyipark/vision-demos/tree/main/basketball),
which calls makes and misses against a hoop. A set has no target to judge against,
so this measures *consistency* instead: same spot, same height, same trajectory,
or not. The full design — what's reused, what's new, and the open questions — is
in [VOLLEYBALL_ADAPTATION_PLAN.md](VOLLEYBALL_ADAPTATION_PLAN.md).

Pose runs on [`vitpose-plus-large`](https://vlm.run/gateway/models/usyd-community-vitpose-plus-large),
the net and ball on [`sam3.1`](https://vlm.run/gateway/models/facebook-sam3.1), both
through the [VLM Run Gateway](https://www.vlm.run/gateway) — a paid, per-call API.

## Status

Early adaptation, built from the design doc but not yet validated against real
footage end to end. The numeric thresholds (how close the ball must come to the
wrist to count as a contact, how loose the pose gate is, etc., all in
`config.py`) are reasonable defaults, not tuned against real sets. Expect to
adjust them once you have real clips to test with.

What the pipeline needs to work well: a phone held still, close enough that the
setter and the ball are a meaningful part of the frame — like the basketball
demo's own recommended setup. Wide shots of a full match from the stands (a
broadcast angle) put the ball too few pixels across for the ball-tracking model
to find it reliably; that's a resolution/distance problem, not a tunable one.

## Run it

1. Get an API key at [app.vlm.run](https://app.vlm.run/sign-in) and put it in a
   `.env` at this repo's root, or any parent directory:

   ```bash
   echo "VLMRUN_API_KEY=..." > .env
   ```

2. Create the environment (`environment.yml`), or just install
   `numpy opencv-python rich matplotlib openai python-dotenv` plus `ffmpeg` on
   your PATH.

3. Put a clip in `data/input/`: landscape, the phone set down and still, close
   enough to the setter that a set's contact is a sizeable part of the frame.

4. Run it:

   ```bash
   python main.py                         # every clip in data/input/
   python main.py data/input/session.mov  # one clip
   python main.py -h                      # every flag
   ```

Each run costs real money against the gateway — every call is billed, and the
cost is printed at the end of each run. Replies are cached in `data/cache/`, so
re-running the same clip doesn't call the models again.

## Output

Each run writes to `data/output/<timestamp>-set-consistency/`:

| file | what |
|---|---|
| `<clip>_volleyball.mp4` | the clip on top, a consistency panel below |
| `plots.png` | every set's ball path aligned on its contact, and the contact point scattered around the setter's own head |
| `sequence.png` | contact height and sideways drift, set by set, in order |
| `arm.png` | each arm's elbow angle through every set, aligned on contact |
| `<clip>_sets.csv`, `<clip>_frames.csv` | one row per set, one row per frame |
| `sets.json`, `session_consistency.json`, `run.json` | per-set results with ball paths; the session's dispersion stats; settings and cost |

## License

Not yet chosen.
