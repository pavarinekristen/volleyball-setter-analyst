"""Settings for the volleyball run. Edit here, or override per run with flags (`python main.py -h`)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Input ────────────────────────────────────────────────────────────────────
INPUT = DATA_DIR / "input"   # a clip, or a folder of clips
VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# ── Conversion ───────────────────────────────────────────────────────────────
INFERENCE_HEIGHT = 1080   # what the models see
EXPORT_HEIGHT = 1080      # the output's total height: the clip on top, the panel below, each half
TRIM_SECONDS = None       # e.g. 10.0 for a quick test
CONVERT_CRF = 23
FORCE_RECONVERT = False

# iPhone HDR looks washed out in OpenCV. "auto" tone-maps HDR clips, "off" never, "force" always.
TONEMAP = "auto"
TONEMAP_ALGORITHM = "bt.2446a"

# ── Gateway ──────────────────────────────────────────────────────────────────
GATEWAY_BASE_URL = "https://gateway.vlm.run/v1/openai"
REQUEST_TIMEOUT = 1800.0
REUSE_CACHE = True        # reuse a cached reply when the request matches

# ── Pose: ViTPose ────────────────────────────────────────────────────────────
POSE_MODEL = "usyd-community/vitpose-plus-large"
POSE_FPS = 15.0          # detector cadence; the pose still runs on every frame
PRECISION = 4             # decimals on normalized coordinates
MAX_PERSONS = 1
PERSON_IOU_THRESHOLD = 0.6

LINE_THICKNESS = 3        # px at 720 px wide; scales with the output
POINT_RADIUS = 2

SKELETON_COLORS = {       # BGR, same palette as the basketball demo
    "left_arm":  (238, 211, 34),    # cyan
    "right_arm": (241, 102, 99),    # indigo
    "left_leg":  (249, 121, 232),   # magenta
    "right_leg": (133, 113, 251),   # rose
    "torso":     (225, 213, 203),   # slate
}

# ── SAM 3.1: the net (optional) and the ball ────────────────────────────────
SAM_MODEL = "facebook/sam3.1"

# ── Net: SAM 3.1, one frame, optional ───────────────────────────────────────
# Secondary calibration source only: used to cross-check the depth read off the
# player's own height. Unlike the basketball's rim, a missed net is not fatal.
NET_PROMPT = "volleyball net"
NET_FRAME_SECONDS = None       # None = the middle of the clip
NET_HEIGHT_M = 2.43            # men's; 2.24 women's, 2.35 mixed/school — recorded in run.json,
                                # not yet used in the height math (see VOLLEYBALL_ADAPTATION_PLAN.md §3.2)
NET_WIDTH_M = 9.5              # net span including the side bands, post to post: the one net
                                # measurement actually used, as a depth cross-check (calibration.py)

# ── Ball: SAM 3.1, around each set only ──────────────────────────────────────
BALL_PROMPT = "volleyball"
BALL_DIAMETER_M = 0.21          # FIVB official, ~65-67cm circumference
BALL_BEFORE_SECONDS = 0.5       # window start, before the hands go up (contact is brief)
BALL_AFTER_SECONDS = 1.2        # window end: enough to see the ball's post-contact arc
BALL_STRIDE = 1                 # the fine pass: every frame, so the contact can't be skipped
BALL_COARSE_STRIDE = 3          # unused unless COARSE_TRACKING is on; see below
BALL_MIN_SCORE = 0.5
BALL_SIZE_TOLERANCE = 0.35
BALL_DRAW_SIZE_TOLERANCE = 0.6
BALL_EXTEND_SECONDS = 1.0
BALL_MAX_EXTENSIONS = 1

# ── Sets: finding the candidate windows (layer 1, loose) ────────────────────
# A candidate: both wrists above this line, however briefly. Looser than the
# basketball's "above the head": a set's contact can be at or just above the
# forehead, not necessarily above the crown.
SET_ARMED_ABOVE = "nose"        # "nose", "shoulder", or "eye"
SET_SMOOTH_FRAMES = 3            # shorter median window than basketball's 5: the event is brief
SET_MIN_GAP_SECONDS = 0.6        # closer than this is the same contact, not two
SET_WINDOW_PAD_SECONDS = 0.5     # padding added each side of a raw crossing, before merging

# ── Sets: finding the contact instant (layer 2, fine) ────────────────────────
CONTACT_PROXIMITY_BALLS = 1.5    # ball must be within this many diameters of a wrist at contact
CONTACT_MAX_HOLD_FRAMES = 4      # longer than this near-zero-velocity near the hands looks like a carry
CONTACT_MIN_CONFIDENCE = 0.4     # sets scoring below this are flagged, not discarded
ARM_ELBOW_MIN_PROJECTED = 0.5    # skip angle frames where an arm segment points at the camera

# ── Height and consistency ───────────────────────────────────────────────────
SIDE_ON_MAX_SHOULDER_RATIO = 0.55   # shoulder width / torso: side-on vs. facing the camera
JUMP_THRESHOLD_M = 0.05             # hip/ankle rise over the standing baseline that counts as a jump
IDEAL_REACH_WINDOW_SECONDS = None   # None = the whole session; else only the last N seconds searched
HEAD_REF_WINDOW_SECONDS = 0.15      # how far back from the rise the "address" head position is sampled

# ── Court: an estimate from one camera ───────────────────────────────────────
CAMERA_HFOV_DEGREES = 70.0    # iPhone 1x lens, landscape
PLAYER_HEIGHT_M = 1.75        # scales every distance; overridden per run with --player-height

# ── Video ────────────────────────────────────────────────────────────────────
BALL_TRAIL_SECONDS = 0.6      # extends a bit past the basketball's 0.5s to show the approach too
BALL_FILL_SECONDS = 0.33
MARKER_SECONDS = 1.0          # how long the contact marker stays lit on the video
PANEL_FONT = "auto"
PANEL_FONT_INDEX = None
OUTPUT_CRF = 20

# Deficit, in meters, treated as "about right": within this of ideal reads green.
HEIGHT_DEFICIT_GOOD_M = 0.10
HEIGHT_DEFICIT_BAD_M = 0.35       # at or beyond this reads fully red

# Credit, bottom-right: ("LABEL", "value") or a string. Or pass --li, --x, --ig.
CREDIT = []
CREDIT_SIZE = 20
CREDIT_FOOTER = 52
CREDIT_SEPARATION = 28
CREDIT_MARGIN = 22
CREDIT_COLOR = (180, 180, 180)
CREDIT_LABEL_COLOR = (120, 120, 120)

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
