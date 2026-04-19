/*
   NEURO-OP — script.js
   Neural Criticality CA Game — Fraile et al. (2018) GoL Model

   MODEL: Fraile GoL — Conway's Game of Life + noise + defects
   Based on: "Cellular Automata and Artificial Brain Dynamics"
   Fraile et al., Math. Comput. Appl. 2018, 23, 75

   STATES:
     0 : INACTIVE  — dead / resting neuron
     1 : ACTIVE    — alive / firing neuron
    -1 : DEFECT    — permanently dead, blocks propagation (Fraile Section 5)

   RULES (GoL B3/S23 + noise):
     INACTIVE → ACTIVE  if exactly T active Moore-neighbours (default T=3)
                         OR with probability p (spontaneous noise)
     ACTIVE   → ACTIVE  if 2 or 3 active neighbours (survival)
     ACTIVE   → INACTIVE otherwise (under/over-population)
     DEFECT   → DEFECT  always (permanent, not counted as neighbour)

   Open boundary conditions (matching the paper).

   PLAYER CONTROLS:
     noise     : spontaneous firing probability p  [slider 0-100 → 0-3%]
     threshold : birth rule neighbour count T      [slider 1-6, default 3]
     defects   : % of permanently dead cells       [slider 0-100 → 0-10%]
     click/drag: manual stimulation
*/


/* 1. CONSTANTS & CONFIGURATION */

// The three possible states a cell can be in
const STATE_INACTIVE = 0;
const STATE_ACTIVE   = 1;
const STATE_DEFECT   = -1;

// Colors for each cell state (drawn on the grid canvas)
// These match the legend colors in index.html
const CELL_COLORS = {
  [STATE_INACTIVE]: '#030e1e',   // dark navy — quiet neuron
  [STATE_ACTIVE]:   '#4db8ff',   // bright cyan — firing neuron
  [STATE_DEFECT]:   '#ff3355',   // red — permanent defect
};
// Glow effect around active cells
const ACTIVE_GLOW_COLOR = 'rgba(77, 184, 255, 0.3)';

// Activity thresholds — tuned via GoL+noise experiments on a 60x60 grid.
// Pure GoL equilibrium sits around 3-5%. Below that, no structures survive.
// GoL-dominated dynamics with spatial structure live around 5-18%.
// Above ~25%, noise overwhelms GoL structure; temporal autocorrelation drops below 0.5.
const TARGET_LOW  = 0.04;   // lower edge of healthy zone (just above GoL equilibrium)
const TARGET_HIGH = 0.18;   // upper edge (GoL still dominates births, spatial order intact)
const DANGER_LOW  = 0.025;  // flatline danger (below pure GoL equilibrium ~3%)
const DANGER_HIGH = 0.28;   // seizure danger (noise-dominated, structure collapsing)

// How long the player can stay in the danger zone before dying (seconds)
const DANGER_SECONDS = 8;
// Health drains at this rate per second when in the danger zone
const HEALTH_DRAIN_RATE = 100 / DANGER_SECONDS;

// Grid size — fixed at 60x60 for richer GoL dynamics than smaller grids
const FIXED_GRID_SIZE = 60;
// 20% of cells start active (matches Fraile's experimental setup)
const INIT_ACTIVE_FRAC = 0.20;

// How many data points to keep for the activity-over-time graph
const HISTORY_LEN = 200;

// Decorative brain outline drawn over the grid (purely visual)
let brainOutlinePath = null;


/* 2. GAME STATE */

// The grid is a 2D array. Each cell is 0 (inactive), 1 (active), or -1 (defect).
let grid     = [];
let nextGrid = [];  // double-buffer: we compute the next state here, then swap
let gridSize = FIXED_GRID_SIZE;

let running    = false;  // is the simulation stepping forward?
let gameActive = false;  // is a game in progress? (false = intro or game-over screen)
let lastTime   = 0;      // timestamp of last animation frame
let stepAccum  = 0;      // accumulates time to know when to run the next sim step

// Player-adjustable parameters (default values)
let simSpeed  = 8;       // how many simulation steps per second
let noise     = 25;      // spontaneous firing slider (0-100, maps to probability 0-3%)
let threshold = 3;       // GoL birth neighbour count T (standard GoL = 3)
let damping   = 0;       // defect % slider (0-100, maps to 0-10% of grid)

// Stable baseline used whenever a run is restarted from game-over.
const STABLE_START = {
  simSpeed: 8,
  noise: 25,
  threshold: 3,
  damping: 0,
};

let health      = 100;   // drains in danger zone, game ends at 0
let elapsed     = 0;     // seconds since game started
let dangerTimer = 0;     // how long the player has been in the danger zone

let activityHistory = [];  // array of activity fractions for the graph
let brainState      = 'stable';  // current state: 'stable', 'warning', 'underactive', 'overloaded'
let lastBrainState  = '';        // previous state (used to detect changes for logging)

let warningOverlayActive = false;
const WARNING_OVERLAY_DURATION = 3.0;  // seconds the warning banner stays visible

// Crisis Event System — random crises that mess with the player's sliders
let eventTimer       = 0;        // seconds until next event
let nextEventAt      = 18;       // first event fires after this many seconds
let activeEvent      = null;     // the current crisis event object, or null
let traumaDefects    = [];       // list of {r,c} cells turned into temporary defects by trauma
let traumaDefectSet  = new Set(); // "r,c" string keys for fast lookup during click-to-repair
let traumaRepairTotal = 0;       // how many cells the trauma originally destroyed
let lastCrisisType   = null;     // 'trauma', 'seizure_spike', or 'sedation' — used in game-over text
let lastCrisisRegion = null;     // brain region name for trauma events (e.g. 'FRONTAL LOBE')

// Canvas references (set in init())
let gridCanvas, gridCtx;
let graphCanvas, graphCtx;
let cellSize    = 0;     // pixel size of each cell on the canvas
let isMouseDown = false; // whether the player is currently dragging on the grid


/* 3. SIMULATION — Fraile GoL + Noise + Defects */

/**
 * Build a decorative brain-shaped outline (Path2D) to overlay on the grid.
 * This is purely cosmetic — it doesn't affect the simulation.
 */
function buildBrainOutline(W, H) {
  // Convert normalized coordinates (0-1) to pixel coordinates
  const x = v => v * W;
  const y = v => v * H;
  const path = new Path2D();
  path.moveTo(x(0.50), y(0.90));
  path.bezierCurveTo(x(0.60), y(0.90), x(0.88), y(0.80), x(0.90), y(0.62));
  path.bezierCurveTo(x(0.96), y(0.46), x(0.96), y(0.30), x(0.82), y(0.18));
  path.bezierCurveTo(x(0.72), y(0.08), x(0.60), y(0.06), x(0.50), y(0.08));
  path.bezierCurveTo(x(0.40), y(0.06), x(0.28), y(0.08), x(0.18), y(0.18));
  path.bezierCurveTo(x(0.04), y(0.30), x(0.04), y(0.46), x(0.10), y(0.62));
  path.bezierCurveTo(x(0.12), y(0.80), x(0.40), y(0.90), x(0.50), y(0.90));
  path.closePath();
  brainOutlinePath = path;
}

/**
 * Initialize the grid for a new game.
 * Seeds INIT_ACTIVE_FRAC (20%) of cells as ACTIVE, then scatters defects
 * based on the current defect slider value.
 */
function initGrid(size = FIXED_GRID_SIZE) {
  gridSize = size;
  grid     = [];
  nextGrid = [];

  // Create empty grids using Int8Array (supports -1 for defects)
  for (let r = 0; r < gridSize; r++) {
    grid[r]     = new Int8Array(gridSize);
    nextGrid[r] = new Int8Array(gridSize);
  }

  // Scatter defects first (permanently dead cells, as described in Fraile Section 5).
  // Fraile's experiments show: at p=0.005, 2% defects halves density.
  // At 5% defects, activity nearly collapses.
  // Slider 0-100 maps to 0-10% defect fraction.
  const defectFrac = (damping / 100) * 0.10;
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      if (Math.random() < defectFrac) {
        grid[r][c] = STATE_DEFECT;
      }
    }
  }

  // Seed active cells randomly (skip cells already marked as defects)
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      if (grid[r][c] === STATE_DEFECT) continue;
      grid[r][c] = Math.random() < INIT_ACTIVE_FRAC ? STATE_ACTIVE : STATE_INACTIVE;
    }
  }
}

/**
 * Count how many ACTIVE neighbours surround a given cell.
 * Uses the 8-cell Moore neighborhood (all adjacent cells including diagonals).
 * Open boundary conditions: cells outside the grid are treated as inactive.
 * Defect cells (-1) are NOT counted as neighbours.
 */
function countActiveNeighbors(row, col) {
  let count = 0;
  // Check all 8 directions around (row, col)
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;  // skip the cell itself
      const nr = row + dr;
      const nc = col + dc;
      // Open boundary: out-of-bounds cells are treated as inactive (Fraile's approach)
      if (nr < 0 || nr >= gridSize || nc < 0 || nc >= gridSize) continue;
      if (grid[nr][nc] === STATE_ACTIVE) count++;
    }
  }
  return count;
}

/**
 * Advance the simulation by one step using Fraile GoL rules + noise.
 *
 * RULES (Conway's GoL B3/S23 with noise extension):
 *   INACTIVE → ACTIVE   if (exactly T active neighbours)     [GoL birth]
 *                        OR (random() < p)                    [noise/spontaneous firing]
 *   ACTIVE   → ACTIVE   if (2 or 3 active neighbours)        [GoL survival]
 *   ACTIVE   → INACTIVE otherwise                             [GoL death: under/over-population]
 *   DEFECT   → DEFECT   always                                [permanent damage]
 */
function stepSimulation() {
  // Convert slider value to actual probability: slider 0-100 → p = 0-3%
  const p = (noise / 100) * 0.03;
  const T = threshold;
  let activeCount = 0;

  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const state = grid[r][c];

      if (state === STATE_DEFECT) {
        // DEFECT: permanently dead, never changes
        nextGrid[r][c] = STATE_DEFECT;

      } else if (state === STATE_INACTIVE) {
        // INACTIVE: check if it should activate via GoL birth rule or random noise
        const n = countActiveNeighbors(r, c);
        if (n === T || Math.random() < p) {
          nextGrid[r][c] = STATE_ACTIVE;
          activeCount++;
        } else {
          nextGrid[r][c] = STATE_INACTIVE;
        }

      } else {
        // ACTIVE: survives only with exactly 2 or 3 active neighbours (standard GoL)
        const n = countActiveNeighbors(r, c);
        if (n === 2 || n === 3) {
          nextGrid[r][c] = STATE_ACTIVE;
          activeCount++;
        } else {
          nextGrid[r][c] = STATE_INACTIVE;  // dies from under- or over-population
        }
      }
    }
  }

  // Swap the two grid buffers (avoids allocating new arrays every step)
  const tmp = grid;
  grid      = nextGrid;
  nextGrid  = tmp;

  // Record this step's activity for the graph
  // Defects are counted as inactive, which drags the activity % down
  const total = gridSize * gridSize;
  const fraction = total > 0 ? activeCount / total : 0;
  activityHistory.push(fraction);
  if (activityHistory.length > HISTORY_LEN) activityHistory.shift();
}

/** Count total defect cells in the grid. */
function countDefects() {
  let count = 0;
  for (let r = 0; r < gridSize; r++)
    for (let c = 0; c < gridSize; c++)
      if (grid[r][c] === STATE_DEFECT) count++;
  return count;
}

/**
 * Activate a circular region of cells — this is what happens when the player
 * clicks or drags on the grid. Also repairs any trauma defects within the radius
 * (converts them back to active cells so they can participate in GoL again).
 */
function stimulateRegion(row, col, radius) {
  let repaired = 0;
  for (let dr = -radius; dr <= radius; dr++) {
    for (let dc = -radius; dc <= radius; dc++) {
      // Only affect cells within a circular radius (not a square)
      if (dr * dr + dc * dc > radius * radius) continue;
      const r = row + dr;
      const c = col + dc;
      if (r < 0 || r >= gridSize || c < 0 || c >= gridSize) continue;

      const key = r + ',' + c;
      if (grid[r][c] === STATE_DEFECT && traumaDefectSet.has(key)) {
        // This is a trauma defect — repair it by making it active
        grid[r][c] = STATE_ACTIVE;
        traumaDefectSet.delete(key);
        repaired++;
      } else if (grid[r][c] === STATE_INACTIVE) {
        // Normal inactive cell — activate it
        grid[r][c] = STATE_ACTIVE;
      }
    }
  }

  // Log repair progress if the player is repairing trauma damage
  if (repaired > 0 && traumaDefects.length > 0) {
    const stillDefect = traumaDefectSet.size;
    if (stillDefect > 0) {
      addLog('Repaired ' + repaired + ' cells — ' + stillDefect + ' remaining', 'warn-entry');
    }
  }
}

/**
 * Regenerate defects when the defect slider changes mid-game.
 * Removes all existing defects first, then places new ones randomly.
 */
function regenerateDefects() {
  const defectFrac = (damping / 100) * 0.10;
  // Clear all existing defects
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      if (grid[r][c] === STATE_DEFECT) {
        grid[r][c] = STATE_INACTIVE;
      }
    }
  }
  // Place new defects randomly
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      if (grid[r][c] === STATE_INACTIVE && Math.random() < defectFrac) {
        grid[r][c] = STATE_DEFECT;
      }
    }
  }
}


/* 4. RENDERING */

/** Resize the grid canvas to fit the available space. */
function resizeGridCanvas() {
  const wrapper    = document.getElementById('grid-wrapper');
  const maxW       = wrapper.clientWidth || 500;
  const maxH       = (window.innerHeight - 130) * 0.82;
  const available  = Math.min(maxW, maxH);
  cellSize         = Math.floor(available / gridSize);
  const canvasSize = cellSize * gridSize;
  gridCanvas.width  = canvasSize;
  gridCanvas.height = canvasSize;
  buildBrainOutline(canvasSize, canvasSize);
}

/** Draw the simulation grid — called every animation frame. */
function renderGrid() {
  // Fill background with dark navy (same as STATE_INACTIVE color)
  gridCtx.fillStyle = '#030e1e';
  gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

  // 1px gap between cells if they're large enough to see it
  const gap = cellSize > 4 ? 1 : 0;

  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const state = grid[r][c];
      if (state === STATE_INACTIVE) continue;  // background is already the inactive color
      const x = c * cellSize;
      const y = r * cellSize;

      if (state === STATE_DEFECT) {
        // Defect cells: dim red (low opacity so they don't dominate visually)
        gridCtx.fillStyle = 'rgba(255, 51, 85, 0.25)';
        gridCtx.fillRect(x + gap, y + gap, cellSize - gap, cellSize - gap);
      } else {
        // Active cells: bright cyan with a subtle glow effect
        gridCtx.fillStyle = CELL_COLORS[STATE_ACTIVE];
        gridCtx.fillRect(x + gap, y + gap, cellSize - gap, cellSize - gap);
        if (cellSize > 3) {
          // Draw glow slightly larger than the cell, then redraw the cell on top
          gridCtx.fillStyle = ACTIVE_GLOW_COLOR;
          gridCtx.fillRect(x - 1, y - 1, cellSize + 2, cellSize + 2);
          gridCtx.fillStyle = CELL_COLORS[STATE_ACTIVE];
          gridCtx.fillRect(x + gap, y + gap, cellSize - gap, cellSize - gap);
        }
      }
    }
  }

  // Draw the decorative brain outline on top of the grid
  if (brainOutlinePath) {
    gridCtx.save();
    // Three strokes at different widths create a soft glowing outline effect
    gridCtx.strokeStyle = 'rgba(0, 140, 255, 0.12)';
    gridCtx.lineWidth   = 18;
    gridCtx.stroke(brainOutlinePath);
    gridCtx.strokeStyle = 'rgba(77, 184, 255, 0.22)';
    gridCtx.lineWidth   = 8;
    gridCtx.stroke(brainOutlinePath);
    gridCtx.strokeStyle = 'rgba(77, 184, 255, 0.55)';
    gridCtx.lineWidth   = 1.5;
    gridCtx.stroke(brainOutlinePath);
    gridCtx.restore();
  }
}

/** Show a circular pulse animation where the player clicked. */
function showPulse(px, py) {
  const pulse = document.createElement('div');
  pulse.className  = 'pulse-effect';
  pulse.style.left = px + 'px';
  pulse.style.top  = py + 'px';
  document.getElementById('grid-wrapper').appendChild(pulse);
  setTimeout(() => pulse.remove(), 450);
}


/* 5. ACTIVITY GRAPH — draws the EEG-style trace of activity over time */

function renderGraph() {
  const w = graphCanvas.width;
  const h = graphCanvas.height;
  graphCtx.clearRect(0, 0, w, h);
  graphCtx.fillStyle = '#0d1220';
  graphCtx.fillRect(0, 0, w, h);

  if (activityHistory.length < 2) return;

  const len        = activityHistory.length;
  const xStep      = w / HISTORY_LEN;
  const GRAPH_MAX  = 0.50;  // the graph's y-axis goes from 0% to 50%
  const yScale     = v => h * (1 - v / GRAPH_MAX);  // convert fraction to pixel y

  // Draw the target zone as a subtle shaded band
  graphCtx.fillStyle = 'rgba(77,184,255,0.05)';
  graphCtx.fillRect(0, yScale(TARGET_HIGH), w, yScale(TARGET_LOW) - yScale(TARGET_HIGH));

  // Draw dashed lines at the target zone boundaries
  graphCtx.strokeStyle = 'rgba(77,184,255,0.2)';
  graphCtx.lineWidth   = 1;
  graphCtx.setLineDash([4, 4]);
  [TARGET_HIGH, TARGET_LOW].forEach(v => {
    graphCtx.beginPath();
    graphCtx.moveTo(0, yScale(v));
    graphCtx.lineTo(w, yScale(v));
    graphCtx.stroke();
  });
  graphCtx.setLineDash([]);

  // Draw a faint dashed line at the seizure danger threshold
  graphCtx.strokeStyle = 'rgba(255,51,85,0.25)';
  graphCtx.lineWidth   = 1;
  graphCtx.setLineDash([2, 4]);
  graphCtx.beginPath();
  graphCtx.moveTo(0, yScale(DANGER_HIGH));
  graphCtx.lineTo(w, yScale(DANGER_HIGH));
  graphCtx.stroke();
  graphCtx.setLineDash([]);

  // Draw the activity line itself
  const startX = (HISTORY_LEN - len) * xStep;
  graphCtx.beginPath();
  graphCtx.strokeStyle = '#4db8ff';  // same cyan as active cells
  graphCtx.lineWidth   = 1.5;
  for (let i = 0; i < len; i++) {
    const x = startX + i * xStep;
    const y = yScale(Math.min(activityHistory[i], GRAPH_MAX));
    if (i === 0) graphCtx.moveTo(x, y);
    else         graphCtx.lineTo(x, y);
  }
  graphCtx.stroke();

  // Fill under the line with a very subtle cyan
  graphCtx.lineTo(startX + (len - 1) * xStep, h);
  graphCtx.lineTo(startX, h);
  graphCtx.closePath();
  graphCtx.fillStyle = 'rgba(77,184,255,0.08)';
  graphCtx.fill();
}

function resizeGraphCanvas() {
  const section      = document.getElementById('graph-section');
  graphCanvas.width  = section.clientWidth || 400;
  graphCanvas.height = 65;
}


/* 6. UI / METRICS — update all the numbers and status indicators */

/**
 * Recalculate activity percentage, update all UI elements, and determine
 * the current brain state. Called once per frame.
 */
function updateMetrics(dt) {
  let activeCount = 0;
  let defectCount = 0;
  const total     = gridSize * gridSize;

  for (let r = 0; r < gridSize; r++)
    for (let c = 0; c < gridSize; c++) {
      if (grid[r][c] === STATE_ACTIVE) activeCount++;
      else if (grid[r][c] === STATE_DEFECT) defectCount++;
    }

  // Activity fraction: active cells / total cells (defects drag this down)
  const activeFrac = total > 0 ? activeCount / total : 0;
  const actPct     = Math.min(100, activeFrac * 100);
  // Bar percentage: the activity bar's max represents 50% activity
  const barPct     = Math.min(100, activeFrac / 0.50 * 100);

  // Determine brain state based on activity thresholds
  if      (activeFrac < DANGER_LOW)  brainState = 'underactive';
  else if (activeFrac > DANGER_HIGH) brainState = 'overloaded';
  else if (activeFrac < TARGET_LOW || activeFrac > TARGET_HIGH) brainState = 'warning';
  else    brainState = 'stable';

  // Update all the UI elements
  document.getElementById('activity-bar').style.width = barPct + '%';
  document.getElementById('activity-pct').textContent  = actPct.toFixed(1) + '%';
  document.getElementById('health-bar').style.width    = health + '%';
  document.getElementById('health-val').textContent    = Math.round(health);
  document.getElementById('time-display').textContent  = Math.floor(elapsed) + 's';
  document.getElementById('m-active').textContent      = actPct.toFixed(1) + '%';

  // Update the state badge (colored label in the left panel)
  const badge = document.getElementById('state-badge');
  const badgeText = {
    stable:      '● STABLE',
    underactive: '▼ UNDERACTIVE',
    overloaded:  '▲ OVERLOADED',
    warning:     '◆ WARNING',
  };
  badge.className   = 'state-badge ' + brainState;
  badge.textContent = badgeText[brainState] || '● STABLE';

  // Log state transitions (only when the state actually changes)
  if (brainState !== lastBrainState) {
    const msgs = {
      stable:      { text: 'Brain reached stable zone',       cls: 'good-entry' },
      underactive: { text: 'WARNING: Underactivity detected', cls: 'warn-entry' },
      overloaded:  { text: 'WARNING: Overload detected!',     cls: 'bad-entry'  },
      warning:     { text: 'Activity drifting out of range',  cls: 'warn-entry' },
    };
    if (msgs[brainState]) addLog(msgs[brainState].text, msgs[brainState].cls);
    lastBrainState = brainState;
  }

  return { activeFrac };
}

/** Add a timestamped entry to the operative log (right panel). */
function addLog(text, cls = '') {
  const log   = document.getElementById('event-log');
  const entry = document.createElement('div');
  entry.className = 'log-entry ' + cls;
  const timeSpan = document.createElement('span');
  timeSpan.className   = 'log-time';
  timeSpan.textContent = Math.floor(elapsed) + 's';
  entry.appendChild(timeSpan);
  entry.appendChild(document.createTextNode(text));
  log.prepend(entry);
  // Keep only the 20 most recent entries
  while (log.children.length > 20) log.removeChild(log.lastChild);
}

/** Show the full-screen warning overlay (flatline or seizure). */
function showWarningOverlay(state) {
  const overlay = document.getElementById('warning-overlay');
  if (!overlay) return;
  const title = document.getElementById('warning-overlay-title');
  const sub   = document.getElementById('warning-overlay-sub');
  const bar   = document.getElementById('warning-countdown-bar');

  if (state === 'underactive') {
    title.textContent = '⚠ FLATLINE IMMINENT';
    sub.textContent   = 'Activity below equilibrium — increase noise or stimulate directly';
    overlay.className = 'warning-overlay show flatline';
  } else {
    title.textContent = '⚠ SEIZURE CASCADE';
    sub.textContent   = 'Noise overwhelming network structure — lower noise or raise threshold';
    overlay.className = 'warning-overlay show seizure';
  }
  // Animate the countdown bar from 100% to 0% over WARNING_OVERLAY_DURATION seconds
  bar.style.transition = 'none';
  bar.style.width      = '100%';
  bar.offsetWidth;  // force browser to apply the above before starting transition
  bar.style.transition = `width ${WARNING_OVERLAY_DURATION}s linear`;
  bar.style.width      = '0%';
}

function hideWarningOverlay() {
  const overlay = document.getElementById('warning-overlay');
  if (overlay) overlay.className = 'warning-overlay';
}

/** Update the brain icon's color and animation speed to match the current state. */
function updateBrainIcon() {
  const icon = document.getElementById('brain-state-icon');
  if (!icon) return;
  // Colors and animation classes for each brain state
  // (reference image used bioluminescent cyan for healthy, red for overload)
  const colors = {
    stable:      { fill: '#00cfff', glow: 'rgba(0,207,255,0.6)',  anim: 'brain-pulse-stable' },
    warning:     { fill: '#e8a020', glow: 'rgba(232,160,32,0.6)', anim: 'brain-pulse-warn'   },
    underactive: { fill: '#2060c0', glow: 'rgba(32,96,192,0.5)',  anim: 'brain-pulse-slow'   },
    overloaded:  { fill: '#ff3355', glow: 'rgba(255,51,85,0.7)',  anim: 'brain-pulse-fast'   },
  };
  const c = colors[brainState] || colors.stable;
  icon.style.filter = `drop-shadow(0 0 10px ${c.glow})`;
  icon.querySelectorAll('path, ellipse, rect').forEach(p => p.style.fill = c.fill);
  icon.className = 'brain-icon-svg ' + c.anim;
  const stateText = document.getElementById('brain-icon-state-text');
  if (stateText) {
    const labels = { stable: 'STABLE', warning: 'WARNING', underactive: 'FLATLINE RISK', overloaded: 'OVERLOAD' };
    stateText.textContent = labels[brainState] || 'STABLE';
    stateText.style.color = c.fill;
  }
}

/** Update the small "p=X%" display next to the noise slider. */
function updateNoiseProbDisplay() {
  const el = document.getElementById('noise-prob');
  if (el) el.textContent = 'p=' + ((noise / 100) * 3).toFixed(1) + '%';
}


/* 7. HEALTH + LOSE CONDITION */

/**
 * Update health based on the current brain state.
 * In the stable zone: health regenerates.
 * In the warning zone: health drains slowly.
 * In the danger zone: health drains fast. Game over at 0.
 */
function updateHealthAndSurvival(dt) {
  // Don't let the health warning overlay hide a crisis announcement
  const crisisAnnouncing = activeEvent && activeEvent.phase === 'announcing';

  if (brainState === 'stable') {
    health       = Math.min(100, health + 8 * dt);  // heal 8 HP/sec in stable zone
    dangerTimer  = 0;
    warningOverlayActive = false;
    if (!crisisAnnouncing) hideWarningOverlay();

  } else if (brainState === 'warning') {
    health       = Math.max(0, health - 4 * dt);  // slow drain
    dangerTimer += dt;
    warningOverlayActive = false;
    if (!crisisAnnouncing) hideWarningOverlay();

  } else {
    // Danger zone (underactive or overloaded): fast health drain
    health      = Math.max(0, health - HEALTH_DRAIN_RATE * dt);
    dangerTimer += dt;

    // Show the warning overlay when health gets critically low
    if (!warningOverlayActive && !crisisAnnouncing && health <= 30) {
      warningOverlayActive = true;
      showWarningOverlay(brainState);
    }
  }

  // Game over when health reaches 0
  if (health <= 0) triggerGameOver();
}

/** End the game and show the game-over modal with explanation. */
function triggerGameOver() {
  running    = false;
  gameActive = false;
  hideWarningOverlay();

  // Icons, titles, and short reasons for each death type
  const icons = { underactive: '💤', overloaded: '⚡', warning: '⚠', stable: '✚' };
  const titles = {
    underactive: 'NEURAL FLATLINE',
    overloaded:  'CASCADE FAILURE',
    warning:     'CRITICAL DESTABILIZATION',
    stable:      'PROCEDURE TERMINATED',
  };
  const reasons = {
    underactive: 'Neural activity fell below equilibrium (~3%). Noise too low to sustain propagation, or defects blocked signal paths.',
    overloaded:  'Activity exceeded 28% — noise overwhelmed spatial structure. Reduce noise or raise firing threshold.',
    warning:     'Sustained drift outside the critical band exhausted brain integrity.',
    stable:      'Unexpected termination.',
  };

  // Get a detailed biological explanation based on what killed the player
  // and which crisis event (if any) was active
  const bioText = getBioExplanation(brainState, lastCrisisType, lastCrisisRegion);

  document.getElementById('gameover-icon').textContent   = icons[brainState]   || '✚';
  document.getElementById('gameover-title').textContent  = titles[brainState]  || titles.stable;
  document.getElementById('gameover-reason').textContent = reasons[brainState] || reasons.stable;
  document.getElementById('gameover-bio').textContent    = bioText;
  document.getElementById('final-time').textContent      = Math.floor(elapsed) + 's';

  document.getElementById('gameover-modal').classList.remove('hidden');
  drawFlatlineCanvas();
  addLog('PROCEDURE TERMINATED', 'bad-entry');
}

/**
 * Generate a biological explanation of why the player died.
 * The explanation varies depending on:
 *   - deathState: how the player died ('underactive', 'overloaded', 'warning')
 *   - crisisType: what crisis event was active ('trauma', 'seizure_spike', 'sedation', or null)
 *   - crisisRegion: which brain region was hit (for trauma events)
 *
 * Each explanation connects the game mechanics to real neuroscience:
 *   - GoL birth/death rules → neural excitation and inhibition
 *   - Noise → spontaneous neural firing
 *   - Defects → amyloid plaques / neuronal damage (Fraile Section 5)
 *   - Critical zone → the brain's "edge of chaos" (Chialvo 2010, Beggs & Timme 2012)
 */
function getBioExplanation(deathState, crisisType, crisisRegion) {
  // Crisis-specific explanations take priority (the crisis is what caused the death)
  if (crisisType === 'trauma') {
    const region = crisisRegion ? crisisRegion.toLowerCase() : 'a brain region';
    if (deathState === 'underactive') {
      return `A traumatic injury destroyed neurons across the ${region}. Without enough active cells to propagate signals, the network fell below the density needed for self-sustaining activity — mirroring focal brain injury, where loss of local circuitry silences the surrounding tissue. In sparse networks, inactive cells rarely receive enough simultaneous neighbor input to reactivate, so activity keeps fading out.`;
    }
    return `A traumatic injury to the ${region} disrupted the network's spatial organization. With noise elevated and fewer structured propagation paths, the remaining cells spiraled into uncoordinated firing. In real neural tissue, focal lesions can paradoxically trigger runaway excitation in surrounding regions by removing inhibitory circuits.`;
  }

  if (crisisType === 'sedation') {
    return `Sedation suppressed spontaneous neural firing to near-zero and raised the activation threshold — mimicking how anesthetic drugs reduce neural excitability across the cortex. With threshold T=4, a neuron needed four simultaneously active neighbors to fire, a rare event in a sparse network. The system dropped below the critical density where activity can sustain itself: each generation produced fewer births than deaths, and the network went quiet.`;
  }

  if (crisisType === 'seizure_spike') {
    return `A sudden surge in spontaneous firing overwhelmed the network's regulatory dynamics. When noise-driven births outnumber stabilizing survival dynamics, spatial structure collapses — cells fire without coordination, no longer in organized repeating patterns that define healthy criticality. This is the computational equivalent of epileptic seizure: runaway excitation that bypasses the brain's self-regulatory dynamics.`;
  }

  // No crisis was active — death from gradual drift
  if (deathState === 'underactive') {
    return `Neural activity drifted below the critical density needed for self-sustaining propagation. Sparse networks lack enough simultaneously active neighbors to reliably trigger new activations — the network gradually goes quiet. Biologically, this resembles the loss of criticality seen in deep anesthesia or severe cortical depression, where activity falls below the threshold for coherent signal propagation.`;
  }
  if (deathState === 'overloaded') {
    return `Unchecked excitation drove the network into saturation. When spontaneous (noise-driven) activations dominate over stabilizing local dynamics, cells fire without spatial coordination — no structured patterns, just random flickering. The brain's analogue is epileptic seizure: excitation propagates indiscriminately, and the organized dynamics that enable computation collapse into noise.`;
  }

  // Fallback (shouldn't normally be reached)
  return `The network lost its critical balance — the narrow operating range where activity is high enough to propagate but low enough to remain structured. Healthy neural function depends on this balance: too quiet and signals can't travel; too active and structure dissolves into noise.`;
}


/* 8. EVENT HANDLERS */

/** Convert a mouse/touch position on the canvas to a grid row and column. */
function canvasPosToGrid(evt) {
  const rect   = gridCanvas.getBoundingClientRect();
  const scaleX = gridCanvas.width  / rect.width;
  const scaleY = gridCanvas.height / rect.height;
  const px = (evt.clientX - rect.left) * scaleX;
  const py = (evt.clientY - rect.top)  * scaleY;
  return {
    row: Math.floor(py / cellSize),
    col: Math.floor(px / cellSize),
    px:  evt.clientX - rect.left,
    py:  evt.clientY - rect.top,
  };
}

function handleCanvasClick(evt) {
  if (!gameActive) return;
  const { row, col, px, py } = canvasPosToGrid(evt);
  // Stimulate a small circular area around the click point
  stimulateRegion(row, col, Math.max(1, Math.floor(gridSize / 15)));
  showPulse(px, py);
}

function handleCanvasMouseMove(evt) {
  if (!isMouseDown || !gameActive) return;
  const { row, col } = canvasPosToGrid(evt);
  // Slightly smaller radius for drag (so it's not too overpowered)
  stimulateRegion(row, col, Math.max(1, Math.floor(gridSize / 20)));
}

/**
 * Reset all control sliders to the stable baseline.
 * Used after game-over so the next run always starts from known stable values.
 */
function applyStableStartPreset() {
  setSlider('speed-ctrl', STABLE_START.simSpeed);
  setSlider('noise-ctrl', STABLE_START.noise);
  setSlider('thresh-ctrl', STABLE_START.threshold);
  setSlider('damp-ctrl', STABLE_START.damping);
}

/** Wire up all event listeners — called once at startup. */
function setupEventHandlers() {
  // Mouse events for click and drag stimulation
  gridCanvas.addEventListener('mousedown', e => { isMouseDown = true; handleCanvasClick(e); });
  gridCanvas.addEventListener('mousemove', handleCanvasMouseMove);
  window.addEventListener('mouseup', () => { isMouseDown = false; });

  // Touch events (mobile support)
  gridCanvas.addEventListener('touchstart', e => {
    e.preventDefault(); handleCanvasClick(e.touches[0]);
  }, { passive: false });
  gridCanvas.addEventListener('touchmove', e => {
    e.preventDefault(); handleCanvasMouseMove(e.touches[0]);
  }, { passive: false });

  // Pause/Resume button
  document.getElementById('pause-btn').addEventListener('click', () => {
    if (!gameActive) return;
    running = !running;
    document.getElementById('pause-btn').textContent = running ? 'PAUSE' : 'RESUME';
    addLog(running ? 'Simulation resumed' : 'Simulation paused');
  });

  // Reset button — reinitializes the grid without ending the game
  document.getElementById('reset-btn').addEventListener('click', () => {
    if (!gameActive) return;
    initGrid();
    activityHistory = [];
    stepAccum       = 0;
    dangerTimer     = 0;
    activeEvent     = null;
    traumaDefects    = [];
    traumaDefectSet  = new Set();
    traumaRepairTotal = 0;
    nextEventAt     = elapsed + 15 + Math.random() * 10;
    hideWarningOverlay();
    warningOverlayActive = false;
    addLog('Simulation reset', 'warn-entry');
  });

  // Sim speed slider (cosmetic — doesn't change the model)
  document.getElementById('speed-ctrl').addEventListener('input', e => {
    simSpeed = parseInt(e.target.value);
    document.getElementById('speed-val').textContent = simSpeed;
  });

  // Noise slider: controls spontaneous firing probability (slider 0-100 → p = 0-3%)
  document.getElementById('noise-ctrl').addEventListener('input', e => {
    noise = parseInt(e.target.value);
    document.getElementById('noise-val').textContent = noise;
    updateNoiseProbDisplay();
  });

  // Threshold slider: GoL birth rule neighbour count T
  document.getElementById('thresh-ctrl').addEventListener('input', e => {
    threshold = parseInt(e.target.value);
    document.getElementById('thresh-val').textContent = threshold;
  });

  // Defect slider: controls % of permanently dead neurons
  document.getElementById('damp-ctrl').addEventListener('input', e => {
    damping = parseInt(e.target.value);
    const pct = ((damping / 100) * 10).toFixed(1);
    document.getElementById('damp-val').textContent = pct + '%';
    // Regenerate defect pattern live as the slider moves
    regenerateDefects();
    addLog('Defects updated: ' + pct + '%', 'warn-entry');
  });

  // Start button (on intro screen 4)
  document.getElementById('start-btn').addEventListener('click', () => {
    document.getElementById('intro-modal').classList.add('hidden');
    startGame();
  });

  // "Play Again" button on game-over screen — goes to launch screen
  document.getElementById('restart-btn').addEventListener('click', () => {
    document.getElementById('gameover-modal').classList.add('hidden');
    applyStableStartPreset();
    document.getElementById('intro-modal').classList.remove('hidden');
    nextScreen(4);
  });

  // "Read Briefing" button on game-over screen — goes back to screen 1
  document.getElementById('restart-briefing-btn').addEventListener('click', () => {
    document.getElementById('gameover-modal').classList.add('hidden');
    applyStableStartPreset();
    document.getElementById('intro-modal').classList.remove('hidden');
    nextScreen(1);
  });

  // Re-fit canvases when the browser window resizes
  window.addEventListener('resize', () => { resizeGridCanvas(); resizeGraphCanvas(); });
}


/* 8b. CRISIS EVENT SYSTEM — random emergencies that challenge the player */

// Brain regions that can be targeted by trauma events.
// Each covers roughly 30-35% of the 60x60 grid (large enough that repairing requires dragging).
const BRAIN_REGIONS = [
  { name: 'FRONTAL LOBE',   row: 0,                                  col: Math.floor(FIXED_GRID_SIZE * 0.08), w: Math.floor(FIXED_GRID_SIZE * 0.84), h: Math.floor(FIXED_GRID_SIZE * 0.38) },
  { name: 'LEFT TEMPORAL',  row: Math.floor(FIXED_GRID_SIZE * 0.20), col: 0,                                  w: Math.floor(FIXED_GRID_SIZE * 0.48), h: Math.floor(FIXED_GRID_SIZE * 0.65) },
  { name: 'RIGHT TEMPORAL', row: Math.floor(FIXED_GRID_SIZE * 0.20), col: Math.floor(FIXED_GRID_SIZE * 0.52), w: Math.floor(FIXED_GRID_SIZE * 0.48), h: Math.floor(FIXED_GRID_SIZE * 0.65) },
  { name: 'OCCIPITAL LOBE', row: Math.floor(FIXED_GRID_SIZE * 0.58), col: Math.floor(FIXED_GRID_SIZE * 0.08), w: Math.floor(FIXED_GRID_SIZE * 0.84), h: Math.floor(FIXED_GRID_SIZE * 0.40) },
  { name: 'PARIETAL LOBE',  row: Math.floor(FIXED_GRID_SIZE * 0.08), col: Math.floor(FIXED_GRID_SIZE * 0.08), w: Math.floor(FIXED_GRID_SIZE * 0.60), h: Math.floor(FIXED_GRID_SIZE * 0.52) },
];

// How long each crisis type is announced before it actually fires (seconds)
const CRISIS_EVENTS = {
  trauma:        { announce: 2.5 },
  seizure_spike: { announce: 2.0 },
  sedation:      { announce: 2.0 },
};

/**
 * Helper: programmatically move a slider and fire its input event.
 * This is the key mechanic — crisis events ACTUALLY move the slider,
 * so the player sees it jump and must drag it back.
 */
function setSlider(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = value;
  el.dispatchEvent(new Event('input'));
}

/** Schedule the next crisis event (15-25 seconds from now). */
function scheduleNextEvent() {
  nextEventAt = elapsed + 15 + Math.random() * 10;
}

/**
 * TRAUMA crisis — destroys an entire brain region.
 *
 * How it works:
 *   1. ALL cells in the target region become temporary DEFECTS (red).
 *   2. Defects block noise from reactivating the region — no self-healing.
 *   3. The noise slider drops to 8, compounding the pressure.
 *   4. The player must CLICK/DRAG across the red zone to repair cells,
 *      AND drag the noise slider back up.
 *
 * Biologically, this models focal brain injury: local neuron death
 * silences surrounding tissue (Fraile Section 5, Alzheimer's analogy).
 */
function triggerTrauma(regionName) {
  lastCrisisType   = 'trauma';
  lastCrisisRegion = regionName;
  let region = BRAIN_REGIONS.find(r => r.name === regionName);
  if (!region) region = BRAIN_REGIONS[Math.floor(Math.random() * BRAIN_REGIONS.length)];

  let killed = 0;
  const newDefects = [];

  for (let r = region.row; r < Math.min(region.row + region.h, gridSize); r++) {
    for (let c = region.col; c < Math.min(region.col + region.w, gridSize); c++) {
      if (grid[r][c] === STATE_DEFECT) continue;  // don't overwrite existing permanent defects
      if (grid[r][c] === STATE_ACTIVE) killed++;
      grid[r][c] = STATE_DEFECT;
      const key = r + ',' + c;
      newDefects.push({ r, c });
      traumaDefectSet.add(key);
    }
  }

  // Store trauma cells — they persist until the player clicks to repair them
  traumaDefects = traumaDefects.concat(newDefects);
  traumaRepairTotal += newDefects.length;

  // Also drop the noise slider, making the situation worse
  setSlider('noise-ctrl', 8);

  addLog('TRAUMA HIT: ' + region.name + ' — ' + killed + ' neurons destroyed. CLICK THE RED ZONE TO REPAIR (' + newDefects.length + ' cells).', 'bad-entry');
}

/**
 * SEIZURE SPIKE crisis — slams the noise slider to maximum.
 *
 * At slider=100 (p=3%), density surges to ~27%, right at seizure threshold.
 * The player must quickly drag the noise slider back down.
 *
 * Biologically, this models a sudden surge in excitatory neurotransmission —
 * like the onset of an epileptic seizure.
 */
function triggerSeizureSpike() {
  lastCrisisType   = 'seizure_spike';
  lastCrisisRegion = null;
  setSlider('noise-ctrl', 100);
  addLog('SEIZURE SPIKE — noise slammed to maximum! Drag it back down!', 'bad-entry');
}

/**
 * SEDATION crisis — kills noise AND raises threshold.
 *
 * Noise at 0 means no spontaneous firing. T=4 means GoL birth needs
 * exactly 4 active neighbors (very rare in a sparse network).
 * Together, density drops to ~1% — deep flatline.
 * The player must fix BOTH sliders: noise up and threshold down to 3.
 *
 * Biologically, this models general anesthesia: suppressed excitability
 * across the entire cortex.
 */
function triggerSedation() {
  lastCrisisType   = 'sedation';
  lastCrisisRegion = null;
  setSlider('noise-ctrl', 0);
  setSlider('thresh-ctrl', 4);
  addLog('SEDATION — noise killed, threshold raised to 4! Fix both sliders!', 'bad-entry');
}

/**
 * Check whether all trauma defects have been repaired by the player.
 * (Actual repair happens in stimulateRegion when the player clicks.)
 */
function updateTraumaDefects() {
  if (traumaDefects.length === 0) return;

  // Count how many trauma cells are still defects
  let remaining = 0;
  for (const cell of traumaDefects) {
    if (grid[cell.r] && grid[cell.r][cell.c] === STATE_DEFECT) remaining++;
  }

  if (remaining === 0) {
    // All cells repaired — clean up
    traumaDefects    = [];
    traumaDefectSet  = new Set();
    traumaRepairTotal = 0;
    addLog('REPAIR COMPLETE — region restored. Stimulate to repopulate!', 'good-entry');
  }
}

/** Show the crisis announcement overlay (warning before the crisis hits). */
function showCrisisAnnouncement(type, regionName) {
  const overlay = document.getElementById('warning-overlay');
  if (!overlay) return;
  const title = document.getElementById('warning-overlay-title');
  const sub   = document.getElementById('warning-overlay-sub');
  const bar   = document.getElementById('warning-countdown-bar');

  const config = CRISIS_EVENTS[type];

  if (type === 'trauma') {
    title.textContent = '⚡ INCOMING TRAUMA';
    sub.textContent   = 'Impact: ' + (regionName || 'UNKNOWN') + ' — region destroyed + noise dropping!';
    overlay.className = 'warning-overlay show seizure';
  } else if (type === 'seizure_spike') {
    title.textContent = '⚡ SEIZURE SPIKE INCOMING';
    sub.textContent   = 'Noise slider will spike to max — grab it and drag it back down!';
    overlay.className = 'warning-overlay show seizure';
  } else if (type === 'sedation') {
    title.textContent = '💉 SEDATION WAVE INCOMING';
    sub.textContent   = 'Noise will drop to zero AND threshold will spike — fix both sliders!';
    overlay.className = 'warning-overlay show flatline';
  }

  // Animate the countdown bar (shows how much time before the crisis hits)
  bar.style.transition = 'none';
  bar.style.width      = '100%';
  bar.offsetWidth;
  bar.style.transition = `width ${config.announce}s linear`;
  bar.style.width      = '0%';
}

/**
 * Main crisis system tick — called every frame from the game loop.
 * Handles three things:
 *   1. Checking if trauma repairs are complete
 *   2. Counting down active crisis announcements
 *   3. Scheduling and triggering new crises
 */
function updateCrisisEvents(dt) {
  // Check if all trauma defects have been repaired
  updateTraumaDefects();

  // If a crisis is being announced, count down
  if (activeEvent && activeEvent.phase === 'announcing') {
    activeEvent.announceRemaining -= dt;
    if (activeEvent.announceRemaining <= 0) {
      // Announcement done — fire the crisis
      hideWarningOverlay();

      if (activeEvent.type === 'trauma') {
        triggerTrauma(activeEvent.regionName);
      } else if (activeEvent.type === 'seizure_spike') {
        triggerSeizureSpike();
      } else if (activeEvent.type === 'sedation') {
        triggerSedation();
      }

      // Crisis is done — the slider stays where we put it
      activeEvent = null;
    }
    return;
  }

  // Schedule a new crisis if enough time has passed
  if (!activeEvent && elapsed >= nextEventAt) {
    const types = ['trauma', 'seizure_spike', 'sedation'];
    const type  = types[Math.floor(Math.random() * types.length)];

    let regionName = null;
    if (type === 'trauma') {
      regionName = BRAIN_REGIONS[Math.floor(Math.random() * BRAIN_REGIONS.length)].name;
    }

    activeEvent = {
      type: type,
      phase: 'announcing',
      announceRemaining: CRISIS_EVENTS[type].announce,
      regionName: regionName,
    };

    showCrisisAnnouncement(type, regionName);
    scheduleNextEvent();
  }
}


/* 9. GAME LOOP */

/**
 * The main game loop — called every animation frame by requestAnimationFrame.
 * Steps the simulation, updates metrics, and renders everything.
 */
function gameLoop(timestamp) {
  if (!gameActive) return;

  // Calculate time since last frame (capped at 0.1s to prevent huge jumps)
  const dt = Math.min((timestamp - lastTime) / 1000, 0.1);
  lastTime = timestamp;

  if (running) {
    elapsed   += dt;
    stepAccum += dt;

    // Update crisis events (announcements, active effects, scheduling)
    updateCrisisEvents(dt);

    // Run as many simulation steps as the accumulated time allows
    const msPerStep = 1 / simSpeed;
    while (stepAccum >= msPerStep) {
      stepSimulation();
      stepAccum -= msPerStep;
    }

    // Update all UI numbers and check if the player is in danger
    updateMetrics(dt);
    updateHealthAndSurvival(dt);
    updateTopBar();
  }

  // Always render (even when paused, so the grid stays visible)
  renderGrid();
  renderGraph();
  requestAnimationFrame(gameLoop);
}

/** Initialize and start a new game. */
function startGame() {
  initGrid();
  resizeGridCanvas();
  resizeGraphCanvas();

  // Reset all game state
  health       = 100;
  elapsed      = 0;
  dangerTimer  = 0;
  brainState   = 'stable';
  lastBrainState  = '';
  stepAccum    = 0;
  activityHistory = [];
  warningOverlayActive = false;

  // Reset crisis event system
  activeEvent      = null;
  traumaDefects    = [];
  traumaDefectSet  = new Set();
  traumaRepairTotal = 0;
  lastCrisisType   = null;
  lastCrisisRegion = null;
  nextEventAt     = 15 + Math.random() * 10;  // first event at 15-25 seconds

  document.getElementById('event-log').innerHTML = '';
  hideWarningOverlay();

  // Read current slider values (player may have adjusted them before starting)
  simSpeed  = parseInt(document.getElementById('speed-ctrl').value);
  noise     = parseInt(document.getElementById('noise-ctrl').value);
  threshold = parseInt(document.getElementById('thresh-ctrl').value);
  damping   = parseInt(document.getElementById('damp-ctrl').value);

  running    = true;
  gameActive = true;
  lastTime   = performance.now();

  document.getElementById('pause-btn').textContent = 'PAUSE';
  addLog('OP INITIATED — neural model active', 'good-entry');

  requestAnimationFrame(gameLoop);
}


/* 10. INITIALIZATION */

/** One-time setup when the page loads. */
function init() {
  gridCanvas  = document.getElementById('grid-canvas');
  gridCtx     = gridCanvas.getContext('2d');
  graphCanvas = document.getElementById('graph-canvas');
  graphCtx    = graphCanvas.getContext('2d');
  setupEventHandlers();
  startIntroBgAnimation();
  buildBrainOutline(400, 400);
  updateNoiseProbDisplay();
  // Initialize the defect display to show "0.0%"
  document.getElementById('damp-val').textContent = '0.0%';
}

document.addEventListener('DOMContentLoaded', init);


/* 11. INTRO SCREENS & CINEMATIC UI */

/** Switch to a specific intro screen (1-4). */
function nextScreen(n) {
  document.querySelectorAll('.intro-screen').forEach(s => s.classList.remove('active'));
  const target = document.getElementById('screen-' + n);
  if (target) target.classList.add('active');
}

/**
 * Draw the animated neural-network background on the intro screens.
 * Creates floating nodes connected by lines (purely decorative).
 */
function startIntroBgAnimation() {
  const canvas = document.getElementById('intro-bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };
  resize();
  window.addEventListener('resize', resize);

  const NODE_COUNT = 60;
  const MAX_DIST   = 160;  // max distance between nodes to draw a connecting line
  const nodes = Array.from({ length: NODE_COUNT }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 0.3,  // slow random drift
    vy: (Math.random() - 0.5) * 0.3,
    r:  Math.random() * 1.5 + 0.5,    // node radius
  }));

  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Move nodes and bounce off edges
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
    });
    // Draw lines between nearby nodes (fades with distance)
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < MAX_DIST) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(0, 150, 220, ${(1 - d / MAX_DIST) * 0.4})`;
          ctx.lineWidth   = 0.5;
          ctx.stroke();
        }
      }
    }
    // Draw the nodes themselves as small glowing dots
    nodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 200, 255, 0.6)';
      ctx.fill();
    });
    requestAnimationFrame(frame);
  }
  frame();
}

/**
 * Draw the animated flatline EKG on the game-over screen.
 * Shows one heartbeat spike, then flatlines — a classic "time of death" visual.
 */
function drawFlatlineCanvas() {
  const canvas = document.getElementById('flatline-canvas');
  if (!canvas) return;
  const ctx  = canvas.getContext('2d');
  const W    = canvas.width;
  const H    = canvas.height;
  const midY = H / 2;
  let progress = 0;

  function frame() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#020508';
    ctx.fillRect(0, 0, W, H);
    progress = Math.min(progress + 3, W);

    ctx.beginPath();
    ctx.strokeStyle = '#ff3355';  // danger red
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = '#ff3355';
    ctx.shadowBlur  = 8;

    // Draw the EKG line — mostly flat with one spike in the middle
    for (let x = 0; x < progress; x++) {
      const frac = x / W;
      let y = midY;
      // Create a sharp spike between 35% and 42% of the width
      if (frac > 0.35 && frac < 0.42) {
        const t = (frac - 0.35) / 0.07;
        if      (t < 0.3) y = midY - (t / 0.3) * (H * 0.7);          // spike up
        else if (t < 0.5) y = midY - ((1 - (t - 0.3) / 0.2)) * (H * 0.7); // come back down
        else if (t < 0.7) y = midY + ((t - 0.5) / 0.2) * (H * 0.3);       // dip below
        else              y = midY + ((1 - (t - 0.7) / 0.3)) * (H * 0.3);  // return to flat
      }
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    if (progress < W) requestAnimationFrame(frame);
  }
  frame();
}

/** Update the top status bar (elapsed time and operation status text). */
function updateTopBar() {
  const mins = Math.floor(elapsed / 60);
  const secs = Math.floor(elapsed % 60);
  const pad  = n => String(n).padStart(2, '0');
  const el   = document.getElementById('top-bar-time');
  if (el) el.textContent = `T+ ${pad(mins)}:${pad(secs)}`;

  const statusEl = document.getElementById('op-status-text');
  if (statusEl) {
    const texts = {
      stable:      'OPERATION IN PROGRESS',
      underactive: 'WARNING: LOW ACTIVITY',
      overloaded:  'ALERT: NEURAL OVERLOAD',
      warning:     'CAUTION: DRIFTING',
    };
    statusEl.textContent = texts[brainState] || 'OPERATION IN PROGRESS';
  }

  updateBrainIcon();
}
