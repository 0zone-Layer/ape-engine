# APE Engine — Full Refactor Prompt for GitHub Copilot

You are working on a single-file React app called **APE (Algo/Adaptive Prediction Engine)** — `App.jsx`, ~6090 lines.
It is a 100-value (0–99) number prediction engine with ~80+ ensemble algorithms, Gaussian PMF vote aggregation, backtest scoring, walk-forward evaluation, weight evolution, localStorage persistence, and a full UI.

Apply **every change listed below** in order. Do NOT break any existing functionality. Do NOT change the UI layout unless a feature explicitly requires it. Do NOT add new bugs. Keep the file as a single `.jsx`. After every change, make sure the component still compiles and all existing state keys are preserved.

---

## PART 1 — BUG FIXES (apply all, in order)

---

### BUG 1 — Stack overflow in `addRow()` — CRITICAL
**Location:** Inside `addRow()`, the line computing the max row number.

**Problem:** `Math.max(...curRows.map(x => x.row))` uses spread as arguments. Crashes with `RangeError: Maximum call stack size exceeded` on large datasets (>65k rows).

**Fix:** Replace with `.reduce()`:
```js
// BEFORE
const maxR = curRows.length ? Math.max(...curRows.map(x => x.row)) : r;

// AFTER
const maxR = curRows.length ? curRows.reduce((m, x) => x.row > m ? x.row : m, r) : r;
```

---

### BUG 2 — Race condition in `addRow()` — CRITICAL
**Location:** The `setRows(prev => { ... setTimeout(() => setS(...)) })` pattern in `addRow()`.

**Problem:** `setRows` is async. The 100ms `setTimeout` reads `cur.datasets[cur.active].rows` from a potentially stale closure. If two rows are added within 100ms, the second auto-generation check fires against the wrong row count, triggering evolution/generation at incorrect milestones.

**Fix:** Remove the `setTimeout`. Merge the entire row-add + auto-generation into a single `setS` functional updater. The `setS` callback receives guaranteed-fresh state:
```js
// AFTER — one atomic update, no setTimeout
setS(cur => {
  const ds = { ...cur.datasets };
  const active = cur.active;
  const curRows = [...(ds[active]?.rows || []).filter(x => x.row !== r), entry]
    .sort((a, b) => a.row - b.row);
  ds[active] = { ...ds[active], rows: curRows };
  let next = { ...cur, datasets: ds };
  // call your existing auto-generation / evolution logic here with next
  next = maybeAutoGenerate(next, curRows.length);
  next = maybeAutoEvolve(next, curRows.length);
  saveS(next);
  return next;
});
```
Remove the now-unused `setRows` call from `addRow`.

---

### BUG 3 — `FreqDecay` exponential overflow — HIGH
**Location:** Algorithm `A.FreqDecay` — the `Math.pow(1.6, i)` weight line.

**Problem:** At 100 rows, `1.6^99 ≈ 1.8×10²¹`. The oldest values get weight `1` vs newest `10²¹`. The algorithm degenerates to always returning the single most-recent value. The `+5` recency bonus for the last 3 values is completely swamped.

**Fix:** Replace the exponential with a gentle recency decay:
```js
// BEFORE
s.forEach((v, i) => { freq[v] = (freq[v] || 0) + Math.pow(1.6, i); });

// AFTER
s.forEach((v, i) => {
  const age = s.length - 1 - i; // 0 = newest
  freq[v] = (freq[v] || 0) + Math.exp(-age * 0.12);
});
```

---

### BUG 4 — "Perfect prediction" alert fires at 4/7 columns — HIGH
**Location:** `checkAndLearn()` — the `if(exactCount===4)` line.

**Problem:** The hardcoded `4` means the alert fires at 57% accuracy (4 of 7 columns). It should fire only when ALL known columns are predicted exactly.

**Fix:**
```js
// BEFORE
if (exactCount === 4) syslog("✨ Perfect prediction!...", "alert");

// AFTER
if (exactCount === knownCols.length && knownCols.length > 0)
  syslog(`✨ Perfect prediction! ${exactCount}/${knownCols.length} columns exact`, "alert");
```

---

### BUG 5 — Vote dominance cap bypassed by cluster boost — HIGH
**Location:** `predictCol()` — the section where `MAX_VOTE_DOMINANCE_MULT` cap is applied, then the dense-cluster boost re-multiplies.

**Problem:** The cap is applied first (`votes[vi] = Math.min(votes[vi], maxVote * MAX_VOTE_DOMINANCE_MULT)`). Then the cluster boost runs and pushes the winner above the cap. The cap is effectively a no-op for the strongest candidate.

**Fix:** Apply the cluster boost BEFORE the cap pass:
```js
// Step 1 — apply cluster boost
if (bestCluster) {
  votes[bestCluster.center] = (votes[bestCluster.center] || 0) * clusterBoost
    + bestCluster.mass * DENSE_CLUSTER_MASS_WEIGHT;
}

// Step 2 — cap AFTER boost (so cap actually enforces the limit)
const newMax = Math.max(...Object.values(votes).filter(v => v > 0));
Object.keys(votes).forEach(vi => {
  votes[vi] = Math.min(votes[vi], newMax * MAX_VOTE_DOMINANCE_MULT);
});
```

---

### BUG 6 — `consensus%` always capped at ~25% — MEDIUM
**Location:** `predictCol()` — the `consensus` calculation after `top5` is built.

**Problem:** `top5[0].algos` is capped to `MAX_TOP_CONTRIBUTING_ALGOS` (24), but `ac` (total algos) is up to 96. So `consensus = top5[0].algos.length / ac * 100` maxes out at `24/96 = 25%`, permanently. The `minConsensusPct` threshold (14%) always appears easily met.

**Fix:** Count agreeing algos from the full `algoAgreement` map before the cap is applied:
```js
// Compute true consensus BEFORE capping top5[0].algos
const trueAgreeing = Object.values(algoAgreement)
  .filter(w => w >= algoAgreement[top5[0].value] * 0.5).length;
const consensus = Math.round(trueAgreeing / ac * 100);
```

---

### BUG 7 — `_contextCache` grows unboundedly — MEDIUM
**Location:** `classifyContext()` — the `_contextCache` module-level object.

**Problem:** Unlike `_TC` (which has `.clear()` called on every data change), `_contextCache` is never cleared. Keys are `col+dataLength+cacheStep+_ver`. In a 300-row auto-train × 7 cols = 2100+ stale entries accumulate. On mobile browsers this contributes to silent OOM tab kills.

**Fix:** Integrate `_contextCache` into `_TC`:
```js
// Add to _TC object definition:
const _TC = {
  tx:{}, sp:{}, cg:{}, gs:{}, ctx:{},  // add ctx here
  _ver:0,
  bumpVer(){ this._ver=(this._ver+1)%1e9; },
  clear(){ this.tx={}; this.sp={}; this.cg={}; this.gs={}; this.ctx={}; },
};

// In classifyContext, replace _contextCache with _TC.ctx:
const cacheKey = col + dataLen + cacheStep + _TC._ver;
if(_TC.ctx[cacheKey]) return _TC.ctx[cacheKey];
// ... compute result ...
_TC.ctx[cacheKey] = result;
return result;
```
Remove the standalone `const _contextCache = {}` declaration entirely.

---

### BUG 8 — `FAILSAFE_PREDICT_MS` silently drops to 1 prediction — MEDIUM
**Location:** `predictCol()` — the failsafe splice section.

**Problem:** When prediction takes >34ms, `top5.splice(1)` silently reduces to 1 candidate. The user sees no indication that the failsafe fired. On loaded mobile devices this fires regularly.

**Fix:** Return a `failsafeFired` flag in the result object and show a `⚡ Throttled` badge in the UI next to the column header when it is true:
```js
// In predictCol return object:
return { top5, confidence, consensus, regime, failsafeFired };

// In the UI (column prediction card):
{result.failsafeFired && (
  <span style={{fontSize:9, color:"#f97316", marginLeft:4}}>⚡</span>
)}
```

---

### BUG 9 — `AlternatingStep` uses absolute index parity — MEDIUM
**Location:** `A.AlternatingStep` — the backtest loop condition `i%2===0`.

**Problem:** The zigzag phase depends on WHERE in the series the pattern begins, not on the absolute index. If the first alternation starts at an odd index, every expected-vs-actual comparison is phase-flipped, and the algo scores near-zero even when the pattern is perfectly real.

**Fix:** Try both phases and keep the better-scoring one:
```js
let best = { sc: -1, k: 0, phase: 0 };
for (const k of candidates) {
  for (const phase of [0, 1]) {
    let sc = 0;
    for (let i = 2; i < n; i++) {
      const expected = (i + phase) % 2 === 0
        ? M.mod(s[i-2])
        : M.mod(s[i-1] + (s[i-1] > s[i-2] ? -k : k));
      if (expected === s[i]) sc++;
    }
    if (sc > best.sc) best = { sc, k, phase };
  }
}
// Use best.phase in the final prediction step as well
```

---

### BUG 10 — `WichmannHill` tests 21 duplicate combos — MEDIUM
**Location:** `A.WichmannHill` — the triple nested loop over `[171,172,170]`.

**Problem:** 3³=27 iterations are run but the three values are just permutations of `{170,171,172}`. Only 3!=6 permutations are unique. The remaining 21 re-test the same `(ma,mb,mc)` values, wasting computation.

**Fix:** Replace with an explicit permutation list:
```js
const perms = [
  [171,172,170],[171,170,172],[172,171,170],
  [172,170,171],[170,171,172],[170,172,171]
];
for (const [ma, mb, mc] of perms) { /* same body */ }
```

---

### BUG 11 — `EntropyAdapt` entropy window ≠ linear fit window — MEDIUM
**Location:** `A.EntropyAdapt` — entropy computed on `s.slice(-8)`, linear fit on full `s`.

**Problem:** `s.slice(-8)` has max entropy `log2(8)=3.0`, making the `>3.2` threshold unreachable. Meanwhile when entropy is "low" the linear fit runs on ALL rows including stale data.

**Fix:** Use a shared window of `min(n, 20)` for both entropy and linear fit:
```js
const w = s.slice(-Math.min(s.length, 20));
// compute entropy on w (max entropy now ~4.3, threshold 3.2 is reachable)
// compute linear fit on w (same data, no stale rows)
w.forEach((v, i) => { /* linear fit */ });
```

---

### BUG 12 — Triple `M.std()` call in `SameRow` signals — MEDIUM
**Location:** `getSameRowHistory()` — the three consecutive `if(M.std(intraVals)<...)` conditions.

**Problem:** `M.std(intraVals)` is called 3 separate times for the same array. Cache it:
```js
// BEFORE
if (M.std(intraVals) < 5 && ...) res["SameRowTight"] = ...;
if (M.std(intraVals) >= 5 && M.std(intraVals) < 8 && ...) res["SameRowSnug"] = ...;
if (M.std(intraVals) < 8 && ...) res["SameRowMed"] = ...;

// AFTER
const ivStd = M.std(intraVals);
if (ivStd < 5 && intraVals.length >= 3) res["SameRowTight"] = ...;
if (ivStd >= 5 && ivStd < 8 && intraVals.length >= 3) res["SameRowSnug"] = ...;
if (ivStd < 8 && intraVals.length >= 3) res["SameRowMed"] = ...;
```

---

### BUG 13 — Dead-zone bias correction barely fires — MEDIUM
**Location:** `predictCol()` — `const _slimAccLog = (S.accLog||[]).slice(-10)`.

**Problem:** Sliced to 10 entries. The correction requires `recentErrs.length >= 4` non-null errors per column. With 10 total entries, some null-filtered, this barely meets the threshold. The bias correction almost never has statistical power.

**Fix:**
```js
// BEFORE
const _slimAccLog = (S.accLog || []).slice(-10);

// AFTER
const _slimAccLog = (S.accLog || []).slice(-20);
```

---

### BUG 14 — Columns E/F/G get wrong temporal weights — LOW
**Location:** `T_MINS` / `T_MINS_BASE` — the auto-assignment of positions for columns E, F, G.

**Problem:** E/F/G are assigned positions 1610, 1790, 1970 minutes (D + n×180). Column G at 1970 min is past midnight. `T_TO_NEXT_A[G] = max(1, 1800-1970) = 1`, giving G the HIGHEST temporal weight (≈0.999), when it should have the lowest (it is furthest from next draw A).

**Fix:** Assign realistic intra-day times for E/F/G:
```js
const T_MINS_BASE = {
  A: 360,   // 6:00 AM
  B: 1080,  // 6:00 PM
  C: 1260,  // 9:00 PM
  D: 1430,  // 11:50 PM
  E: 1500,  // wraps — treat as 60 min next day (use modular distance)
  F: 1560,
  G: 1620,
};
// For E/F/G use modular wrap when computing T_TO_NEXT_A:
// T_TO_NEXT_A[col] = ((T_MINS_A + 1440 - T_MINS[col]) % 1440) || 1
```

---

### BUG 15 — `Cyclic` algo off-by-one boundary — LOW
**Location:** `A.Cyclic` — the for loop condition.

**Problem:** `for(let i=n%p||p; i<=n; i+=p)` — when `i===n`, `s[n-i]=s[0]` (the oldest value) is included with weight `exp(-n*0.3)≈0`. Semantically incorrect: a period match at exact distance `n` is meaningless.

**Fix:**
```js
// BEFORE
for (let i = n%p || p; i <= n; i += p)

// AFTER
for (let i = n%p || p; i < n; i += p)
```

---

### BUG 16 — `localeCompare` for date sorting — LOW
**Location:** Any `.sort()` call that uses `a.date.localeCompare(b.date)` (appears in `getGlobalSeries` and `doAutoTrain`).

**Problem:** `localeCompare` is locale-sensitive. Works accidentally for YYYY-MM-DD. Breaks silently if any date is stored in a different format.

**Fix:** Replace every occurrence with a ternary string comparison:
```js
// BEFORE
return a.date.localeCompare(b.date);

// AFTER
return a.date < b.date ? -1 : a.date > b.date ? 1 : 0;
```

---

## PART 2 — NEW ALGORITHMS (add all inside the existing `A` algorithm object)

Add each algorithm as a named function on the `A` object, following the exact same pattern as existing algos: `A.AlgoName = s => { ... return M.mod(result); }`. Each algo must return a value in [0,99]. Use the existing `M` helper for all modular math. Guard against short series with `if(s.length < minLength) return null`.

---

### NEW ALGO 1 — `KalmanFilter`
A 1D Kalman filter treating each column as a noisy linear system. Maintains `x` (state estimate) and `P` (error covariance). After processing all series values, predicts the next state.

```js
A.KalmanFilter = s => {
  if (s.length < 4) return null;
  const Q = 2;    // process noise
  const R = 8;    // measurement noise
  let x = s[0];
  let P = 10;
  for (let i = 1; i < s.length; i++) {
    // Predict
    const xp = x;
    const Pp = P + Q;
    // Update
    const K = Pp / (Pp + R);
    x = xp + K * (s[i] - xp);
    P = (1 - K) * Pp;
  }
  return M.mod(Math.round(x));
};
```

---

### NEW ALGO 2 — `FFTHarmonic3`
Extends single-period DFT to extract the **top 3 dominant frequencies** and predict as their superposition. Handles mixed-period series that `DFTPeriod` misses.

```js
A.FFTHarmonic3 = s => {
  if (s.length < 8) return null;
  const n = s.length;
  // Compute DFT magnitudes for periods 2..n/2
  const powers = [];
  for (let p = 2; p <= Math.floor(n / 2); p++) {
    let re = 0, im = 0;
    for (let t = 0; t < n; t++) {
      const angle = -2 * Math.PI * t / p;
      re += s[t] * Math.cos(angle);
      im += s[t] * Math.sin(angle);
    }
    powers.push({ p, mag: Math.sqrt(re*re + im*im), re, im });
  }
  powers.sort((a, b) => b.mag - a.mag);
  const top3 = powers.slice(0, 3);
  // Extrapolate each harmonic one step ahead and average
  let sum = 0, wSum = 0;
  for (const { p, re, im, mag } of top3) {
    const phase = Math.atan2(im, re);
    const predicted = mag / n * Math.cos(2 * Math.PI * n / p + phase) * 2;
    sum += (s[s.length-1] + predicted) * mag;
    wSum += mag;
  }
  return wSum > 0 ? M.mod(Math.round(sum / wSum)) : null;
};
```

---

### NEW ALGO 3 — `SparseTransitionGraph`
Builds a decade-bucket (0–9, 10–19…) transition graph weighted by recency decay. Predicts the next decade, then returns the historical mean within that decade. More robust than exact-value Markov on short series.

```js
A.SparseTransitionGraph = s => {
  if (s.length < 6) return null;
  const bucket = v => Math.floor(M.mod(v) / 10);
  const G = {}; // G[from][to] = weight
  for (let i = 1; i < s.length; i++) {
    const age = s.length - 1 - i;
    const from = bucket(s[i-1]);
    const to = bucket(s[i]);
    if (!G[from]) G[from] = {};
    G[from][to] = (G[from][to] || 0) + Math.exp(-age * 0.18);
  }
  const cur = bucket(s[s.length - 1]);
  const edges = G[cur] || {};
  // Prune weak edges
  const threshold = 0.25;
  let bestTo = -1, bestW = -1;
  for (const [to, w] of Object.entries(edges)) {
    if (w >= threshold && w > bestW) { bestW = w; bestTo = parseInt(to); }
  }
  if (bestTo < 0) return null;
  // Return mean of historical values in that decade
  const inDec = s.filter(v => bucket(v) === bestTo);
  return inDec.length ? M.mod(Math.round(M.mean(inDec))) : M.mod(bestTo * 10 + 5);
};
```

---

### NEW ALGO 4 — `CUSUMChangePoint`
Detects when the series mean has shifted using the CUSUM statistic. After detecting a change-point, uses ONLY post-shift data for its prediction (via frequency mode). Ignores stale pre-shift distribution.

```js
A.CUSUMChangePoint = s => {
  if (s.length < 8) return null;
  const mu0 = M.mean(s.slice(0, Math.floor(s.length / 2)));
  const sigma = Math.max(M.std(s) || 5, 3);
  const k = 0.5;
  const h = 4;
  let Sp = 0, Sn = 0;
  let cpIdx = 0;
  for (let i = Math.floor(s.length / 2); i < s.length; i++) {
    Sp = Math.max(0, Sp + (s[i] - mu0) / sigma - k);
    Sn = Math.max(0, Sn - (s[i] - mu0) / sigma - k);
    if (Sp > h || Sn > h) { cpIdx = i; Sp = 0; Sn = 0; }
  }
  const useSince = cpIdx > 0 ? cpIdx : 0;
  const recent = s.slice(useSince);
  if (recent.length < 2) return null;
  const freq = {};
  recent.forEach((v, i) => {
    const w = Math.exp(-(recent.length - 1 - i) * 0.1);
    freq[v] = (freq[v] || 0) + w;
  });
  return M.mod(parseInt(Object.entries(freq).sort((a,b) => b[1]-a[1])[0][0]));
};
```

---

### NEW ALGO 5 — `FibonacciRetracement`
In the bounded [0,99] space, detects if values tend to cluster at Fibonacci ratio levels of the recent high/low range. Projects the next Fibonacci level from the current value's position.

```js
A.FibonacciRetracement = s => {
  if (s.length < 8) return null;
  const w = s.slice(-Math.min(s.length, 16));
  const hi = Math.max(...w);
  const lo = Math.min(...w);
  const range = hi - lo;
  if (range < 5) return null;
  const ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0];
  const levels = ratios.map(r => lo + range * r);
  const cur = s[s.length - 1];
  // Find which level current value is closest to
  let nearestIdx = 0, nearestDist = Infinity;
  levels.forEach((lv, i) => {
    const d = Math.abs(cur - lv);
    if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
  });
  // Project to the next Fibonacci level (in the direction of recent trend)
  const trend = s[s.length-1] - s[s.length - Math.min(4, s.length)];
  const nextIdx = trend >= 0
    ? Math.min(nearestIdx + 1, levels.length - 1)
    : Math.max(nearestIdx - 1, 0);
  return M.mod(Math.round(levels[nextIdx]));
};
```

---

### NEW ALGO 6 — `BayesianDirichlet`
Maintains a Dirichlet posterior over all 100 values. Each observation increments the count for that value (with recency weighting). Prediction = mode of the posterior. More statistically principled than `Sticky` or `FreqDecay`.

```js
A.BayesianDirichlet = s => {
  if (s.length < 4) return null;
  const alpha = new Array(100).fill(0.5); // weak uniform prior
  s.forEach((v, i) => {
    const age = s.length - 1 - i;
    const w = Math.exp(-age * 0.08) + 0.1;
    alpha[M.mod(v)] += w;
  });
  let bestVal = 0, bestAlpha = -1;
  for (let i = 0; i < 100; i++) {
    if (alpha[i] > bestAlpha) { bestAlpha = alpha[i]; bestVal = i; }
  }
  return bestVal;
};
```

---

### NEW ALGO 7 — `AdaptiveForgettingRLS`
Recursive Least Squares with an adaptive forgetting factor λ. When recent prediction errors spike (regime change), λ drops to rapidly discount old data. When errors are low, λ rises to stabilise.

```js
A.AdaptiveForgettingRLS = s => {
  if (s.length < 6) return null;
  // AR(1) model: y[t] = theta * y[t-1] + noise
  let theta = 0.5;
  let P = 1.0;
  let lambda = 0.97;
  const lambda_min = 0.90, lambda_max = 0.99;
  for (let t = 1; t < s.length; t++) {
    const x = s[t-1];
    const y = s[t];
    const yhat = theta * x;
    const err = y - yhat;
    // Adapt lambda based on error magnitude
    const errNorm = Math.abs(err) / (Math.max(M.std(s.slice(0, t)), 1));
    lambda = Math.max(lambda_min, Math.min(lambda_max, lambda - 0.01 * errNorm + 0.005));
    // RLS update
    const denom = lambda + P * x * x;
    const K = P * x / denom;
    theta = theta + K * err;
    P = (P - K * x * P) / lambda;
  }
  return M.mod(Math.round(theta * s[s.length - 1]));
};
```

---

### NEW ALGO 8 — `EchoStateNetwork`
A fixed random reservoir (N=12 nodes) whose output weights are trained with ridge regression. Captures non-linear temporal dependencies that Markov and AR models fundamentally cannot.

```js
A.EchoStateNetwork = s => {
  if (s.length < 12) return null;
  const N = 12;
  const spectralRadius = 0.9;
  // Deterministic pseudo-random reservoir (seeded by col length for reproducibility)
  const seed = s.length * 31 + N;
  const rand = (i) => (Math.sin(seed * 9301 + i * 49297 + 233720) * 0.5 + 0.5) * 2 - 1;
  const W = Array.from({length:N}, (_, i) => Array.from({length:N}, (_, j) => rand(i*N+j) * spectralRadius));
  const Win = Array.from({length:N}, (_, i) => rand(i + N*N) * 0.5);
  // Collect reservoir states
  const states = [];
  let r = new Array(N).fill(0);
  for (let t = 0; t < s.length - 1; t++) {
    const x = s[t] / 99;
    r = r.map((_, i) =>
      Math.tanh(W[i].reduce((sum, w, j) => sum + w * r[j], 0) + Win[i] * x)
    );
    states.push([...r]);
  }
  // Ridge regression: solve Wout such that Wout · state[t] ≈ s[t+1]/99
  const targets = s.slice(1).map(v => v / 99);
  const ridge = 1e-4;
  // Simplified: use dot product correlation as readout weights
  const Wout = r.map((_, i) => {
    const num = states.reduce((sum, st, t) => sum + st[i] * targets[t], 0);
    const den = states.reduce((sum, st) => sum + st[i]*st[i], 0) + ridge;
    return num / den;
  });
  const prediction = r.reduce((sum, ri, i) => sum + Wout[i] * ri, 0);
  return M.mod(Math.round(prediction * 99));
};
```

---

## PART 3 — IMPROVE EXISTING ALGORITHMS

---

### IMPROVE: `A.MarkovChain` — add recency-decayed transition weights
Currently every transition is counted equally. Weight recent transitions more heavily so the model adapts faster to distribution shifts.

```js
// In MarkovChain, change the transition count line from:
trans[from][to] = (trans[from][to] || 0) + 1;

// To:
const age = s.length - 1 - i;
trans[from][to] = (trans[from][to] || 0) + Math.exp(-age * 0.08);
```

---

### IMPROVE: `A.DFTPeriod` — guard division by zero when all values are identical
If all series values are the same, the DFT magnitudes are all zero and the argmax returns period=2 always. Add a variance guard:

```js
// At the top of DFTPeriod:
if (M.std(s) < 0.5) return s[s.length - 1]; // flat series — repeat last
```

---

### IMPROVE: `A.RunLength` — clamp run prediction to [0,99]
The predicted `next = last +/- runLen * step` can exceed [0,99] if a run is long. Wrap with `M.mod`:

```js
// BEFORE
return last + (goingUp ? runLen * step : -runLen * step);

// AFTER
return M.mod(last + (goingUp ? runLen * step : -runLen * step));
```

---

### IMPROVE: `A.BounceDetect` — use circular distance for level proximity
Currently uses linear `Math.abs(cur - level)`. For a circular [0,99] space, a value of 99 is 1 away from 0, not 99 away.

```js
// BEFORE
const dist = Math.abs(cur - level);

// AFTER
const dist = M.cd(cur, level); // M.cd is already the circular distance
```

---

### IMPROVE: `A.ExpSmooth` — adaptive alpha based on recent volatility
Fixed alpha=0.3 is too slow for volatile regimes and too fast for stable ones. Adjust dynamically:

```js
// Compute volatility of last 8 values, map to alpha range [0.15, 0.55]
const recent = s.slice(-8);
const vol = M.std(recent);
const alpha = Math.max(0.15, Math.min(0.55, 0.15 + vol / 30));
// Then use alpha in the existing EMA loop
```

---

## PART 4 — NEW FEATURES (add to the UI and state)

---

### FEATURE 1 — Storage meter in Data/Settings tab
Add a small storage indicator showing current localStorage usage vs the 5MB limit. Show a warning when usage exceeds 70%. Block new row saves at 90% and show an export prompt.

```js
// Add this utility:
function getStorageUsageKB() {
  let total = 0;
  try {
    for (const key of Object.keys(localStorage)) {
      total += (localStorage.getItem(key) || '').length * 2; // UTF-16
    }
  } catch(e) {}
  return Math.round(total / 1024);
}

// In the UI, render a storage bar:
const usageKB = getStorageUsageKB();
const pct = Math.min(100, (usageKB / 5120) * 100);
const barColor = pct > 90 ? '#ef4444' : pct > 70 ? '#f97316' : '#34d399';
// <div> Storage: {usageKB}KB / 5120KB
// <div style={{width: pct+'%', background: barColor, height:4}} />
```

When `saveS` catches a `QuotaExceededError`, call `setMsg` with a visible error toast instead of only `console.error`.

---

### FEATURE 2 — Failsafe indicator in prediction column header
When `predictCol` returns `failsafeFired: true` (from Bug Fix 8 above), show a small `⚡` icon next to the column name in the results panel, with a tooltip: "Throttled — device was under load, only 1 candidate shown".

---

### FEATURE 3 — Per-column hot/cold streak indicator
Track consecutive exact hits and misses per column from `accLog`. Show 🔥 (3+ consecutive hits) or 🧊 (3+ consecutive misses) badge next to each column name in the prediction results.

```js
// Add this helper:
function getStreaks(accLog, col) {
  const recent = (accLog || []).slice(-10).map(e => e?.[col]);
  let hits = 0, misses = 0;
  for (let i = recent.length - 1; i >= 0; i--) {
    if (recent[i] === null || recent[i] === undefined) break;
    if (recent[i] === true) { if (misses > 0) break; hits++; }
    else { if (hits > 0) break; misses++; }
  }
  return { hits, misses };
}
// In column header: streak.hits >= 3 ? '🔥' : streak.misses >= 3 ? '🧊' : ''
```

---

### FEATURE 4 — Named weight snapshots before every import
Before any weight import (`importWeightsFile`), auto-save a named backup in localStorage keyed as `ape_backup_{timestamp}`. Add a "Restore last backup" button in the Data tab.

```js
// Before the import state update:
const backupKey = `ape_backup_${Date.now()}`;
try {
  localStorage.setItem(backupKey, localStorage.getItem('ape_state_v2') || '');
  // Keep only last 3 backups — prune older ones
  const backupKeys = Object.keys(localStorage)
    .filter(k => k.startsWith('ape_backup_'))
    .sort();
  if (backupKeys.length > 3)
    backupKeys.slice(0, backupKeys.length - 3).forEach(k => localStorage.removeItem(k));
} catch(e) {}
```

---

### FEATURE 5 — Regime change alert banner
When `classifyContext()` returns `CHAOTIC` or `VOLATILE` for 3+ consecutive predictions on the same column, emit a persistent syslog alert with visual emphasis. Prompt the user to run auto-train.

```js
// Track regime history in state (add to S):
regimeHistory: { A:[], B:[], C:[], D:[], E:[], F:[], G:[] }

// After each prediction, push the regime to regimeHistory[col].slice(-5)
// If last 3 are all 'chaotic' or 'volatile':
const lastThree = regimeHistory[col].slice(-3);
if (lastThree.length === 3 && lastThree.every(r => r === 'chaotic' || r === 'volatile')) {
  syslog(`⚠️ ${COL_NAMES[col]}: 3 consecutive chaotic regimes — consider Auto-Train`, 'warn');
}
```

---

### FEATURE 6 — Rolling accuracy sparkline per column
In the Results/Analysis view, add a small inline sparkline (SVG path) showing exact hit rate per column over the last 20 sessions from `accLog`. Green dot = hit, red = miss, grey = no data.

```js
// For each col, extract from accLog:
const sparkData = (accLog || []).slice(-20).map(e =>
  e?.[col] === true ? 1 : e?.[col] === false ? 0 : null
);
// Render as a row of small colored dots:
sparkData.map((v, i) => (
  <span key={i} style={{
    display:'inline-block', width:5, height:5, borderRadius:'50%', margin:1,
    background: v===1 ? '#34d399' : v===0 ? '#f87171' : '#1a1e35'
  }}/>
))
```

---

## PART 5 — GENERAL CODE QUALITY

1. **Add `runPredict` concurrency guard:** Use a `useRef(false)` flag `predictingRef`. At start of `runPredict`, if `predictingRef.current === true`, return early. Set to `true` before compute, `false` in `finally`. Prevents multiple simultaneous predict runs on rapid taps.

2. **Wrap all `runPredict` compute in `requestIdleCallback` with a 500ms timeout fallback** so it doesn't block the UI thread on mobile:
```js
const rIC = window.requestIdleCallback || (cb => setTimeout(() => cb({timeRemaining:()=>50}), 0));
rIC(() => { /* existing predict compute */ }, { timeout: 500 });
```

3. **Remove the production `console.log("Algo count:", ALGO_COUNT)` line.** Replace with:
```js
if (process.env.NODE_ENV === 'development') console.debug('[APE] Algo count:', ALGO_COUNT);
```

4. **`_TC.bumpVer()` should call `_TC.clear()`** on every bump to prevent stale cache memory accumulation across long sessions:
```js
bumpVer() { this._ver = (this._ver + 1) % 1e9; this.clear(); },
```

---

## CONSTRAINTS — MUST FOLLOW

- Do NOT change the shape of the `S` state object without providing a migration in `loadS` that fills missing keys with defaults.
- Do NOT change any key names in localStorage.
- Do NOT remove any existing algorithm from the `A` object — only add or improve.
- Do NOT change the `M` helper object — only add methods if needed.
- Do NOT alter the `COLS`, `COL_NAMES`, or `pad2` definitions.
- Every new algorithm must return either `M.mod(integer)` in [0,99] or `null` (never `undefined`, never a float).
- New state keys added to `S` must have a default in the initial state object AND in the `loadS` merge.
- After all changes, run a mental compilation check: no undefined variable references, no missing closing braces, no duplicate `const` declarations in the same scope.
- Keep the file as a **single `.jsx`** — do not split into multiple files.

---

## DELIVERY

After applying all changes:
1. State clearly how many bugs were fixed, how many algos were added/improved, how many features were added.
2. List any change you could NOT apply safely and why.
3. Do a final scan for accidental duplicate variable declarations introduced during edits.
