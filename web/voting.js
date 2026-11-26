/* N-best edit voting / confidence-based filtering — browser port of implementation/gecalgo/voting.py
 * and edits.py (weights, support, overlap resolution, edit application).
 *
 * Alignment and edit extraction are NOT redone here: web/build_data.py runs the Python aligner once and
 * stores each beam's edits, because they do not depend on tau / T / mode. Everything below is the part
 * that does, and it mirrors the Python line for line (same summation order, same stable tie-breaking),
 * so outputs are byte-identical; the page verifies this against hashes of the historical outputs.
 */
(function (root) {
  "use strict";

  function hypothesisWeights(scores, temperature, mode) {
    const n = scores.length;
    if (n === 0) return [];
    if (mode === "uniform") return scores.map(() => 1.0 / n);
    if (mode === "softmax") {
      if (!(temperature > 0)) throw new Error("temperature must be > 0");
      let mx = -Infinity;
      for (const s of scores) if (s > mx) mx = s;
      const ex = scores.map((s) => Math.exp((s - mx) / temperature));
      let z = 0;
      for (const x of ex) z += x;
      return ex.map((x) => x / z);
    }
    throw new Error("unknown weight mode " + mode);
  }

  const keyOf = (e) => e[0] + "," + e[1] + "|" + e[2].join("\u0001");

  function overlaps(a, b) {
    const aIns = a.start === a.end, bIns = b.start === b.end;
    if (aIns && bIns) return a.start === b.start;
    if (aIns) return b.start <= a.start && a.start < b.end;
    if (bIns) return a.start <= b.start && b.start < a.end;
    return a.start < b.end && b.start < a.end;
  }

  function resolveOverlaps(scored) {
    // Python: sorted(key=(-support, start, length)) — Array.prototype.sort is stable, like sorted().
    const order = scored.slice().sort((x, y) =>
      (y.support - x.support) || (x.start - y.start) || ((x.end - x.start) - (y.end - y.start)));
    const chosen = [];
    for (const e of order) if (!chosen.some((c) => overlaps(e, c))) chosen.push(e);
    return chosen.sort((x, y) => (x.start - y.start) || (x.end - y.end));
  }

  function applyEdits(src, edits) {
    const out = src.slice();
    const desc = edits.slice().sort((x, y) => (y.start - x.start) || (y.end - x.end));
    for (const e of desc) out.splice(e.start, e.end - e.start, ...e.cor);
    return out;
  }

  /** Step 1 (independent of tau and mode): weighted support of every distinct edit, in first-seen order.
   *  sentence = {src, top1, scores, beams: [[text, [[start, end, cor[]], ...]], ...]} */
  function prepareSentence(sentence, temperature, weightMode) {
    const { src, scores, beams } = sentence;
    const weights = hypothesisWeights(scores, temperature, weightMode);
    const byKey = new Map();         // insertion-ordered, like the Python dict
    for (let k = 0; k < beams.length && k < weights.length; k++) {
      const seen = new Set();
      for (const e of beams[k][1]) {
        const key = keyOf(e);
        if (seen.has(key)) continue;
        seen.add(key);
        let c = byKey.get(key);
        if (!c) {
          c = { start: e[0], end: e[1], cor: e[2], orig: src.slice(e[0], e[1]), support: 0, key, beams: [] };
          byKey.set(key, c);
        }
        c.support += weights[k];
        c.beams.push(k);
      }
    }
    const top = new Set(beams.length ? beams[0][1].map(keyOf) : []);
    const candidates = [...byKey.values()];
    for (const c of candidates) c.inTop1 = top.has(c.key);
    return { src, weights, candidates };
  }

  /** Step 2 (cheap): threshold at tau, resolve overlaps, apply. */
  function selectEdits(prep, tau, restrictToTop1) {
    if (!(tau >= 0 && tau <= 1)) throw new Error("tau must be in [0, 1]");
    const scored = prep.candidates.filter((c) => (!restrictToTop1 || c.inTop1) && c.support >= tau - 1e-12);
    const kept = resolveOverlaps(scored);
    return { tokens: applyEdits(prep.src, kept), kept, prep };
  }

  function consensusCorrection(sentence, tau, temperature, weightMode, restrictToTop1) {
    return selectEdits(prepareSentence(sentence, temperature, weightMode), tau, restrictToTop1);
  }

  /** Prepared dataset for one weighting; reuse it across every tau and both modes. */
  function prepareDataset(sentences, temperature, weightMode) {
    return sentences.map((s) => prepareSentence(s, temperature, weightMode));
  }

  /** Run one configuration over a whole dataset. mode: "vote" | "filter" | "top1".
   *  Pass `prepared` (from prepareDataset with the same weighting) to skip step 1. */
  function runConfig(sentences, cfg, prepared) {
    if (cfg.mode === "top1") return sentences.map((s) => ({ tokens: s.top1, kept: null }));
    const prep = prepared || prepareDataset(sentences, cfg.temperature, cfg.weightMode);
    return prep.map((p) => selectEdits(p, cfg.tau, cfg.mode === "filter"));
  }

  // Same filename rules as experiments/run_voting_filtering.py (fmt_num / fmt_tau).
  const fmtNum = (x) => String(Number(x.toPrecision(6)));
  const fmtTau = (t) => (Math.round(t * 10) / 10 === t ? t.toFixed(1) : fmtNum(t));
  function configName(cfg) {
    if (cfg.mode === "top1") return "coedit__top1";
    const label = cfg.weightMode === "uniform" ? "unif" : "soft" + fmtNum(cfg.temperature);
    return `coedit__${cfg.mode}_t${fmtTau(cfg.tau)}_${label}`;
  }

  /** 32-bit FNV-1a over UTF-16 code units — identical to fnv1a() in web/build_data.py. */
  function fnv1a(text) {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  }

  const outputText = (results) => results.map((r) => r.tokens.join(" ")).join("\n") + "\n";

  root.NBestVoting = { hypothesisWeights, prepareSentence, selectEdits, consensusCorrection, prepareDataset,
                       resolveOverlaps, applyEdits, overlaps, runConfig, configName, fnv1a, outputText };
})(typeof window !== "undefined" ? window : globalThis);
