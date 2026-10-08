/* data.js -- fills the SPA pages from /api/dashboard (which reads MCS_health.csv). */
(function () {
  var PAGES = { overview: 'page-overview', health: 'page-health-score-fuzzy-decision', urgency: 'page-competitive-urgency', battle: 'page-battle-forecast' };
  var D = null, mtime = null;
  var $ = function (s) { return document.querySelector(s); };
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var num = function (v, d) { return v == null ? '–' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d || 0, maximumFractionDigits: d || 0 }); };
  var pc = function (v, d) { return v == null ? '–' : (v > 0 ? '+' : '') + (v * 100).toFixed(d == null ? 1 : d) + '%'; };
  var tone = function (v) { return v == null ? 'muted' : v >= 70 ? 'success' : v >= 45 ? 'warning' : 'danger'; };
  var sgn = function (v) { return v == null ? 'muted' : v >= 0 ? 'success' : 'danger'; };
  var rp = function (v) { return v == null ? '–' : 'Rp ' + num(v, v < 100 ? 2 : 0); };
  var pill = function (t, c) { return '<span class="pill" style="background:var(--surface-2);color:var(--' + c + ');border:1px solid var(--border);">' + esc(t) + '</span>'; };
  var bar = function (v, c) { return '<div class="bar-track"><div class="bar-fill" style="width:' + Math.max(0, Math.min(100, v || 0)) + '%;background:var(--' + c + ');"></div></div>'; };
  var card = function (icon, title, right, body, st) {
    return '<div class="card" style="' + (st || '') + '"><div class="card-header"><span class="material-symbols-outlined text-primary">' + icon + '</span><span class="card-title">' + title + '</span>' +
      (right ? '<span style="margin-left:auto;">' + right + '</span>' : '') + '</div>' + body + '</div>';
  };
  var lab = function (t) { return '<div class="mono text-muted" style="font-size:9px;text-transform:uppercase;">' + t + '</div>'; };
  var head = function (kicker, title, sub) {
    return '<div><div class="mono text-primary" style="font-size:11px;text-transform:uppercase;">' + kicker + '</div><h1 class="page-title">' + title + '</h1>' + (sub ? '<p class="page-sub">' + sub + '</p>' : '') + '</div>';
  };
  var note = function (icon, html) {
    return '<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:.75rem;margin-top:1rem;display:flex;gap:8px;"><span class="material-symbols-outlined text-primary" style="font-size:18px;">' + icon + '</span><p class="text-soft" style="font-size:12px;margin:0;">' + html + '</p></div>';
  };
  var SHORT = { sentiment: 'Sent.', trend: 'Trend', stability: 'Stab.', xgboost: 'Vol.', breakout: 'Brk.' };
  var STC = { 'AVOID': 'danger', 'WATCH': 'warning', 'EXPLORE': 'text-soft', 'STRONG CASE': 'success', 'HIGH CONVICTION': 'primary' };
  var stc = function (s) { return STC[s] || 'muted'; };
  var name = function (o) { return esc(o.name || o.short); };

  /* ---------- sentiment badge: per company, pushed by website/sentiment_trigger.py -> /api/sentiment ----------
     The server swaps a company's live sentiment into its latest day and re-scores its health, so the badge value comes
     with the dashboard JSON (focus.sentiment / peers[i].sentiment: {score,count,live,mock,...}). We only poll
     /api/sentiment to notice that something was pushed and then reload the dashboard. */
  var SENT = { updated: null }, sentReady = false;
  var SENT_RGB = { neg: [239, 68, 68], neu: [245, 197, 24], pos: [74, 222, 128] };   // = --danger / --primary / --success
  function sentColor(s) {   // -1 red -> 0 yellow -> +1 green, blended smoothly in between
    s = Math.max(-1, Math.min(1, s || 0));
    var a = s < 0 ? SENT_RGB.neg : SENT_RGB.neu, b = s < 0 ? SENT_RGB.neu : SENT_RGB.pos, t = s < 0 ? s + 1 : s;
    return 'rgb(' + a.map(function (v, i) { return Math.round(v + (b[i] - v) * t); }).join(',') + ')';
  }
  var CUR = null;   // peer currently shown in the Active Focus card
  function sentBadge() {
    var cs = CUR && CUR.sentiment ? CUR.sentiment : {}, live = !!cs.live, s = cs.score != null ? cs.score : null;
    var word = s == null ? 'No data' : s >= 0.25 ? 'Positive' : s <= -0.25 ? 'Negative' : 'Neutral';
    var r = s == null ? null : Math.round(s * 100) / 100;
    var tip = live ? 'LIVE: ' + cs.count + ' news item(s) from ' + (cs.source || 'the news file') + ', scored ' + new Date(cs.updated * 1000).toLocaleTimeString() + '. Replaces today\'s sentiment for ' + (CUR.short || 'this company') + ' and drives its health score.'
      : s != null ? 'From MCS_health.csv: ' + (cs.count != null ? cs.count + ' article(s), ' : '') + 'as of ' + String(cs.date).slice(0, 10) + '. Run website/sentiment_trigger.py to score new news.'
      : cs.mock ? 'No real news sentiment for this company yet: MCS_health.csv only holds the pipeline\'s random mock data, which is ignored by the health score. Run website/sentiment_trigger.py.'
      : 'No sentiment in MCS_health.csv for this company';
    var tag = live ? '<span style="color:var(--success);">● LIVE</span>' : '<span>DATA</span>';
    return '<div title="' + esc(tip) + '" style="text-align:right;line-height:1.15;white-space:nowrap;">' + lab('Sentiment Score · ' + tag) +
      '<div class="mono" style="color:' + (s == null ? 'var(--muted)' : sentColor(s)) + ';font-weight:700;font-size:16px;">' + (r == null ? '–' : (r > 0 ? '+' : '') + r.toFixed(2)) +
      ' <span style="font-size:9px;font-weight:600;">' + word + '</span></div></div>';
  }
  function paintSent(flash) {
    var el = document.getElementById('ms-sent'); if (!el) return;
    el.innerHTML = sentBadge();
    if (flash) { el.classList.remove('sent-flash'); void el.offsetWidth; el.classList.add('sent-flash'); }
  }
  function loadSent() {
    return fetch('/api/sentiment', { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error) return;
      var changed = j.updated !== SENT.updated; SENT = j;
      // a push changes that company's health score on the server -> reload so every number follows (no page refresh)
      if (changed && sentReady && D) load(D.focus.symbol).then(function () { paintSent(true); });
      sentReady = true;
    }).catch(function () { });
  }

  function setPages(html) { Object.keys(PAGES).forEach(function (k) { var e = document.getElementById(PAGES[k]); if (e) e.innerHTML = typeof html === 'function' ? html(k) : html; }); }
  function msg(t, sub) { return '<div class="card" style="text-align:center;padding:3rem;"><div class="mono text-primary" style="font-size:12px;">' + t + '</div><p class="text-soft" style="font-size:13px;">' + (sub || '') + '</p></div>'; }

  /* ---------- pages ---------- */
  function overview(f, m) {
    var top = D.peers.slice(0, 3);
    var pills = f.factors.map(function (x) { return '<span class="pill mono" style="background:var(--surface-2);border:1px solid var(--border);">' + esc(SHORT[x.key] || x.key) + ' <span class="text-' + tone(x.score) + '">' + num(x.score) + '</span></span>'; }).join('');
    var c = D.chart.lines[0].values.filter(function (v) { return v != null; });
    var pts = c.map(function (v, i) { var mn = Math.min.apply(0, c), mx = Math.max.apply(0, c) || 1; return (5 + i * 230 / Math.max(1, c.length - 1)).toFixed(1) + ',' + (90 - (v - mn) / ((mx - mn) || 1) * 80).toFixed(1); }).join(' ');
    var sig = signals(f).map(function (s) { return '<div style="display:flex;gap:8px;background:var(--surface-container-lowest,#161616);border:1px solid var(--border);border-radius:8px;padding:8px;"><span class="material-symbols-outlined text-' + s[0] + '" style="font-size:18px;">' + s[1] + '</span><span class="text-soft" style="font-size:13px;"><strong style="color:#fff;">' + s[2] + ':</strong> ' + s[3] + '</span></div>'; }).join('');
    return '<section style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border);"><div><div class="mono text-primary" style="font-size:11px;text-transform:uppercase;margin-bottom:4px;">Mikir Sekali Terminal</div>' +
      '<h1 class="page-title">' + name(f) + ' <span class="text-muted" style="font-weight:400;font-size:20px;">(' + f.short + ')</span></h1><div class="mono text-muted" style="font-size:12px;margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;"><span>Sector: <span style="color:#fff;">' + esc(f.sector || 'Transportation') + '</span></span><span>•</span><span>Close: <span style="color:#fff;">' + rp(f.close) + '</span> <span class="text-' + sgn(f.ret) + '">(' + pc(f.ret, 2) + ')</span></span><span>•</span><span>Data as of ' + f.date + '</span></div></div>' +
      '<span class="pill" style="background:rgba(74,222,128,.1);color:var(--success);border:1px solid rgba(74,222,128,.3);align-self:flex-start;">Input Coverage: ' + num((f.coverage || 0) * 100) + '%</span></section>' +
      '<div class="grid" style="grid-template-columns:repeat(4,1fr);">' +
      '<div class="card" style="min-height:280px;"><div class="card-header"><span class="card-title">Structural Health</span>' + pill(f.stance || '–', stc(f.stance)) + '</div><div style="text-align:center;"><div style="font-size:40px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">' + num(f.health_predict, 1) + '<span style="font-size:16px;color:var(--muted);">/100</span></div><div class="mono text-muted" style="font-size:10px;">Predicted Health Score</div></div><p class="text-soft" style="font-size:12px;margin-top:.75rem;">Real-data score today: <strong style="color:#fff;">' + num(f.health_real, 1) + '</strong>. 30-day drift: <span class="text-' + sgn(f.drift30) + '">' + (f.drift30 == null ? '–' : (f.drift30 >= 0 ? '+' : '') + f.drift30.toFixed(1) + ' pts') + '</span>.</p><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:.75rem;padding-top:.75rem;border-top:1px solid var(--border);">' + pills + '</div></div>' +
      '<div class="card" style="min-height:280px;"><div class="card-header"><span class="card-title">Stance Engine</span><span class="mono text-primary" style="font-size:11px;margin-left:auto;">Non-Binary</span></div>' + lab('Inferred State (real → predicted)') + '<div style="font-weight:600;">' + esc(f.stance_real || '–') + ' <span class="text-primary">→</span> <span class="text-' + stc(f.stance) + '">' + esc(f.stance || '–') + '</span></div>' + spectrumStats(f) + spectrumBar(f) + '</div>' +
      '<div class="card" style="min-height:280px;"><div class="card-header"><span class="card-title">Threat Velocity</span>' + pill(D.peers.length + ' peers', 'primary') + '</div><div style="font-weight:600;font-size:14px;margin-bottom:.5rem;">Top peers by urgency</div><div style="display:flex;flex-direction:column;gap:8px;">' +
      top.map(function (p, i) { var c = ['primary', 'warning', 'muted'][i]; return '<div style="background:var(--surface-2);padding:8px;border-radius:6px;border:1px solid var(--border);"><div style="display:flex;justify-content:space-between;font-size:13px;"><span>' + esc(p.short) + '</span><span class="text-' + c + ' mono" style="font-weight:700;">' + num(p.urgency) + '%</span></div>' + bar(p.urgency, c) + '</div>'; }).join('') +
      '</div><a class="btn" data-page="competitive-urgency" href="#/competitive-urgency" style="width:100%;justify-content:center;margin-top:.75rem;">Open Competitor Map</a></div>' +
      '<div class="card" style="min-height:280px;"><div class="card-header"><span class="card-title">Multi-Horizon Track</span><span class="mono text-primary" style="font-size:11px;margin-left:auto;">~90d Close</span></div><div style="display:flex;justify-content:space-between;font-size:14px;font-weight:600;margin-bottom:6px;"><span>30-Day Momentum</span><span class="text-' + sgn(f.momentum) + '">' + pc(f.momentum) + '</span></div><svg viewBox="0 0 240 100" style="width:100%;height:100px;"><polyline points="' + pts + '" fill="none" stroke="#F5C518" stroke-width="2.5"/></svg><a class="btn" data-page="battle-forecast" href="#/battle-forecast" style="width:100%;justify-content:center;margin-top:.75rem;">See Full Battle Forecast</a></div></div>' +
      '<div class="card" style="border-left:3px solid var(--primary);"><div class="card-header"><span class="material-symbols-outlined text-primary">psychology</span><span class="card-title">Signal Synthesis</span>' + pill('Rule-Based', 'primary') + '</div><div style="display:flex;flex-direction:column;gap:8px;">' + sig + '</div></div>';
  }
  function signals(f) {
    var s = [], v = f.volume, sr = f.sr, out = [];
    if (f.breakout != null) out.push([f.breakout >= .5 ? 'success' : 'warning', 'deployed_code', 'Breakout Probability', pc(f.breakout, 0).replace('+', '') + ' chance Close exceeds its 1-year (252-session) high within 21 sessions (regime: ' + esc(f.regime || 'unknown') + ').']);
    else out.push(['muted', 'deployed_code', 'Breakout Probability', 'No breakout model available for ' + f.short + ' (too little history, or the trained model did not pass its hold-out check); scored without the breakout component.']);
    if (sr.sr_support != null && sr.sr_resistance != null) out.push(['primary', 'straighten', 'Support / Resistance', 'Support ' + rp(sr.sr_support) + ' (' + pc(sr.sr_dist_to_support_pct == null ? null : -Math.abs(sr.sr_dist_to_support_pct)) + ' away) and resistance ' + rp(sr.sr_resistance) + ' (' + pc(sr.sr_dist_to_resistance_pct) + ' away).']);
    else out.push(['muted', 'straighten', 'Support / Resistance', 'Not enough history yet to compute support/resistance lines.']);
    if (sr.sr_break_up > 0) out.push(['success', 'trending_up', 'Resistance Break', 'Close broke above the resistance line (strength ' + num(sr.sr_break_up * 100) + '%): a positive LSTM-Hurst contribution to the health score.']);
    if (sr.sr_hit_support > 0) out.push(['warning', 'trending_down', 'Support Hit', 'Price reached the support line (strength ' + num(sr.sr_hit_support * 100) + '%): a negative LSTM-Hurst contribution to the health score.']);
    if (v.applicable && v.pred != null && v.vs_norm != null) out.push([v.vs_norm >= 0 ? 'success' : 'warning', 'bar_chart', 'Volume Outlook', 'Forecast next-day volume ' + num(v.pred) + ' is ' + pc(v.vs_norm, 0) + ' vs the model\'s usual forecast for this weekday (' + num(v.norm) + '): volume expected to ' + (v.vs_norm >= 0 ? 'rise' : 'fall') + '.']);
    else out.push(['muted', 'bar_chart', 'Volume Outlook', 'No outlook for the next session (an old-style model forecasting a weekend, or not enough forecast history yet). Left out of the health score.']);
    return out;
  }
  function spectrumStats(f) {
    var cell = function (l, v) { return '<div>' + lab(l) + '<div style="font-weight:600;">' + (v == null ? '–' : num(v * 100) + '%') + '</div></div>'; };
    return '<div class="grid" style="grid-template-columns:repeat(3,1fr);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:.5rem;margin:.75rem 0;text-align:center;">' + cell('Volume Fit', f.fit) + cell('Coverage', f.coverage) + cell('Stability', f.stability == null ? null : f.stability / 100) + '</div>';
  }
  function spectrumBar(f) {
    var p = f.health_predict == null ? 0 : f.health_predict;
    return '<div class="bar-track" style="background:linear-gradient(to right,#3a3939,#f5c518);position:relative;height:8px;"><div style="position:absolute;top:-3px;left:' + p + '%;width:6px;height:14px;background:#fff;border-radius:3px;box-shadow:0 0 8px rgba(245,197,24,.9);"></div></div><div style="display:flex;justify-content:space-between;font-size:9px;color:var(--muted);margin-top:4px;"><span>Avoid</span><span class="text-primary" style="font-weight:700;">' + (p / 100).toFixed(2) + '</span><span>Conviction</span></div>';
  }

  function health(f) {
    var hp = f.health_predict || 0;
    var rows = f.factors.map(function (x) { return '<tr><td><strong>' + esc(x.label) + '</strong><br/><span class="text-muted" style="font-size:11px;">Weight ' + num(x.weight * 100) + '% → ' + num(x.points, 1) + ' pts</span></td><td style="text-align:right;" class="text-' + tone(x.score) + ' mono">' + num(x.score) + '</td><td class="text-soft">' + esc(x.evidence) + '</td></tr>'; }).join('');
    var bars = f.factors.map(function (x) { return '<div><div style="display:flex;justify-content:space-between;font-size:12px;"><span>' + esc(x.label) + '</span><span class="text-' + tone(x.score) + '" style="font-weight:700;">' + num(x.score) + '/100</span></div>' + bar(x.score, tone(x.score)) + '</div>'; }).join('');
    var sr = f.sr, cps = [
      ['Checkpoint Alpha', sr.sr_support != null ? 'Close breaks below support ' + rp(sr.sr_support) + '.' : 'Close drops more than 5% below the 30-day mean.', 'WATCH', 'warning'],
      ['Checkpoint Beta', 'Predicted health score falls below 40.', 'AVOID', 'danger'],
      ['Checkpoint Gamma', sr.sr_resistance != null ? 'Close clears resistance ' + rp(sr.sr_resistance) + ' on rising volume.' : 'The volume outlook stays below the model\'s usual forecast for a week.', 'RE-EVALUATE', 'primary']];
    return head('Valuation &amp; Decision Engine / Synthesis', 'Decision Confidence &amp; Health Score', 'Predicted health score for ' + name(f) + ', broken down into the exact fuzzy factors used by the pipeline.') +
      '<div class="grid" style="grid-template-columns:5fr 7fr;">' + card('vital_signs', 'Composite Health Index', '',
        '<div style="display:flex;align-items:center;gap:1rem;"><div style="width:140px;height:140px;border-radius:50%;background:conic-gradient(var(--primary) 0% ' + hp + '%, var(--surface-2) ' + hp + '% 100%);display:flex;align-items:center;justify-content:center;flex-shrink:0;"><div style="width:106px;height:106px;border-radius:50%;background:var(--surface);display:flex;flex-direction:column;align-items:center;justify-content:center;"><span style="font-size:36px;font-weight:700;font-family:\'JetBrains Mono\',monospace;">' + num(hp, 1) + '</span><span class="mono text-muted" style="font-size:9px;">/ 100 PTS</span></div></div><div>' + pill(f.stance || '–', stc(f.stance)) + '<p class="text-soft" style="font-size:13px;margin-top:8px;">Real-data score: <strong style="color:#fff;">' + num(f.health_real, 1) + '</strong> (no forecast inputs).</p><div class="text-' + sgn(f.drift30) + ' mono" style="font-size:12px;margin-top:6px;">30-day drift: ' + (f.drift30 == null ? '–' : (f.drift30 >= 0 ? '+' : '') + f.drift30.toFixed(1) + ' pts') + '</div></div></div><div style="margin-top:1rem;padding-top:.75rem;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:10px;">' + bars + '</div>') +
      card('account_tree', 'Stance Engine (Non-Binary Model)', '', '<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:1rem;">' + lab('Inferred Decision State') + '<div style="font-size:24px;font-weight:700;">' + esc(f.stance_real || '–') + ' <span class="text-primary">→</span> <span class="text-' + stc(f.stance) + '">' + esc(f.stance || '–') + '</span></div></div>' + spectrumStats(f) + '<div style="margin-top:1rem;"><div class="mono text-muted" style="font-size:11px;">Continuous Decision Spectrum (score / 100)</div><div style="margin-top:12px;">' + spectrumBar(f) + '</div><div style="display:grid;grid-template-columns:repeat(5,1fr);text-align:center;font-size:9px;margin-top:8px;" class="mono text-muted"><span class="text-danger">Avoid</span><span class="text-warning">Watch</span><span>Explore</span><span class="text-success">Strong Case</span><span style="color:#ffe08b;">High Conviction</span></div></div>' +
        note('info', '<strong style="color:#fff;">Volume Fit</strong> = 1 − MAPE of the volume forecast over the last 90 trading days (in-sample: the model was trained on this history).'), 'border-left:3px solid var(--primary);') + '</div>' +
      card('table_rows', 'Deep Vector Factor Ledger', '', '<div style="overflow-x:auto;"><table><thead><tr><th>Vector Factor</th><th style="text-align:right;">Score</th><th>Evidence</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
        note('info', 'Each factor is a 0–100 fuzzy score (50 = neutral). A factor with no signal today (mock sentiment, no forecast history yet, no breakout model) is left out and the remaining weights are renormalised, so the points always add up to the health score.' + (D && D.meta && D.meta.sentiment_mock ? ' Sentiment in this CSV is random mock data, so it is excluded until real BERT sentiment is supplied.' : ''))) +
      card('psychology_alt', 'Falsification Framework &amp; Decision Journal', '', '<p class="text-soft" style="font-size:13px;">Observable conditions that should force a stance review, derived from the latest support/resistance and score.</p><div class="grid" style="grid-template-columns:repeat(3,1fr);">' +
        cps.map(function (c) { return '<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:1rem;">' + lab(c[0]) + '<p style="font-size:13px;font-weight:500;margin:.5rem 0;">' + c[1] + '</p><div style="border-top:1px solid var(--border);padding-top:.5rem;display:flex;justify-content:space-between;font-size:11px;"><span class="text-muted">Forced Action:</span><span class="text-' + c[3] + '" style="font-weight:700;">→ ' + c[2] + '</span></div></div>'; }).join('') + '</div>' +
        '<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:1rem;margin-top:1rem;">' + lab('Personal Falsification Note') + '<textarea id="ms-note" rows="3" style="margin-top:6px;" placeholder="Write the condition that would make you change your mind…"></textarea></div>', 'border-left:3px solid var(--primary);');
  }

  function urgency(f) {
    var P = D.peers, sel = P[0];
    var led = P.map(function (p, i) {
      var c = p.urgency >= 75 ? 'primary' : p.urgency >= 55 ? 'warning' : 'text-soft';
      return '<div class="ms-peer" data-i="' + i + '" style="cursor:pointer;background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:10px;"><div style="display:flex;justify-content:space-between;"><span style="font-weight:600;">' + esc(p.short) + ' <span class="text-muted" style="font-weight:400;font-size:11px;">' + esc(p.name || '') + '</span></span>' + pill(p.label, c) + '</div><div class="grid" style="grid-template-columns:repeat(3,1fr);margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:13px;"><div>' + lab('Urgency') + '<span class="text-' + (p.urgency >= 55 ? 'primary' : 'soft') + '" style="font-weight:700;">' + num(p.urgency) + '%</span></div><div>' + lab('Momentum') + '<span class="text-' + sgn(p.momentum) + '">' + pc(p.momentum) + '</span></div><div>' + lab('Overlap') + (p.overlap == null ? '–' : num(p.overlap * 100) + '%') + '</div></div></div>';
    }).join('');
    var bub = P.map(function (p) {
      var x = 8 + (p.overlap || 0) * 84, y = 50 - Math.max(-1, Math.min(1, (p.momentum || 0) / 0.3)) * 42, sz = 24 + (p.health || 0) * 0.3;
      return '<div title="' + esc(p.short) + '" style="position:absolute;top:' + y + '%;left:' + x + '%;transform:translate(-50%,-50%);width:' + sz + 'px;height:' + sz + 'px;border-radius:50%;background:rgba(245,197,24,.2);border:1px solid var(--primary);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:9px;color:var(--primary);">' + esc(p.short) + '</div>';
    }).join('');
    return head('Market Intelligence Cockpit / Sector Dynamic', 'Competitive Urgency', 'Peers of ' + name(f) + ' ranked by pressure. Urgency = 40% peer health + 30% return correlation (overlap) + 30% 30-day momentum.') +
      '<div class="grid" style="grid-template-columns:4fr 5fr 3fr;">' + card('radar', 'Urgency Ledger', '', '<div style="display:flex;flex-direction:column;gap:8px;max-height:520px;overflow-y:auto;">' + led + '</div>') +
      card('bubble_chart', 'Pressure Bubble Map', '<span class="mono text-muted" style="font-size:10px;">Overlap vs Momentum</span>', '<div style="position:relative;width:100%;height:340px;background:var(--surface-container-lowest);border-radius:12px;border:1px solid var(--border);overflow:hidden;"><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border-strong);"></div><div style="position:absolute;top:50%;left:0;right:0;height:1px;background:var(--border-strong);"></div>' + bub + '<span style="position:absolute;top:6px;right:10px;font-size:9px;color:var(--muted);">Momentum ↑</span><span style="position:absolute;bottom:6px;right:10px;font-size:9px;color:var(--muted);">Return correlation →</span></div>') +
      '<div class="card" id="ms-focus"></div></div>';
  }
  function focusCard(p, f) {
    CUR = p;
    return '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">' + pill('Active Focus', 'primary') + '<div id="ms-sent" style="margin-left:auto;">' + sentBadge() + '</div></div>' +
      '<div style="display:flex;align-items:baseline;gap:8px;margin-top:8px;"><h2 style="margin:0;">' + esc(p.short) + '</h2><span class="text-primary" style="font-weight:700;">' + esc(p.label) + '</span></div><div class="text-soft" style="font-size:13px;">' + esc(p.name || '') + '</div>' +
      '<div class="mono text-muted" style="font-size:10px;text-transform:uppercase;margin-top:1rem;">Why It Matters</div>' +
      '<p style="background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:.75rem;font-size:13px;">Health score <strong>' + num(p.health, 1) + '</strong>, 30-day momentum <strong>' + pc(p.momentum) + '</strong>, moves with ' + esc(f.short) + ' on ' + (p.overlap == null ? '–' : num(p.overlap * 100) + '%') + ' of daily returns.</p>' +
      '<ul style="font-size:12px;padding-left:1rem;color:var(--text-soft);"><li>Price regime: ' + esc(p.regime || 'unknown') + '</li><li>Breakout probability: ' + (p.breakout == null ? 'n/a' : num(p.breakout * 100) + '%') + '</li></ul><button class="btn btn-primary ms-select" data-s="' + esc(p.symbol) + '" style="width:100%;justify-content:center;margin-top:1rem;">Set ' + esc(p.short) + ' as workspace</button>';
  }

  function battle(f, m) {
    var L = D.chart.lines, cols = ['#f5c518', '#E5A54A', '#8affaa', '#8F877E'], all = [];
    L.forEach(function (l) { l.values.forEach(function (v) { if (v != null) all.push(v); }); });
    var mn = Math.min.apply(0, all), mx = Math.max.apply(0, all), n = D.chart.dates.length;
    var X = function (i) { return 60 + i * 780 / Math.max(1, n - 1); }, Y = function (v) { return 270 - (v - mn) / ((mx - mn) || 1) * 230; };
    var paths = L.map(function (l, k) {
      var d = l.values.map(function (v, i) { return v == null ? '' : (i && l.values[i - 1] != null ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(v).toFixed(1); }).join(' ');
      return '<path d="' + d + '" fill="none" stroke="' + cols[k] + '" stroke-width="' + (k ? 2 : 3) + '"/>';
    }).join('');
    var legend = L.map(function (l, k) { var last = l.values[l.values.length - 1]; return '<span class="' + (k ? 'text-muted' : '') + '"><span style="display:inline-block;width:12px;height:3px;background:' + cols[k] + ';"></span> ' + esc(l.short) + ' (' + (last == null ? '–' : pc(last / 100 - 1)) + ')</span>'; }).join('');
    var fl = L[0].values[L[0].values.length - 1], tl = L[1] ? L[1].values[L[1].values.length - 1] : null;
    var stat = function (l, v, c) { return '<div style="background:var(--surface-2);border-radius:8px;padding:8px;">' + lab(l) + '<div class="mono text-' + c + '" style="font-weight:700;">' + v + '</div></div>'; };
    var v = f.volume, sr = f.sr;
    var c3 = function (t, col, pl, body, foot) { return '<div class="card" style="border-top:3px solid var(--' + col + ');"><div style="display:flex;justify-content:space-between;"><span style="font-weight:600;">' + t + '</span>' + pill(pl, col) + '</div><p style="font-size:13px;">' + body + '</p><p style="font-size:12px;" class="text-muted">' + foot + '</p></div>'; };
    return '<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:1rem;">' + head('Predictive Modelling Engine', 'Battle Forecast', 'Close-price index (base 100) of ' + name(f) + ' against its most urgent peers, plus the model outputs for the next session.') + '<div style="background:var(--surface-2);padding:.5rem 1rem;border-radius:12px;align-self:flex-start;"><span class="mono text-warning" style="font-size:11px;">Volume fit: ' + (f.fit == null ? '–' : num(f.fit * 100) + '%') + ' (in-sample)</span></div></div>' +
      card('ssid_chart', 'Relative Performance Matrix', '<span class="mono text-muted" style="font-size:10px;">Index Base = 100</span>', '<div style="display:flex;gap:1rem;flex-wrap:wrap;font-size:11px;margin-bottom:.5rem;">' + legend + '</div><div style="background:var(--surface-2);border-radius:10px;padding:8px;"><svg viewBox="0 0 860 300" style="width:100%;height:300px;"><g stroke="#8F877E" stroke-opacity=".15" stroke-dasharray="2 4"><line x1="50" x2="840" y1="40" y2="40"/><line x1="50" x2="840" y1="155" y2="155"/><line x1="50" x2="840" y1="270" y2="270"/></g><text x="8" y="44" fill="#8F877E" font-size="10">' + mx.toFixed(0) + '</text><text x="8" y="274" fill="#8F877E" font-size="10">' + mn.toFixed(0) + '</text>' + paths + '<text x="60" y="292" fill="#8F877E" font-size="10">' + D.chart.dates[0] + '</text><text x="840" y="292" fill="#8F877E" font-size="10" text-anchor="end">' + D.chart.dates[n - 1] + '</text></svg></div>' +
        '<div class="grid" style="grid-template-columns:repeat(4,1fr);margin-top:.75rem;">' + stat('Spread (' + esc(f.short) + ' vs ' + esc(L[1] ? L[1].short : '–') + ')', fl != null && tl != null ? (fl - tl >= 0 ? '+' : '') + (fl - tl).toFixed(1) + ' pts' : '–', fl != null && tl != null && fl >= tl ? 'success' : 'danger') + stat('30-Day Momentum', pc(f.momentum), sgn(f.momentum)) + stat('Price Regime', esc(f.regime || 'unknown'), 'warning') + stat('Hurst Exponent', f.hurst == null ? '–' : f.hurst.toFixed(2), 'soft') + '</div>') +
      '<div class="grid" style="grid-template-columns:repeat(3,1fr);">' +
      c3('Volume Forecast (T+1)', v.applicable && v.vs_norm != null && v.vs_norm >= 0 ? 'success' : 'warning', 'XGBoost', v.applicable && v.pred != null ? 'Forecast next-session volume <strong>' + num(v.pred) + '</strong>, ' + pc(v.vs_norm, 0) + ' vs the model\'s usual forecast for this weekday (' + num(v.norm) + ').' : 'No forecast for the next session (an old-style model forecasting a weekend, or not enough forecast history yet).', 'Latest actual volume: ' + num(v.actual) + '. Only this rise/fall signal feeds the health score.') +
      c3('Breakout Probability', 'primary', f.breakout == null ? 'No model' : num(f.breakout * 100) + '%', f.breakout == null ? 'No breakout model for ' + esc(f.short) + ' (too little history, or it did not pass its hold-out check).' : 'Chance that Close exceeds its 1-year (252-session) high within the next 21 sessions.', 'Regime: ' + esc(f.regime || 'unknown')) +
      c3('Support / Resistance', 'danger', 'Pivot lines', sr.sr_support != null ? 'Support <strong>' + rp(sr.sr_support) + '</strong> (touched ' + num(sr.sr_support_touches) + '×), resistance <strong>' + rp(sr.sr_resistance) + '</strong> (' + num(sr.sr_resistance_touches) + '×).' : 'Not enough history to compute levels.', 'Close: ' + rp(f.close)) + '</div>' +
      '<div class="card" style="display:flex;align-items:center;gap:.75rem;background:var(--surface-container-lowest);"><span class="material-symbols-outlined text-muted">verified_user</span><p class="text-muted" style="font-size:12px;margin:0;">Model outputs are signals based on historical data, not guarantees. Source: ' + esc(m.csv) + '.</p></div>';
  }

  /* ---------- shell / wiring ---------- */
  function shell(f, m) {
    var ws = document.querySelector('.topbar .ws'); if (ws) { ws.children[1].textContent = 'Workspace: ' + (f.name || f.short) + ' (' + f.short + ')'; ws.children[2].textContent = 'Live CSV'; }
    var rows = document.querySelectorAll('.sidebar-bottom .row');
    if (rows[1]) rows[1].children[1].textContent = new Date(m.mtime * 1000).toLocaleString();
    if (rows[1]) rows[1].children[0].textContent = 'CSV Updated';
    if (rows[2]) rows[2].children[1].textContent = m.engine;
    var mo = document.querySelector('.market-open'); if (mo) mo.lastChild.textContent = 'Data ' + m.asof;
    var inp = $('.search input'), dl = document.getElementById('ms-list');
    if (inp && !dl) { dl = document.createElement('datalist'); dl.id = 'ms-list'; document.body.appendChild(dl); inp.setAttribute('list', 'ms-list'); inp.placeholder = 'Search ticker or company…'; inp.addEventListener('change', function () { var q = inp.value.trim().toLowerCase(); var hit = D.meta.symbols.filter(function (s) { return s.short.toLowerCase() === q || (s.name || '').toLowerCase().indexOf(q) > -1; })[0]; if (hit) select(hit.symbol); }); }
    if (dl) dl.innerHTML = m.symbols.map(function (s) { return '<option value="' + esc(s.short) + '">' + esc(s.name || '') + '</option>'; }).join('');
  }
  function render() {
    var f = D.focus, m = D.meta;
    document.getElementById(PAGES.overview).innerHTML = overview(f, m);
    document.getElementById(PAGES.health).innerHTML = health(f);
    document.getElementById(PAGES.urgency).innerHTML = urgency(f);
    document.getElementById(PAGES.battle).innerHTML = battle(f, m);
    var keep = CUR && D.peers.filter(function (p) { return p.symbol === CUR.symbol; })[0];   // stay on the peer the user picked
    var fc = document.getElementById('ms-focus'); if (fc) fc.innerHTML = focusCard(keep || D.peers[0], f);
    shell(f, m);
  }
  function load(sym) {
    return fetch('/api/dashboard' + (sym ? '?symbol=' + encodeURIComponent(sym) : ''), { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (j) {
      if (j.error) { setPages(msg('Cannot load MCS_health.csv', esc(j.error) + '<br/><span class="mono">' + esc(j.hint || '') + '</span>')); return; }
      D = j; mtime = j.meta.mtime; try { localStorage.setItem('ms_symbol', j.focus.symbol); } catch (e) { }
      render();
    }).catch(function (e) { setPages(msg('Server not reachable', 'Start it with <span class="mono">python website/server.py</span> and open http://127.0.0.1:8000/ (do not open the .html file directly).')); });
  }
  function select(s) { load(s); }

  document.addEventListener('click', function (e) {
    var b = e.target.closest('.ms-select'); if (b) { select(b.getAttribute('data-s')); return; }
    var p = e.target.closest('.ms-peer'); if (p && D) { var fc = document.getElementById('ms-focus'); if (fc) fc.innerHTML = focusCard(D.peers[+p.getAttribute('data-i')], D.focus); }
  });
  setPages(msg('Loading MCS_health.csv…', 'Reading the latest backend output.'));
  var saved = null; try { saved = localStorage.getItem('ms_symbol'); } catch (e) { }
  load(saved);
  loadSent(); setInterval(loadSent, 2000);
  setInterval(function () { fetch('/api/status', { cache: 'no-store' }).then(function (r) { return r.json(); }).then(function (j) { if (j.mtime && mtime && j.mtime !== mtime) load(D && D.focus.symbol); }).catch(function () { }); }, 10000);
})();
