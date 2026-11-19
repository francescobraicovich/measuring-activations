(() => {
  const canvas = document.getElementById('murmuration');
  const context = canvas.getContext('2d', {alpha: false});
  if (!context) return;
  const scene = canvas.parentElement;
  const control = scene.querySelector('button');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const tau = 2 * Math.PI;
  const random = Math.random;
  const between = (a, b) => a + (b - a) * random();
  const step = 1 / 30;
  const margin = 8;
  const waveSpeed = 75;
  let birds = [], cells = [], columns, rows, width, height, reach;
  let time = 0, remainder = 0, sample = 0, wave = null;
  let paused = motion.matches, visible = false, frame = 0, previous = 0;

  function resize() {
    const box = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width / 1.28));
    const h = Math.max(1, Math.round(box.height / 1.28));
    if (birds.length && w === canvas.width && h === canvas.height) return;
    const oldWidth = width, oldHeight = height;
    canvas.width = w; canvas.height = h;
    width = w + 2 * margin; height = h + 2 * margin;
    reach = Math.min(27, Math.max(15, w * .061));
    columns = Math.max(1, Math.floor(width / reach));
    rows = Math.max(1, Math.floor(height / reach));
    cells = Array.from({length: columns * rows}, () => ({members: [], nearby: []}));
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < columns; x++) {
        const nearby = cells[y * columns + x].nearby;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const cell = cells[((y + dy + rows) % rows) * columns + (x + dx + columns) % columns];
            if (!nearby.includes(cell)) nearby.push(cell);
          }
        }
      }
    }
    if (birds.length) {
      for (const b of birds) { b.x *= width / oldWidth; b.y *= height / oldHeight; }
      if (wave) { wave.x *= width / oldWidth; wave.y *= height / oldHeight; }
    } else {
      birds = Array.from({length: Math.round(w * 2.05)}, () => {
        const x = random(), stream = Math.floor(random() * 3);
        const y = .3 + .2 * stream + .18 * Math.sin(tau * x + stream) + between(-.18,.18);
        const angle = between(0,tau);
        return {x: x * width, y: ((y + 1) % 1) * height, vx: Math.cos(angle), vy: Math.sin(angle),
          turn: 0, noise: 0, density: 0, speedCorrection: 0, alignmentResponse: 0, maturity: 0, core: false, alarm: 0, kickX: 0, kickY: 0, burst: -Infinity, depth: random(), beat: between(3.3,5.7), phase: between(0,tau)};
      });
      if (paused) for (let i = 0; i < 30; i++) advance();
    }
    draw();
  }

  // Signed distance across the wraparound boundary.
  function offset(delta, span) {
    return delta > span / 2 ? delta - span : delta < -span / 2 ? delta + span : delta;
  }

  // Find connected patches of closely spaced birds travelling together.
  // Count actual members so a few stray birds cannot trigger a burst.
  function releaseSmallestFlock() {
    const groups = [];
    let clustered = 0;
    for (const first of birds) {
      if (!first.core || first.visited === sample) continue;
      const members = [first];
      first.visited = sample;
      let x = 0, y = 0, vx = 0, vy = 0, maturity = 0;
      for (let i = 0; i < members.length; i++) {
        const b = members[i];
        x += offset(b.x - first.x, width); y += offset(b.y - first.y, height);
        vx += b.vx; vy += b.vy; maturity += b.maturity;
        for (const cell of b.cell.nearby) for (const other of cell.members) {
          if (!other.core || other.visited === sample || b.vx * other.vx + b.vy * other.vy < .74) continue;
          const dx = offset(other.x - b.x, width), dy = offset(other.y - b.y, height);
          if (dx * dx + dy * dy > (.48 * reach) ** 2) continue;
          other.visited = sample;
          members.push(other);
        }
      }
      const n = members.length;
      if (n < Math.max(18, birds.length * .032) || Math.hypot(vx, vy) / n < .8) continue;
      clustered += n;
      const cx = (first.x + x / n + width) % width, cy = (first.y + y / n + height) % height;
      if (cx < margin + 5 || cx > width - margin - 5 || cy < margin + 5 || cy > height - margin - 5) continue;
      groups.push({members, x: cx, y: cy, maturity: maturity / n});
    }
    if (groups.length < 2 || clustered < .48 * birds.length) return;
    groups.sort((a, b) => a.members.length - b.members.length);
    const small = groups[0], large = groups[groups.length - 1];
    if (small.members.length > large.members.length * .7 || small.maturity < .74) return;

    wave = {x: small.x, y: small.y, start: time};
    for (const b of small.members) {
      const dx = offset(b.x - small.x, width), dy = offset(b.y - small.y, height);
      const direction = Math.atan2(dy, dx) + between(-.20,.20);
      const impulse = between(72,100);
      b.kickX = Math.cos(direction) * impulse; b.kickY = Math.sin(direction) * impulse;
      b.burst = time;
    }
  }

  // Continuous-heading variant of Vicsek alignment (doi:10.1103/PhysRevLett.75.1226).
  // Each update reads the same snapshot. A short-range repulsion leaves space
  // between birds; a weaker attraction keeps neighbours within sight.
  function advance() {
    time += step;
    for (const cell of cells) cell.members.length = 0;
    for (const b of birds) {
      const index = Math.floor(b.y / height * rows) * columns + Math.floor(b.x / width * columns);
      b.cell = cells[index];
      b.cell.members.push(b);
    }
    for (const b of birds) {
      let weight = 0, alignX = 0, alignY = 0, pullX = 0, pullY = 0;
      // Crowded cores spread sideways and along the flight direction so a
      // growing flock occupies more sky instead of collapsing to one pixel.
      const spacing = 4.7 + 6.5 * b.density / (b.density + 28) + 6 * b.alarm;
      for (const cell of b.cell.nearby) {
        for (const other of cell.members) {
          if (other === b) continue;
          let dx = other.x - b.x, dy = other.y - b.y;
          if (dx > width / 2) dx -= width; else if (dx < -width / 2) dx += width;
          if (dy > height / 2) dy -= height; else if (dy < -height / 2) dy += height;
          const d2 = dx * dx + dy * dy;
          if (d2 >= reach * reach) continue;
          const distance = Math.sqrt(d2);
          const influence = 1 - distance / reach;
          weight += influence;
          alignX += influence * other.vx; alignY += influence * other.vy;
          const force = influence * (distance - spacing) / (1 + d2 * .035);
          pullX += dx * force; pullY += dy * force;
        }
      }
      b.density = weight;
      const norm = 1 / (2 + weight);
      b.speedCorrection = Math.max(-9, Math.min(9, 1.4 * norm * (pullX * b.vx + pullY * b.vy)));
      const polar = Math.hypot(alignX, alignY) * norm;
      b.core = weight > 5 && polar > .67;
      b.maturity += step * ((b.core && b.alarm < .15 ? 1 : 0) - b.maturity) * (b.core ? .33 : 1.2);
      // The disturbance travels out from the burst. Birds ahead of it keep
      // their heading; those behind it turn independently and gain speed.
      b.alarm = 0;
      if (wave) {
        const distance = Math.hypot(offset(b.x - wave.x, width), offset(b.y - wave.y, height));
        const elapsed = time - wave.start - .55 - distance / waveSpeed;
        if (elapsed > 0) b.alarm = 1.35 * (1 - Math.exp(-.8 * elapsed)) * Math.exp(-.23 * elapsed);
      }
      // Persistent angular noise gives a turn a duration of about half a second.
      b.noise = .94 * b.noise + .59 * between(-1,1);
      const nx = tau * b.x / width, ny = tau * b.y / height;
      const gust = Math.sin(nx + ny - time * .42) * Math.cos(ny - time * .26);
      const noise = .75 + 1.7 * (1 - polar) + .55 * gust + 12 * b.alarm;
      // Neighbour influence builds over several seconds of undisturbed flight.
      // A passing disturbance erodes it locally before a flock can reform.
      b.alignmentResponse += step * ((1 - b.alignmentResponse) / 7 - b.alarm * b.alignmentResponse);
      const coupling = norm * (.15 + .85 * b.alignmentResponse) * (1 - b.alarm) ** 3;
      const steerX = coupling * (5.5 * alignX + 1.08 * pullX) + .95 * Math.sin(ny + time * .37);
      const steerY = coupling * (5.5 * alignY + 1.08 * pullY) + .95 * Math.cos(nx - time * .28);
      b.turn = Math.max(-8, Math.min(8, b.vx * steerY - b.vy * steerX + b.noise * noise));
    }
    // Sample connectivity at 4 Hz; the birds themselves move at 30 Hz.
    if (++sample % 8 === 0) {
      if (wave && time - wave.start > 13 + Math.hypot(width, height) / (2 * waveSpeed)) wave = null;
      if (!wave) releaseSmallestFlock();
    }
    for (const b of birds) {
      const angle = b.turn * step, cos = Math.cos(angle), sin = Math.sin(angle);
      const vx = b.vx * cos - b.vy * sin;
      b.vy = b.vx * sin + b.vy * cos; b.vx = vx;
      const stride = (23 + 13 * b.depth + b.speedCorrection + 30 * b.alarm) * step;
      b.x = (b.x + b.vx * stride + b.kickX * step + width) % width;
      b.y = (b.y + b.vy * stride + b.kickY * step + height) % height;
      b.kickX *= .89; b.kickY *= .89;
    }
  }

  function draw() {
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#000';
    for (const b of birds) {
      const sinceBurst = time - b.burst;
      // As the rim opens, a few birds recede to single pixels and return in
      // the scattered flight. Their trajectories continue throughout.
      const retreat = sinceBurst < 2 ? Math.sin(Math.PI * sinceBurst / 2) ** 2 : 0;
      if (retreat > .76 + .24 * b.depth) continue;
      const depth = b.depth * (1 - retreat);
      const x = Math.round(b.x) - margin, y = Math.round(b.y) - margin;
      context.fillRect(x, y, 1, 1);
      if (depth < .43) continue;
      const wing = Math.round(Math.sin(time * b.beat * tau + b.phase));
      const bank = b.vx < 0 ? -1 : 1;
      context.fillRect(x - 1, y + wing, 1, 1);
      context.fillRect(x + 1, y + wing, 1, 1);
      if (depth > .86) {
        context.fillRect(x + 2 * bank, y + wing * 2, 1, 1);
        context.fillRect(x - 2 * bank, y + wing, 1, 1);
        if (depth > .97) context.fillRect(x - 3 * bank, y + wing * 2, 1, 1);
      }
    }
  }

  function tick(now) {
    if (previous) remainder += Math.min((now - previous) / 1000, .1);
    previous = now;
    while (remainder >= step) {
      advance();
      remainder -= step;
    }
    draw();
    frame = requestAnimationFrame(tick);
  }

  function playback() {
    cancelAnimationFrame(frame);
    previous = 0;
    control.toggleAttribute('data-paused', paused);
    control.title = paused ? 'Play animation' : 'Pause animation';
    control.setAttribute('aria-label', control.title);
    if (visible && !paused && !document.hidden) frame = requestAnimationFrame(tick);
  }

  scene.hidden = false;
  resize();
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; playback(); }).observe(scene);
  control.addEventListener('click', () => { paused = !paused; playback(); });
  motion.addEventListener('change', () => { paused = motion.matches; playback(); });
  document.addEventListener('visibilitychange', playback);
  playback();
})();
