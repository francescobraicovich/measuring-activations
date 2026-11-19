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
  const reach = 20, margin = 8;
  let birds = [], cells = [], columns, rows, width, height;
  let time = 0, remainder = 0;
  let paused = motion.matches, visible = false, frame = 0, previous = 0;

  function resize() {
    const box = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width / 1.28));
    const h = Math.max(1, Math.round(box.height / 1.28));
    if (birds.length && w === canvas.width && h === canvas.height) return;
    const oldWidth = width, oldHeight = height;
    canvas.width = w; canvas.height = h;
    width = w + 2 * margin; height = h + 2 * margin;
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
    } else {
      birds = Array.from({length: Math.round(w * 1.65)}, () => {
        const x = random(), stream = Math.floor(random() * 3);
        const y = .3 + .2 * stream + .18 * Math.sin(tau * x + stream) + between(-.18,.18);
        const angle = between(0,tau);
        return {x: x * width, y: ((y + 1) % 1) * height, vx: Math.cos(angle), vy: Math.sin(angle),
          turn: 0, noise: 0, pressure: 0, coherence: 0, visibility: 1, depth: random(), beat: between(3.3,5.7), phase: between(0,tau)};
      });
      if (paused) for (let i = 0; i < 30; i++) advance();
    }
    draw();
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
          const spacing = 4 + 3 * b.pressure;
          const force = influence * (distance - spacing) / (1 + d2 * .035);
          pullX += dx * force; pullY += dy * force;
        }
      }
      // A bird responds to sustained crowding gradually. Noisy turns spread
      // through alignment, and the exposed edges of a flock can shear away.
      const crowd = weight / (weight + 6);
      b.pressure += step * .55 * (crowd - b.pressure);
      const norm = 1 / (2 + weight);
      const polar = Math.hypot(alignX, alignY) * norm;
      // A small coherent pocket can lose its outline in a gust. Shrink the
      // silhouettes of birds that peel away, then let them emerge in the flow.
      b.coherence += step * .7 * (polar - b.coherence);
      const breakup = Math.min(1, Math.max(0, (b.coherence - polar - .12) * 3.8));
      b.visibility += step * 7 * (1 - breakup - b.visibility);
      // Persistent angular noise gives a turn a duration of about half a second.
      b.noise = .94 * b.noise + .59 * between(-1,1);
      const nx = tau * b.x / width, ny = tau * b.y / height;
      const gust = Math.sin(nx + ny - time * .42) * Math.cos(ny - time * .26);
      const noise = .75 + 2.4 * (1 - polar) + 4.5 * b.pressure ** 3 + 1.2 * gust;
      const steerX = norm * (4.4 * alignX + .78 * pullX) + .95 * Math.sin(ny + time * .37);
      const steerY = norm * (4.4 * alignY + .78 * pullY) + .95 * Math.cos(nx - time * .28);
      b.turn = Math.max(-5, Math.min(5, b.vx * steerY - b.vy * steerX + b.noise * noise));
    }
    for (const b of birds) {
      const angle = b.turn * step, cos = Math.cos(angle), sin = Math.sin(angle);
      const vx = b.vx * cos - b.vy * sin;
      b.vy = b.vx * sin + b.vy * cos; b.vx = vx;
      const stride = (23 + 13 * b.depth) * step;
      b.x = (b.x + b.vx * stride + width) % width;
      b.y = (b.y + b.vy * stride + height) % height;
    }
  }

  function draw() {
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#000';
    for (const b of birds) {
      if (b.visibility < .28 + .18 * b.depth) continue;
      const depth = b.depth * b.visibility;
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
