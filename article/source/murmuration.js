// Individual wingbeats run through a shared wind field. A moving, folded sheet
// gives each flock its silhouette while birds circulate within it.
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
  const smooth = (start, end, value) => {
    const u = Math.max(0, Math.min(1, (value - start) / (end - start)));
    return u * u * (3 - 2 * u);
  };

  let birds = [], flocks = [], age = 0, time = 0;
  let paused = motion.matches, visible = false, frame = 0, previous = 0;

  function formFlocks() {
    // Two large masses and two satellite flocks occupy different depths.
    const mirror = random() < .5;
    flocks = [
      [.32, .51, .245, .19], [.72, .43, .156, .14],
      [.61, .76, .079, .074], [.85, .24, .049, .045]
    ].map(([x, y, width, height], i) => ({
      x: (mirror ? 1 - x : x) + between(-.025,.025), y: y + between(-.04,.04),
      width, height, turn: between(0,tau), tilt: between(-.24,.24),
      burst: i < 2 ? Infinity : 12.3 + (3 - i) * .32
    }));
    for (const bird of birds) {
      const choice = random();
      bird.flock = flocks[choice < .55 ? 0 : choice < .83 ? 1 : choice < .94 ? 2 : 3];
    }
  }

  function resize() {
    const box = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(box.width / 1.28));
    const height = Math.max(1, Math.round(box.height / 1.28));
    if (birds.length && width === canvas.width && height === canvas.height) return;
    const oldWidth = canvas.width, oldHeight = canvas.height;
    canvas.width = width;
    canvas.height = height;
    if (birds.length) {
      for (const b of birds) { b.x *= width / oldWidth; b.y *= height / oldHeight; }
    } else {
      birds = Array.from({length: Math.round(width * 1.9)}, () => {
        const x = random();
        const stream = Math.floor(random() * 3);
        return {
          x: x * width, y: height * (.3 + .2 * stream + .18 * Math.sin(tau * x + stream) + between(-.18,.18)),
          vx: between(-12,12), vy: between(-8,8), depth: random(), phase: between(0,tau),
          radius: Math.sqrt(random()), offset: between(-1,1), orbit: between(.23,.56),
          beat: between(3.3,5.7), hidden: false, renewed: false
        };
      });
      formFlocks();
      if (paused) {
        age = 10;
        for (const b of birds) guide(b, 0, true);
      }
    }
    draw();
  }

  function guide(b, dt, pose = false) {
    const w = canvas.width, h = canvas.height, f = b.flock;
    const sinceBurst = age - f.burst;
    if (sinceBurst >= 0 && sinceBurst < .48) {
      // The tight core opens into a few loose flecks and then vanishes.
      const speed = 45 + 80 * b.radius;
      b.x += Math.cos(b.phase) * speed * dt;
      b.y += Math.sin(b.phase) * speed * dt;
      b.hidden = sinceBurst > .1 + .36 * b.depth;
      return;
    }
    if (sinceBurst >= .48 && !b.renewed) {
      b.hidden = true;
      if (sinceBurst < .9 + 2.1 * b.radius) return;
      // Re-enter at the edge of an existing stream, one wingbeat at a time.
      const neighbour = birds[Math.floor(random() * birds.length)];
      if (neighbour.hidden) return;
      b.x = neighbour.x + between(-2,2); b.y = neighbour.y + between(-2,2);
      b.vx = neighbour.vx; b.vy = neighbour.vy;
      b.hidden = false; b.renewed = true;
    }

    const gather = smooth(2.2,7.3,age) * (1 - smooth(12.3,14.5,age));
    const theta = b.phase + time * b.orbit;
    const u = b.radius * Math.cos(theta), v = b.radius * Math.sin(theta);
    const wave = time * .78 + f.turn;
    const pinch = Number.isFinite(f.burst) ? 1 - .89 * smooth(f.burst - .6, f.burst, age) : 1;
    const cx = (f.x + .048 * Math.sin(wave * .64)) * w;
    const cy = (f.y + .089 * Math.cos(wave * .78)) * h;
    const x = u + .27 * v * Math.sin(wave + u * 2.8);
    const y = v * (.68 + .43 * Math.cos(wave * .64 + u * 2.8)) + .27 * b.offset
      + .66 * Math.sin(2.8 * u + wave) + .25 * Math.sin(4.7 * u - wave * .67);
    const tx = cx + pinch * f.width * w * x;
    const ty = cy + pinch * (f.height * h * y + f.tilt * f.width * w * u);
    if (pose) { b.x = tx; b.y = ty; return; }

    const nx = b.x / w, ny = b.y / h;
    // Crossing shear flows carry neighbours together without trapping them at a sink.
    // Different headings fan the survivors out when the flock loses cohesion.
    const diagonal = Math.sin(4.5 * nx + 3.6 * ny + time * .65);
    const windX = 29 * Math.sin(6.1 * ny - time * .58) + 21 * diagonal + 21 * Math.cos(theta);
    const windY = 22 * Math.cos(6.8 * nx + time * .47) - 21 * 1.25 * h / w * diagonal + 17 * Math.sin(theta);
    const pull = sinceBurst < 0 && age > 11.7 && Number.isFinite(f.burst) ? 12 : 3.1;
    const response = 1 - Math.exp(-dt * 4.8);
    b.vx += ((1-gather) * windX + gather * (tx-b.x) * pull - b.vx) * response;
    b.vy += ((1-gather) * windY + gather * (ty-b.y) * pull - b.vy) * response;
    b.x += b.vx * dt; b.y += b.vy * dt;
    // The wind wraps beyond the frame. Gathering birds can steer in from outside.
    if (gather < .01) {
      if (b.x < -6) b.x += w + 12; else if (b.x > w + 6) b.x -= w + 12;
      if (b.y < -6) b.y += h + 12; else if (b.y > h + 6) b.y -= h + 12;
    }
  }

  function draw() {
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#000';
    for (const b of birds) {
      if (b.hidden) continue;
      const x = Math.round(b.x), y = Math.round(b.y);
      context.fillRect(x, y, 1, 1);
      if (b.depth < .43) continue;
      const wing = Math.round(Math.sin(time * b.beat * tau + b.phase));
      const bank = b.vx < 0 ? -1 : 1;
      context.fillRect(x - 1, y + wing, 1, 1);
      context.fillRect(x + 1, y + wing, 1, 1);
      if (b.depth > .86) {
        context.fillRect(x + 2 * bank, y + wing * 2, 1, 1);
        context.fillRect(x - 2 * bank, y + wing, 1, 1);
        if (b.depth > .97) context.fillRect(x - 3 * bank, y + wing * 2, 1, 1);
      }
    }
  }

  function tick(now) {
    const dt = previous ? Math.min((now - previous) / 1000, .05) : 0;
    previous = now; time += dt; age += dt;
    if (age >= 18) {
      age = 0;
      formFlocks();
      for (const b of birds) { b.hidden = false; b.renewed = false; }
    }
    for (const b of birds) guide(b, dt);
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
