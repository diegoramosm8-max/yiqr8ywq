"use strict";

/* =========================================================================
   MANGA CAM — lógica principal
   Todo corre en el dispositivo: cámara, seguimiento facial (MediaPipe Face
   Landmarker) y segmentación de piel (MediaPipe Image Segmenter) se cargan
   desde CDN pero procesan el video localmente, sin enviar nada a un server.
   ========================================================================= */

const ANGLES = [
  { id: "perfil-izq", label: "Perfil izq." },
  { id: "3-4-izq", label: "3/4 izq." },
  { id: "frontal", label: "Frontal" },
  { id: "3-4-der", label: "3/4 der." },
  { id: "perfil-der", label: "Perfil der." },
];

const EMOTIONS = [
  { id: "neutral", label: "Neutral" },
  { id: "feliz", label: "Feliz" },
  { id: "riendo", label: "Riendo" },
  { id: "decepcion", label: "Decepción" },
  { id: "sorprendido", label: "Sorprendido" },
  { id: "enojado", label: "Enojado" },
];

/** stickers[emotionId][angleId] = { url, img } */
const stickers = {};
EMOTIONS.forEach((e) => (stickers[e.id] = {}));

const state = {
  facingMode: "user",
  stream: null,
  track: null,
  mirrorAngle: false,
  handOcclusionEnabled: false,
  handOcclusionAvailable: false,
  fitSize: 1,
  fitOffset: 0,
  isRecording: false,
  recordStart: 0,
  visionModule: null,
  faceLandmarker: null,
  imageSegmenter: null,
  lastFace: null, // {cx, cy, size, roll, yaw, emotion, angle}
  segmentationMask: null,
  deferredInstallPrompt: null,
};

const el = (id) => document.getElementById(id);

/* ---------------------------------------------------------------------- */
/* 1. Arranque / permiso de cámara                                        */
/* ---------------------------------------------------------------------- */

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  el(id).classList.add("active");
}

async function initBoot() {
  if (!window.isSecureContext) {
    el("insecure-warning").classList.remove("hidden");
    el("btn-start-camera").classList.add("hidden");
    return;
  }
  el("btn-start-camera").addEventListener("click", requestCameraAndStart);
  el("btn-retry-camera").addEventListener("click", requestCameraAndStart);
}

async function requestCameraAndStart() {
  el("denied-warning").classList.add("hidden");
  try {
    await startCamera(state.facingMode);
    showScreen("screen-camera");
    await initModels();
    startRenderLoop();
  } catch (err) {
    console.error(err);
    el("denied-reason").textContent = describeCameraError(err);
    el("denied-warning").classList.remove("hidden");
  }
}

function describeCameraError(err) {
  if (err && err.name === "NotAllowedError") {
    return "Rechazaste el permiso de cámara (o el sistema operativo lo tiene bloqueado para el navegador).";
  }
  if (err && err.name === "NotFoundError") {
    return "No se encontró ninguna cámara en este dispositivo.";
  }
  if (err && err.name === "NotReadableError") {
    return "La cámara parece estar en uso por otra app.";
  }
  return "Puede que hayas rechazado el permiso, o que otra app la esté usando.";
}

/* ---------------------------------------------------------------------- */
/* 2. Cámara: getUserMedia, flip, zoom                                    */
/* ---------------------------------------------------------------------- */

async function startCamera(facingMode) {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
  }
  // Sin ancho/alto fijos: evitamos forzar un zoom digital no deseado.
  const constraints = {
    audio: camCfg.mic,
    video: {
      facingMode: { ideal: facingMode },
      ...(RES[camCfg.res] ? { width: { ideal: RES[camCfg.res][0] }, height: { ideal: RES[camCfg.res][1] } } : {}),
      ...(camCfg.fps !== "auto" ? { frameRate: { ideal: Number(camCfg.fps) } } : {}),
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  state.stream = stream;
  state.track = stream.getVideoTracks()[0];
  state.track = stream.getVideoTracks()[0];
  const video = el("video");
  video.srcObject = stream;
  await video.play();
  setupZoomControl();
}

function setupZoomControl() {
  const wrap = el("zoom-wrap");
  const slider = el("zoom-slider");
  try {
    const caps = state.track.getCapabilities ? state.track.getCapabilities() : {};
    if (caps.zoom) {
      wrap.classList.remove("hidden");
      slider.min = caps.zoom.min;
      slider.max = caps.zoom.max;
      slider.step = caps.zoom.step || 0.1;
      slider.value = caps.zoom.min;
      slider.oninput = () => {
        state.track.applyConstraints({ advanced: [{ zoom: Number(slider.value) }] }).catch(() => {});
      };
    } else {
      wrap.classList.add("hidden");
    }
  } catch {
    wrap.classList.add("hidden");
  }
}

el("btn-flip-camera").addEventListener("click", async () => {
  state.facingMode = state.facingMode === "user" ? "environment" : "user";
  try {
    await startCamera(state.facingMode);
  } catch (err) {
    console.error("No se pudo cambiar de cámara", err);
  }
});

/* ---------------------------------------------------------------------- */
/* 3. Modelos de MediaPipe (face landmarker + segmentador de piel)        */
/* ---------------------------------------------------------------------- */

// @mediapipe/tasks-vision se distribuye solo como módulo ES (no expone
// globals de window). Por eso se carga con import() dinámico en vez de
// un <script> clásico — esa era la causa de que el seguimiento facial y
// la segmentación de piel fallaran silenciosamente al crearse.
const MEDIAPIPE_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

async function loadVisionModule() {
  if (!state.visionModule) {
    state.visionModule = await import(MEDIAPIPE_CDN);
  }
  return state.visionModule;
}

async function initModels() {
  let vision;
  try {
    vision = await loadVisionModule();
  } catch (err) {
    console.error("No se pudo cargar la librería de MediaPipe (¿sin conexión la primera vez?):", err);
    el("tracking-hint").textContent = "No se pudo cargar el seguimiento facial (revisá tu conexión e recargá).";
    el("tracking-hint").classList.remove("hidden");
    updateHandOcclusionUi();
    return;
  }
  const { FilesetResolver, FaceLandmarker, ImageSegmenter } = vision;

  let filesetResolver;
  try {
    filesetResolver = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
  } catch (err) {
    console.error("No se pudo inicializar el runtime WASM de MediaPipe:", err);
    el("tracking-hint").textContent = "No se pudo cargar el seguimiento facial en este dispositivo.";
    el("tracking-hint").classList.remove("hidden");
    updateHandOcclusionUi();
    return;
  }

  try {
    state.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  } catch (err) {
    console.error("No se pudo cargar el seguimiento facial", err);
    el("tracking-hint").textContent = "No se pudo cargar el seguimiento facial en este dispositivo.";
    el("tracking-hint").classList.remove("hidden");
  }

  try {
    state.imageSegmenter = await ImageSegmenter.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
    state.handOcclusionAvailable = true;
  } catch (err) {
    console.warn("Segmentación de piel no disponible en este dispositivo:", err);
    state.handOcclusionAvailable = false;
  }
  updateHandOcclusionUi();
}

function updateHandOcclusionUi() {
  const toggle = el("toggle-hand-occlusion");
  const status = el("hand-occlusion-status");
  if (!state.handOcclusionAvailable) {
    toggle.disabled = true;
    toggle.checked = false;
    state.handOcclusionEnabled = false;
    status.textContent = "No disponible en este dispositivo — el resto de la app sigue funcionando normalmente.";
  } else {
    toggle.disabled = false;
    status.textContent = "Puede afectar el rendimiento en celulares más antiguos.";
  }
}

/* ---------------------------------------------------------------------- */
/* 4. Clasificación de expresión y ángulo a partir de MediaPipe            */
/* ---------------------------------------------------------------------- */

const TUNING = { minScore: 0.5, stickiness: 0.08, holdMs: 140, smoothTauMs: 55 };
const clamp01 = (v) => Math.max(0, Math.min(1, v));

function scoreEmotions(blendshapes) {
  const sc = { neutral: 0.4, feliz: 0, riendo: 0, decepcion: 0, sorprendido: 0, enojado: 0 };
  if (!blendshapes || !blendshapes.length) return sc;
  const { smile, frown, jaw, browUp, browDown, eyeWide, squint } = applyCalib(faceFeatures(blendshapes));
  const noSmile = 1 - clamp01(smile / 0.3);
  sc.sorprendido = (clamp01(browUp / 0.5) + clamp01(eyeWide / 0.4) + clamp01(jaw / 0.35)) / 3;
  sc.riendo = Math.sqrt(clamp01(smile / 0.6) * clamp01(jaw / 0.3));
  sc.feliz = clamp01(smile / 0.45) * (1 - 0.6 * clamp01(jaw / 0.4));
  sc.enojado = ((clamp01(browDown / 0.5) + clamp01(squint / 0.4)) / 2) * noSmile;
  sc.decepcion = clamp01(frown / 0.4) * noSmile;
  return sc;
}

const emoState = { current: "neutral", candidate: "neutral", since: 0 };
function stabilizeEmotion(scores, now) {
  let best = "neutral", bestScore = TUNING.minScore;
  for (const k in scores) {
    if (k === "neutral") continue;
    const v = scores[k] + (k === emoState.current ? TUNING.stickiness : 0);
    if (v >= bestScore) { best = k; bestScore = v; }
  }
  if (best === emoState.current) { emoState.candidate = best; return best; }
  if (best !== emoState.candidate) { emoState.candidate = best; emoState.since = now; }
  else if (now - emoState.since >= TUNING.holdMs) emoState.current = best;
  return emoState.current;
}

function classifyAngle(yawDegrees, invert) {
  const c = calib; // con calibración, el offset, los límites y la inversión salen de ella
  let y = yawDegrees - (c ? c.offset : 0);
  if (c ? c.invert : invert) y = -y;
  const side = y < 0 ? "izq" : "der";
  const M = c ? (y < 0 ? c.left : c.right) : 50, m = Math.abs(y);
  if (m > (c ? Math.max(18, 0.8 * M) : 40)) return "perfil-" + side;
  if (m > (c ? Math.max(8, 0.24 * M) : 12)) return "3-4-" + side;
  return "frontal";
}

/** Extrae yaw/roll (grados) de la matriz de transformación facial 4x4. */
function anglesFromMatrix(matrix) {
  const d = matrix.data;
  // d está en orden column-major (4x4)
  const m00 = d[0], m10 = d[1], m20 = d[2];
  const m21 = d[6], m22 = d[10];
  const yaw = Math.atan2(-m20, Math.sqrt(m21 * m21 + m22 * m22)) * (180 / Math.PI);
  const roll = Math.atan2(m10, m00) * (180 / Math.PI);
  return { yaw, roll };
}

/* ---------------------------------------------------------------------- */
/* 5. Loop de render: video + sticker + oclusión, dibujado en canvas       */
/* ---------------------------------------------------------------------- */

const canvas = el("canvas-output");
const ctx = canvas.getContext("2d", { willReadFrequently: false });
const video = el("video");
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

let lastVideoTime = -1;
let segFrameCounter = 0;

function resizeCanvasToVideo() {
  if (video.videoWidth && canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    maskCanvas.width = video.videoWidth;
    maskCanvas.height = video.videoHeight;
  }
}

function startRenderLoop() {
  requestAnimationFrame(renderFrame);
}

function renderFrame() {
  requestAnimationFrame(renderFrame);
  if (video.readyState < 2 || video.videoWidth === 0 || procBusy) return;
  resizeCanvasToVideo();

  const nowMs = performance.now();
  perfTick(nowMs);
  if (hasRvfc ? !newFrame : video.currentTime === lastPaintT) return; // sin frame nuevo: no repintar
  newFrame = false; lastPaintT = video.currentTime; paintCount++;

  // 1) Dibujar el frame de cámara (espejado si es cámara frontal).
  ctx.save();
  if (state.facingMode === "user") {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  // 2) Seguimiento facial.
  frameN++;
  if (state.faceLandmarker && nowMs - lastDetT >= detInterval() && (hasRvfc ? vfMeta !== lastDetMeta : video.currentTime !== lastVideoTime)) {
    lastVideoTime = video.currentTime; lastDetMeta = vfMeta;
    try {
      const tw = 384, th = Math.round((tw * video.videoHeight) / video.videoWidth);
      if (trackCanvas.width !== tw || trackCanvas.height !== th) { trackCanvas.width = tw; trackCanvas.height = th; }
      trackCtx.drawImage(video, 0, 0, tw, th); // MediaPipe trabaja a baja resolución: no hace falta pasarle 1080p
      const d0 = performance.now();
      const result = state.faceLandmarker.detectForVideo(trackCanvas, nowMs);
      detEma = detEma ? detEma * 0.8 + (performance.now() - d0) * 0.2 : performance.now() - d0;
      lastDetT = performance.now();
      pushSample(MET.detMs, lastDetT - d0); detCount++;
      updateFaceState(result);
    } catch (err) {
      /* si el modelo falla en un frame puntual, seguimos con lo último detectado */
    }
  }

  el("tracking-hint").classList.toggle("hidden", !!state.lastFace);

  // 3) Sticker sobre el rostro.
  if (state.lastFace) drawSticker(easeFace(state.lastFace, nowMs));
  else state.disp = null;

  // 4) Oclusión de manos/antebrazos (experimental, a media frecuencia).
  if (state.handOcclusionEnabled && state.imageSegmenter && !P().noOcclusion && !rawRec) {
    segFrameCounter = (segFrameCounter + 1) % P().segEvery;
    if (segFrameCounter === 0) {
      try {
        const r = state.imageSegmenter.segmentForVideo(video, nowMs);
        if (r.categoryMask) { updateOcclusionMask(r.categoryMask); r.categoryMask.close(); }
      } catch { /* seguimos con la máscara anterior */ }
    }
    drawHandOcclusion();
  }
  feedRecorder(); // grabación WebCodecs: un frame de cámara = un frame de video
  pushSample(MET.frameMs, performance.now() - nowMs); renderCount++;
}

function smoothFace(raw, now) {
  const p = state.smooth;
  if (!p) return (state.smooth = { ...raw, t: now });
  const k = 1 - Math.exp(-(now - p.t) / TUNING.smoothTauMs); // dependiente del tiempo: sin lag extra
  for (const key of ["cxNorm", "cyNorm", "faceW", "faceH", "roll", "yaw"]) p[key] += (raw[key] - p[key]) * k;
  p.t = now;
  return p;
}

function updateFaceState(result) {
  if (!result.faceLandmarks || !result.faceLandmarks.length) {
    state.lastFace = null;
    state.smooth = null;
    return;
  }
  const landmarks = result.faceLandmarks[0];

  // Bounding box en coordenadas normalizadas -> tamaño de cabeza aproximado.
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const cxNorm = (minX + maxX) / 2;
  const cyNorm = (minY + maxY) / 2;
  const faceW = maxX - minX;
  const faceH = maxY - minY;

  let yaw = 0, roll = 0;
  if (result.facialTransformationMatrixes && result.facialTransformationMatrixes.length) {
    const a = anglesFromMatrix(result.facialTransformationMatrixes[0]);
    yaw = a.yaw;
    roll = a.roll;
  }

  state.raw = { f: faceFeatures(result.faceBlendshapes), yaw };
  const now = clock();
  const sm = smoothFace({ cxNorm, cyNorm, faceW, faceH, roll, yaw }, now);
  const emotion = stabilizeEmotion(scoreEmotions(result.faceBlendshapes), now);
  const angle = stabilizeAngle(classifyAngle(sm.yaw, state.mirrorAngle), now);
  state.lastFace = { cxNorm: sm.cxNorm, cyNorm: sm.cyNorm, faceW: sm.faceW, faceH: sm.faceH, roll: sm.roll, yaw: sm.yaw, emotion, angle };
}

function pickStickerImage(emotion, angle) {
  const chain = [[emotion, angle], [emotion, "frontal"], ["neutral", angle], ["neutral", "frontal"]];
  for (const [e, a] of chain) {
    const st = stickers[e] && stickers[e][a];
    if (st && st.img) return { img: st.img, key: e + "|" + a };
  }
  return null;
}

function drawSticker(face) {
  const pick = pickStickerImage(face.emotion, face.angle);
  if (!pick) return;
  const img = pick.img, tf = getTf(pick.key);

  // El video ya se dibujó espejado si es cámara frontal, así que trabajamos
  // en coordenadas de pantalla (no normalizadas-espejadas) para el centro.
  let cxNorm = face.cxNorm;
  if (state.facingMode === "user") cxNorm = 1 - cxNorm;

  const cx = cxNorm * canvas.width;
  const cy = face.cyNorm * canvas.height;

  // Tamaño base = mayor dimensión de la cabeza detectada, ampliado para
  // cubrir pelo/orejas/frente (no solo el óvalo facial), y ajustado por
  // el control manual de tamaño.
  const headSpan = Math.max(face.faceW * canvas.width, face.faceH * canvas.height);
  const baseSize = headSpan * 2.1 * state.fitSize;

  const aspect = img.naturalWidth / img.naturalHeight || 1;
  const drawW = baseSize * tf.scale;
  const drawH = drawW / aspect;

  const verticalOffsetPx = state.fitOffset * headSpan;
  const rollRad = ((state.facingMode === "user" ? -face.roll : face.roll) * Math.PI) / 180;

  ctx.save();
  ctx.translate(cx + tf.x * headSpan, cy + verticalOffsetPx + tf.y * headSpan);
  ctx.rotate(rollRad + (tf.rot * Math.PI) / 180);
  if (tf.mirror) ctx.scale(-1, 1);
  ctx.drawImage(img, -drawW / 2, -drawH / 2, drawW, drawH);
  ctx.restore();
}

// Canvases/buffers reutilizados (se recrean solo si cambia el tamaño).
let occ = null;
function updateOcclusionMask(mask) {
  const w = mask.width, h = mask.height, W = canvas.width, H = canvas.height;
  if (!occ || occ.w !== w || occ.h !== h || occ.W !== W || occ.H !== H) {
    const mk = document.createElement("canvas"); mk.width = w; mk.height = h;
    const sk = document.createElement("canvas"); sk.width = W; sk.height = H;
    occ = { w, h, W, H, mk, mkc: mk.getContext("2d"), sk, skc: sk.getContext("2d"), img: new ImageData(w, h) };
  }
  const d = mask.getAsUint8Array(), px = occ.img.data;
  for (let i = 0; i < d.length; i++) px[i * 4 + 3] = d[i] === 2 ? 255 : 0; // 2 = body-skin
  occ.mkc.putImageData(occ.img, 0, 0);
}

function drawHandOcclusion() {
  if (!occ) return;
  const { W, H, skc: c } = occ;
  const mirror = state.facingMode === "user";
  // La máscara está en orientación de video: se escala a W×H y se espeja igual que el fondo.
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.globalCompositeOperation = "source-over";
  c.clearRect(0, 0, W, H);
  c.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? W : 0, 0);
  c.drawImage(video, 0, 0, W, H);
  c.globalCompositeOperation = "destination-in";
  c.drawImage(occ.mk, 0, 0, W, H);
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.globalCompositeOperation = "source-over";
  ctx.drawImage(occ.sk, 0, 0);
}

/* ---------------------------------------------------------------------- */
/* 6. Controles de ajuste fino (tamaño / altura) e inversión               */
/* ---------------------------------------------------------------------- */

function bindFitControls(sizeInput, offsetInput) {
  sizeInput.addEventListener("input", () => (state.fitSize = Number(sizeInput.value)));
  offsetInput.addEventListener("input", () => (state.fitOffset = Number(offsetInput.value)));
}
bindFitControls(el("fit-size"), el("fit-offset"));

el("toggle-mirror-angle").addEventListener("change", (e) => {
  state.mirrorAngle = e.target.checked;
});

el("toggle-hand-occlusion").addEventListener("change", (e) => {
  state.handOcclusionEnabled = e.target.checked && state.handOcclusionAvailable;
});

/* ---------------------------------------------------------------------- */
/* 7. Panel de ajustes: grilla de 30 stickers                             */
/* ---------------------------------------------------------------------- */

function buildStickerGrid() {
  const grid = el("sticker-grid");
  grid.innerHTML = "";
  EMOTIONS.forEach((emotion) => {
    const group = document.createElement("div");
    group.className = "emotion-group";
    const h4 = document.createElement("h4");
    h4.textContent = emotion.label;
    group.appendChild(h4);

    const slots = document.createElement("div");
    slots.className = "angle-slots";

    ANGLES.forEach((angle) => {
      const slot = document.createElement("div");
      slot.className = "angle-slot";
      slot.dataset.emotion = emotion.id;
      slot.dataset.angle = angle.id;

      const label = document.createElement("span");
      label.className = "angle-label";
      label.textContent = angle.label;
      slot.appendChild(label);

      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/png";
      input.addEventListener("change", (e) => handleStickerUpload(emotion.id, angle.id, e));
      slot.appendChild(input);

      slots.appendChild(slot);
    });

    group.appendChild(slots);
    grid.appendChild(group);
  });
}

function handleStickerUpload(emotionId, angleId, event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const prev = stickers[emotionId][angleId];
    if (prev && prev.url) URL.revokeObjectURL(prev.url);
    stickers[emotionId][angleId] = { url, img };
    refreshSlotVisual(emotionId, angleId);
    persistSticker(emotionId, angleId, file);
  };
  img.src = url;
}

function refreshSlotVisual(emotionId, angleId) {
  const slot = document.querySelector(
    `.angle-slot[data-emotion="${emotionId}"][data-angle="${angleId}"]`
  );
  if (!slot) return;
  const entry = stickers[emotionId][angleId];

  slot.querySelectorAll("img, .slot-remove").forEach((n) => n.remove());

  if (entry && entry.url) {
    slot.classList.add("has-image");
    const img = document.createElement("img");
    img.src = entry.url;
    slot.insertBefore(img, slot.firstChild);

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "slot-remove";
    removeBtn.textContent = "✕";
    removeBtn.addEventListener("click", (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      URL.revokeObjectURL(entry.url);
      delete stickers[emotionId][angleId];
      removeStickerDb(emotionId, angleId);
      slot.classList.remove("has-image");
      slot.querySelectorAll("img, .slot-remove").forEach((n) => n.remove());
      const fileInput = slot.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = "";
    });
    slot.appendChild(removeBtn);
    const editBtn = document.createElement("button");
    editBtn.type = "button";
    editBtn.className = "slot-edit";
    editBtn.textContent = "✎";
    editBtn.addEventListener("click", (evt) => { evt.preventDefault(); evt.stopPropagation(); openEditor(emotionId, angleId); });
    slot.appendChild(editBtn);
  } else {
    slot.classList.remove("has-image");
  }
}

buildStickerGrid();

/* ---------------------------------------------------------------------- */
/* 8. Panel de ajustes: abrir/cerrar                                      */
/* ---------------------------------------------------------------------- */

el("btn-open-settings").addEventListener("click", () => {
  el("panel-settings").classList.add("open");
  el("settings-scrim").classList.add("show");
});
function closeSettings() {
  el("panel-settings").classList.remove("open");
  el("settings-scrim").classList.remove("show");
}
el("btn-close-settings").addEventListener("click", closeSettings);
el("settings-scrim").addEventListener("click", closeSettings);

/* ---------------------------------------------------------------------- */
/* 9. Grabación de video (sticker + oclusión ya "horneados")               */
/* ---------------------------------------------------------------------- */

let mediaRecorder = null;
let recordedChunks = [];
let recordedBlobUrl = null;
let timerInterval = null;

function pickMimeType() {
  const candidates = [
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
    "video/mp4",
  ];
  return candidates.find((c) => window.MediaRecorder && MediaRecorder.isTypeSupported(c)) || "";
}

el("btn-record").addEventListener("click", () => {
  if (state.isRecording) stopRecording();
  else startRecording();
});

async function startRecording() {
  if (camCfg.recMode === "fluido" && startRawRecording()) return;
  if (await startWebCodecsRecording()) return;
  const canvasStream = canvas.captureStream(recFps());
  const audioTracks = state.stream ? state.stream.getAudioTracks() : [];
  audioTracks.forEach((t) => canvasStream.addTrack(t));

  recordedChunks = [];
  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(canvasStream, { ...(mimeType ? { mimeType } : {}), videoBitsPerSecond: recBitrate(), audioBitsPerSecond: 128000 });
  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = onRecordingStopped;
  mediaRecorder.start();

  state.isRecording = true;
  state.recordStart = Date.now();
  el("btn-record").classList.add("is-recording");
  el("rec-timer").classList.remove("hidden");
  timerInterval = setInterval(updateTimerDisplay, 250);
}

function stopRecording() {
  if (rawRec) { uiRecStop(); rawRec.stop(); return; }
  if (wc) { stopWebCodecs(); return; }
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  state.isRecording = false;
  el("btn-record").classList.remove("is-recording");
  el("rec-timer").classList.add("hidden");
  clearInterval(timerInterval);
}

function updateTimerDisplay() {
  const elapsed = Math.floor((Date.now() - state.recordStart) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const ss = String(elapsed % 60).padStart(2, "0");
  el("rec-timer-value").textContent = `${mm}:${ss}`;
}

function onRecordingStopped() {
  const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "video/webm" });
  if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
  recordedBlobUrl = URL.createObjectURL(blob);

  const previewVideo = el("preview-video");
  previewVideo.src = recordedBlobUrl;
  el("btn-download").href = recordedBlobUrl;

  const ext = (mediaRecorder.mimeType || "").includes("mp4") ? "mp4" : "webm";
  el("btn-download").setAttribute("download", `manga-cam.${ext}`);

  showScreen("screen-preview");
}

el("btn-discard").addEventListener("click", () => {
  if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
  recordedBlobUrl = null;
  showScreen("screen-camera");
});

/* ---------------------------------------------------------------------- */
/* 10. Instalación de la PWA                                              */
/* ---------------------------------------------------------------------- */

window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  state.deferredInstallPrompt = e;
  el("btn-install").classList.remove("hidden");
});

el("btn-install").addEventListener("click", async () => {
  if (!state.deferredInstallPrompt) return;
  state.deferredInstallPrompt.prompt();
  await state.deferredInstallPrompt.userChoice;
  state.deferredInstallPrompt = null;
  el("btn-install").classList.add("hidden");
});

function showManualInstallInstructions() {
  const isStandalone =
    window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone;
  if (isStandalone) return;

  const ua = navigator.userAgent;
  const isIOS = /iPad|iPhone|iPod/.test(ua) && !window.MSStream;
  const isAndroid = /Android/.test(ua);
  const box = el("install-instructions");

  if (isIOS) {
    box.textContent =
      'En iPhone/iPad: tocá el botón "Compartir" en Safari y elegí "Agregar a inicio".';
    box.classList.remove("hidden");
  } else if (isAndroid && !state.deferredInstallPrompt) {
    box.textContent = 'En Android/Chrome: abrí el menú (⋮) y elegí "Instalar app".';
    box.classList.remove("hidden");
  }
}

/* ---------------------------------------------------------------------- */
/* 11. Service worker                                                     */
/* ---------------------------------------------------------------------- */

if ("serviceWorker" in navigator) {
  window.addEventListener("load", async () => {
    try {
      const hadController = !!navigator.serviceWorker.controller;
      const reg = await navigator.serviceWorker.register("./sw.js");
      const offer = (w) => {
        el("update-banner").classList.remove("hidden");
        el("btn-update").onclick = () => w.postMessage("SKIP_WAITING");
      };
      if (reg.waiting && hadController) offer(reg.waiting);
      reg.addEventListener("updatefound", () => {
        const w = reg.installing;
        if (w) w.addEventListener("statechange", () => { if (w.state === "installed" && navigator.serviceWorker.controller) offer(w); });
      });
      let reloaded = false;
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (!hadController || reloaded) return;
        reloaded = true; location.reload();
      });
      setInterval(() => reg.update().catch(() => {}), 60 * 60 * 1000);
    } catch (err) { console.warn("No se pudo registrar el service worker:", err); }
  });
}

/* ---------------------------------------------------------------------- */
/* Arranque                                                                */
/* ---------------------------------------------------------------------- */

initBoot();
showManualInstallInstructions();

/* ---------------------------------------------------------------------- */
/* 12. Biblioteca de personajes (IndexedDB, todo local)                    */
/* ---------------------------------------------------------------------- */

let dbp = null, activeChar = null;
function openDb() {
  return dbp || (dbp = new Promise((res, rej) => {
    const r = indexedDB.open("manga-cam", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("characters", { keyPath: "id" });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  }));
}
async function dbTx(mode, fn) {
  const d = await openDb();
  return new Promise((res, rej) => {
    const t = d.transaction("characters", mode);
    const rq = fn(t.objectStore("characters"));
    t.oncomplete = () => res(rq && rq.result);
    t.onerror = () => rej(t.error);
  });
}
const dbAll = () => dbTx("readonly", (st) => st.getAll());
const dbPut = (c) => dbTx("readwrite", (st) => st.put(c));
const dbDel = (id) => dbTx("readwrite", (st) => st.delete(id));

function newCharacter(name) {
  const t = Date.now();
  return { id: crypto.randomUUID(), name, created: t, modified: t, stickers: {} };
}
function persistSticker(e, a, file) {
  if (!activeChar) return;
  activeChar.stickers[e + "|" + a] = file;
  activeChar.modified = Date.now();
  dbPut(activeChar).catch((err) => console.warn("No se pudo guardar el sticker", err));
}
function removeStickerDb(e, a) {
  if (!activeChar) return;
  delete activeChar.stickers[e + "|" + a];
  activeChar.modified = Date.now();
  dbPut(activeChar).catch(() => {});
}

async function loadCharacter(c) {
  activeChar = c;
  EMOTIONS.forEach((e) => {
    for (const a in stickers[e.id]) { const st = stickers[e.id][a]; if (st && st.url) URL.revokeObjectURL(st.url); }
    stickers[e.id] = {};
  });
  buildStickerGrid();
  await Promise.all(Object.keys(c.stickers).map((k) => new Promise((done) => {
    const [e, a] = k.split("|");
    if (!stickers[e]) return done();
    const url = URL.createObjectURL(c.stickers[k]);
    const img = new Image();
    img.onload = () => { stickers[e][a] = { url, img }; refreshSlotVisual(e, a); done(); };
    img.onerror = () => done();
    img.src = url;
  })));
  localStorage.setItem("mc-active", c.id);
  await renderCharacterSelect();
}

async function renderCharacterSelect() {
  const sel = el("char-select");
  const all = (await dbAll()).sort((a, b) => a.created - b.created);
  sel.innerHTML = "";
  all.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = `${c.name} (${Object.keys(c.stickers).length}/30)`;
    if (activeChar && c.id === activeChar.id) o.selected = true;
    sel.appendChild(o);
  });
  const hc = el("home-char");
  hc.innerHTML = sel.innerHTML;
  hc.value = activeChar ? activeChar.id : "";
  return all;
}

async function initLibrary() {
  try {
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist();
    let all = await dbAll();
    if (!all.length) { await dbPut(newCharacter("Personaje 1")); all = await dbAll(); }
    const id = localStorage.getItem("mc-active");
    await loadCharacter(all.find((c) => c.id === id) || all[0]);
  } catch (err) {
    console.warn("Biblioteca no disponible; los stickers no se guardarán:", err);
    el("char-status").textContent = "El guardado local no está disponible en este dispositivo: los stickers se perderán al cerrar.";
  }
}

el("char-select").addEventListener("change", async (e) => {
  const c = (await dbAll()).find((x) => x.id === e.target.value);
  if (c) await loadCharacter(c);
});
el("btn-char-new").addEventListener("click", async () => {
  const name = (prompt("Nombre del personaje:") || "").trim();
  if (!name) return;
  const c = newCharacter(name); await dbPut(c); await loadCharacter(c);
});
el("btn-char-rename").addEventListener("click", async () => {
  const name = (prompt("Nuevo nombre:", activeChar.name) || "").trim();
  if (!name) return;
  activeChar.name = name; activeChar.modified = Date.now();
  await dbPut(activeChar); await renderCharacterSelect();
});
el("btn-char-dup").addEventListener("click", async () => {
  const c = { ...newCharacter(activeChar.name + " copia"), stickers: { ...activeChar.stickers }, transforms: { ...activeChar.transforms } };
  await dbPut(c); await loadCharacter(c);
});
el("btn-char-del").addEventListener("click", async () => {
  if (!confirm(`¿Eliminar a "${activeChar.name}" y todos sus stickers?`)) return;
  await dbDel(activeChar.id);
  let all = await dbAll();
  if (!all.length) { await dbPut(newCharacter("Personaje 1")); all = await dbAll(); }
  await loadCharacter(all[0]);
});

/* ---------------------------------------------------------------------- */
/* 13. Editor de stickers (transformaciones separadas del PNG original)    */
/* ---------------------------------------------------------------------- */

const DEF_TF = { scale: 1, x: 0, y: 0, rot: 0, mirror: false };
const getTf = (key) => (activeChar && activeChar.transforms && activeChar.transforms[key]) || DEF_TF;
let saveTimer = null;
function saveActiveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => activeChar && dbPut(activeChar).catch(() => {}), 400);
}
function setTf(key, patch) {
  activeChar.transforms = activeChar.transforms || {};
  activeChar.transforms[key] = { ...getTf(key), ...patch };
  activeChar.modified = Date.now();
  saveActiveSoon();
}

let edKey = null;
function drawEditorPreview() {
  const cv = el("ed-canvas"), c = cv.getContext("2d"), S = cv.width, head = S * 0.38;
  c.clearRect(0, 0, S, S);
  const [e, a] = edKey.split("|");
  const st = stickers[e] && stickers[e][a];
  if (st && st.img) {
    const tf = getTf(edKey), base = head * 2.1 * tf.scale, asp = st.img.naturalWidth / st.img.naturalHeight || 1;
    c.save();
    c.translate(S / 2 + tf.x * head, S / 2 + tf.y * head);
    c.rotate((tf.rot * Math.PI) / 180);
    if (tf.mirror) c.scale(-1, 1);
    c.drawImage(st.img, -base / 2, -base / (2 * asp), base, base / asp);
    c.restore();
  }
  c.strokeStyle = "rgba(255,255,255,.45)"; c.setLineDash([6, 6]); c.beginPath();
  c.ellipse(S / 2, S / 2, head * 0.42, head * 0.55, 0, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
}
function syncEditor() {
  const tf = getTf(edKey);
  el("ed-scale").value = tf.scale; el("ed-x").value = tf.x; el("ed-y").value = tf.y;
  el("ed-rot").value = tf.rot; el("ed-mirror").checked = tf.mirror;
  drawEditorPreview();
}
function openEditor(e, a) {
  edKey = e + "|" + a;
  el("ed-title").textContent = `${EMOTIONS.find((x) => x.id === e).label} · ${ANGLES.find((x) => x.id === a).label}`;
  syncEditor();
  el("editor").classList.remove("hidden");
}
[["ed-scale", "scale"], ["ed-x", "x"], ["ed-y", "y"], ["ed-rot", "rot"]].forEach(([id, k]) =>
  el(id).addEventListener("input", (ev) => { setTf(edKey, { [k]: Number(ev.target.value) }); drawEditorPreview(); }));
el("ed-mirror").addEventListener("change", (ev) => { setTf(edKey, { mirror: ev.target.checked }); drawEditorPreview(); });
const allKeys = () => Object.keys(activeChar.stickers);
el("ed-all").addEventListener("click", () => { const t = getTf(edKey); allKeys().forEach((k) => setTf(k, { scale: t.scale, x: t.x, y: t.y, rot: t.rot })); });
el("ed-mirror-all").addEventListener("click", () => { const m = getTf(edKey).mirror; allKeys().forEach((k) => setTf(k, { mirror: m })); });
el("ed-reset").addEventListener("click", () => { if (activeChar.transforms) delete activeChar.transforms[edKey]; saveActiveSoon(); syncEditor(); });
el("ed-reset-all").addEventListener("click", () => { activeChar.transforms = {}; saveActiveSoon(); syncEditor(); });
el("ed-close").addEventListener("click", () => el("editor").classList.add("hidden"));

/* ---------------------------------------------------------------------- */
/* 14. Calibración personal (global: depende de tu rostro, no del personaje) */
/* ---------------------------------------------------------------------- */

const DEF_PEAK = { smile: 0.7, frown: 0.5, jaw: 0.5, browUp: 0.6, browDown: 0.6, eyeWide: 0.5, squint: 0.5 };
let calib = null;
try { calib = JSON.parse(localStorage.getItem("mc-calib") || "null"); } catch { calib = null; }

function faceFeatures(bs) {
  const m = {};
  if (bs && bs.length) bs[0].categories.forEach((c) => (m[c.categoryName] = c.score));
  const g = (k) => m[k] || 0;
  return {
    smile: (g("mouthSmileLeft") + g("mouthSmileRight")) / 2,
    frown: (g("mouthFrownLeft") + g("mouthFrownRight")) / 2,
    jaw: g("jawOpen"),
    browUp: (g("browInnerUp") + g("browOuterUpLeft") + g("browOuterUpRight")) / 3,
    browDown: (g("browDownLeft") + g("browDownRight")) / 2,
    eyeWide: (g("eyeWideLeft") + g("eyeWideRight")) / 2,
    squint: (g("eyeSquintLeft") + g("eyeSquintRight")) / 2,
  };
}
// Reescala cada rasgo al rango personal (reposo→máximo) y lo devuelve a la escala de los umbrales por defecto.
function applyCalib(f) {
  if (!calib) return f;
  const o = {};
  for (const k in f) {
    const b = calib.base[k] || 0, p = calib.peak[k];
    o[k] = p && p - b >= 0.12 ? clamp01((f[k] - b) / (p - b)) * DEF_PEAK[k] : f[k];
  }
  return o;
}

const CALIB_STEPS = [
  { title: "1/8 · De frente", text: "Mirá a la cámara con cara relajada." },
  { title: "2/8 · Izquierda", text: "Girá la cabeza hacia TU izquierda y mantené." },
  { title: "3/8 · Derecha", text: "Girá la cabeza hacia TU derecha y mantené." },
  { title: "4/8 · Sonrisa", text: "Sonreí sin abrir la boca." },
  { title: "5/8 · Risa", text: "Reíte o abrí bien la boca sonriendo." },
  { title: "6/8 · Ojos", text: "Abrí los ojos lo más que puedas." },
  { title: "7/8 · Cejas", text: "Levantá las cejas al máximo." },
  { title: "8/8 · Enojo", text: "Poné cara de enojo: ceño fruncido y boca hacia abajo." },
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const sorted = (a) => [...a].sort((x, y) => x - y);
const p90 = (a) => sorted(a)[Math.min(a.length - 1, Math.floor(a.length * 0.9))];
const median = (a) => sorted(a)[Math.floor(a.length / 2)];
let calibAbort = false;

function computeCalib(d) {
  const keys = Object.keys(DEF_PEAK), base = {}, peak = {};
  keys.forEach((k) => { base[k] = d[0].reduce((t, s) => t + s.f[k], 0) / d[0].length; peak[k] = 0; });
  const offset = median(d[0].map((s) => s.yaw));
  const dl = median(d[1].map((s) => s.yaw - offset)), dr = median(d[2].map((s) => s.yaw - offset));
  const mag = (arr) => Math.max(20, p90(arr.map((s) => Math.abs(s.yaw - offset))));
  for (let i = 3; i < d.length; i++) keys.forEach((k) => (peak[k] = Math.max(peak[k], p90(d[i].map((s) => s.f[k])))));
  return { base, peak, offset, invert: dl > dr, left: mag(d[1]), right: mag(d[2]) };
}
function endCalib(msg) {
  el("calib").classList.add("hidden");
  el("calib-status").textContent = msg || (calib ? "Calibración personal activa." : "Sin calibrar (valores predeterminados).");
}
async function runCalibration() {
  if (!state.faceLandmarker || !el("screen-camera").classList.contains("active")) {
    el("calib-status").textContent = "Abrí la cámara y esperá a que detecte tu rostro antes de calibrar.";
    return;
  }
  closeSettings(); calibAbort = false; el("calib").classList.remove("hidden");
  const data = [];
  for (const st of CALIB_STEPS) {
    el("calib-title").textContent = st.title; el("calib-text").textContent = st.text; el("calib-fill").style.width = "0%";
    for (let n = 2; n > 0; n--) { el("calib-count").textContent = `Preparate… ${n}`; await sleep(1000); if (calibAbort) return endCalib("Calibración cancelada."); }
    el("calib-count").textContent = "¡Ahora!";
    const samples = [], t0 = performance.now();
    while (performance.now() - t0 < 1600) {
      if (calibAbort) return endCalib("Calibración cancelada.");
      if (state.lastFace && state.raw) samples.push(state.raw);
      el("calib-fill").style.width = ((performance.now() - t0) / 16) + "%";
      await sleep(50);
    }
    if (samples.length < 8) return endCalib("No detecté tu rostro en el paso " + st.title.split(" ")[0] + ". Probá de nuevo con mejor luz.");
    data.push(samples);
  }
  calib = computeCalib(data);
  localStorage.setItem("mc-calib", JSON.stringify(calib));
  Object.assign(emoState, { current: "neutral", candidate: "neutral" });
  endCalib("Calibración guardada ✓");
}
el("btn-calib-start").addEventListener("click", runCalibration);
el("btn-calib-cancel").addEventListener("click", () => (calibAbort = true));
el("btn-calib-reset").addEventListener("click", () => { calib = null; localStorage.removeItem("mc-calib"); endCalib("Valores predeterminados restaurados."); });
endCalib();

/* ---------------------------------------------------------------------- */
/* 15. Ajustes de cámara, rendimiento, compatibilidad y avisos            */
/* ---------------------------------------------------------------------- */

const angState = { current: "frontal", candidate: "frontal", since: 0 };
function stabilizeAngle(a, now) {
  if (a === angState.current) { angState.candidate = a; return a; }
  if (a !== angState.candidate) { angState.candidate = a; angState.since = now; }
  else if (now - angState.since >= TUNING.holdMs) angState.current = a;
  return angState.current;
}

let toastTimer;
function toast(msg) {
  const t = el("toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), 3500);
}

const RES = { "480": [640, 480], "720": [1280, 720], "1080": [1920, 1080] };
const camCfg = { adapt: true, res: "720", fps: "30", mic: true, hud: false, recMode: "fluido", mirrorImport: false };
const perfCfg = { profile: "equilibrada", auto: true };
try {
  Object.assign(camCfg, JSON.parse(localStorage.getItem("mc-cam3") || "{}"));
  Object.assign(perfCfg, JSON.parse(localStorage.getItem("mc-perf") || "{}"));
} catch { /* valores por defecto */ }

const PROFILES = {
  alta: { detectEvery: 1, segEvery: 2, recFps: 30, noOcclusion: false },
  equilibrada: { detectEvery: 1, segEvery: 3, recFps: 30, noOcclusion: false },
  ahorro: { detectEvery: 2, segEvery: 6, recFps: 24, noOcclusion: true },
};
const PROFILE_ORDER = ["alta", "equilibrada", "ahorro"];
const P = () => PROFILES[perfCfg.profile] || PROFILES.equilibrada;
let frameN = 0;
const perfMon = { last: 0, ema: 0, slow: 0 };

// Si el promedio baja de ~20 fps durante unos segundos, pasa al perfil inmediatamente más liviano.
function perfTick(now) {
  const dt = now - perfMon.last; perfMon.last = now;
  if (dt <= 0 || dt > 500) return; // pausas o pestaña en segundo plano
  perfMon.ema = perfMon.ema ? perfMon.ema * 0.96 + dt * 0.04 : dt;
  if (perfMon.ema <= 50) { perfMon.slow = 0; return; }
  if (!perfCfg.auto || ++perfMon.slow < 90) return;
  const next = PROFILE_ORDER[PROFILE_ORDER.indexOf(perfCfg.profile) + 1];
  perfMon.slow = 0; perfMon.ema = 0;
  if (!next) return;
  perfCfg.profile = next; saveCfg(); syncCfgUi();
  toast(`Rendimiento reducido automáticamente: modo ${next}.`);
}

const saveCfg = () => { localStorage.setItem("mc-cam3", JSON.stringify(camCfg)); localStorage.setItem("mc-perf", JSON.stringify(perfCfg)); };
function syncCfgUi() {
  el("cfg-res").value = camCfg.res; el("cfg-fps").value = camCfg.fps; el("cfg-mic").checked = camCfg.mic; el("cfg-hud").checked = camCfg.hud; el("cfg-adapt").checked = camCfg.adapt; el("cfg-recmode").value = camCfg.recMode; el("cfg-mirror-import").checked = camCfg.mirrorImport;
  el("cfg-perf").value = perfCfg.profile; el("cfg-perf-auto").checked = perfCfg.auto;
  if (P().noOcclusion && state.handOcclusionEnabled) el("hand-occlusion-status").textContent = "Pausada en modo Ahorro.";
}
async function applyCam() {
  saveCfg();
  if (!el("screen-camera").classList.contains("active")) return;
  if (state.isRecording) { toast("Detené la grabación para cambiar la cámara."); return; }
  try { await startCamera(state.facingMode); } catch (err) { toast(describeCameraError(err)); }
}
[["cfg-res", "res"], ["cfg-fps", "fps"]].forEach(([id, k]) => el(id).addEventListener("change", (e) => { camCfg[k] = e.target.value; applyCam(); }));
el("cfg-mic").addEventListener("change", (e) => { camCfg.mic = e.target.checked; applyCam(); });
el("cfg-perf").addEventListener("change", (e) => { perfCfg.profile = e.target.value; perfMon.slow = 0; saveCfg(); syncCfgUi(); });
el("cfg-perf-auto").addEventListener("change", (e) => { perfCfg.auto = e.target.checked; saveCfg(); });

// Compatibilidad: estado comprensible por función, sin errores técnicos.
function renderCompat() {
  const hasMR = !!window.MediaRecorder, sup = (t) => hasMR && MediaRecorder.isTypeSupported(t);
  const camOn = el("screen-camera").classList.contains("active");
  const model = (m) => (m ? true : camOn ? false : "pend");
  const zoom = !!(state.track && state.track.getCapabilities && state.track.getCapabilities().zoom);
  const rows = [
    ["Cámara", !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)],
    ["Grabación de video", hasMR], ["Formato WebM", sup("video/webm")], ["Formato MP4", sup("video/mp4")],
    ["Seguimiento facial", model(state.faceLandmarker)], ["Oclusión de manos", model(state.imageSegmenter)],
    ["Grabación fluida (WebCodecs)" + (wcInfo && wcInfo.reason ? " [" + wcInfo.reason + "]" : ""), wcInfo ? wcInfo.ok : "pend"],
    ["Zoom de cámara", camOn ? zoom : "pend"],
    ["Instalar como app", matchMedia("(display-mode: standalone)").matches || !el("btn-install").classList.contains("hidden")],
  ];
  el("compat-list").innerHTML = rows.map(([n, ok]) =>
    `<li>${ok === true ? "✓" : ok === "pend" ? "…" : "✗"} ${n}${ok === false ? " — Esta función no está disponible en tu dispositivo." : ok === "pend" ? " — se comprueba al abrir la cámara." : ""}</li>`).join("");
}
el("btn-open-settings").addEventListener("click", renderCompat);
if (!window.MediaRecorder) el("btn-record").classList.add("hidden");

// Inicio: personaje activo + acceso a la biblioteca.
el("home-char").addEventListener("change", (e) => { el("char-select").value = e.target.value; el("char-select").dispatchEvent(new Event("change")); });
el("btn-home-chars").addEventListener("click", () => el("btn-open-settings").click());
syncCfgUi();

/* ---------------------------------------------------------------------- */
/* 16. Grabación a fps altos: fps reales, bitrate y medición              */
/* ---------------------------------------------------------------------- */

let lastPaintT = -1, paintCount = 0, measuredFps = 0;

// fps que entrega realmente la cámara (hasta 60); Ahorro graba a 24.
function recFps() {
  if (perfCfg.profile === "ahorro") return 24;
  const f = state.track && state.track.getSettings ? state.track.getSettings().frameRate : 0;
  return Math.max(15, Math.min(60, Math.round(f || measuredFps || 30)));
}
// Bitrate orientativo: ~0,15 bits por píxel y frame, entre 4 y 16 Mbps (estimación, ajustable).
function recBitrate() {
  const bps = canvas.width * canvas.height * recFps() * 0.15;
  return Math.round(Math.max(4e6, Math.min(16e6, bps)));
}
function detectEvery() {
  if (rawRec) return 4; // modo fluido: la vista previa no compite con el codificador
  return P().detectEvery * (state.isRecording && measuredFps > 40 ? 2 : 1);
}

setInterval(() => {
  measuredFps = paintCount * 2; paintCount = 0;
  const h = el("fps-hud");
  h.classList.toggle("hidden", !camCfg.hud);
  if (camCfg.hud) h.textContent = `${measuredFps} fps · ${canvas.width}×${canvas.height} · ${perfCfg.profile}` + (state.isRecording ? ` · REC ${recFps()} fps ${(recBitrate() / 1e6).toFixed(0)} Mbps` + (wc ? ` · ${wc.frames} frames, ${wc.dropped} perdidos` : "") : "");
}, 500);
el("cfg-hud").addEventListener("change", (e) => { camCfg.hud = e.target.checked; saveCfg(); });

/* ---------------------------------------------------------------------- */
/* 17. Grabación fluida: WebCodecs (H.264 + AAC) y MP4 propio, sin        */
/*     librerías externas. Cada frame de cámara se codifica con su         */
/*     timestamp real. Si algo no está disponible, se usa MediaRecorder.   */
/* ---------------------------------------------------------------------- */

let wc = null, wcInfo = null, recBusy = false;
const toU8 = (x) => new Uint8Array(ArrayBuffer.isView(x) ? x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength) : x.slice(0));

// Escritor MP4 mínimo (faststart): ftyp + moov + mdat, 1 muestra por chunk.
const buildMp4 = (() => {
  const z = (n) => new Array(n).fill(0);
  const u16 = (n) => [(n >> 8) & 255, n & 255];
  const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
  const u32s = (a) => { const o = new Uint8Array(a.length * 4), v = new DataView(o.buffer); a.forEach((x, i) => v.setUint32(i * 4, x >>> 0)); return o; };
  const tag = (t) => [...t].map((c) => c.charCodeAt(0));
  const cat = (...a) => { let n = 0; for (const x of a) n += x.length; const o = new Uint8Array(n); let p = 0; for (const x of a) { o.set(x, p); p += x.length; } return o; };
  const bx = (t, ...c) => { const b = cat(...c); return cat(u32(b.length + 8), tag(t), b); };
  const fb = (t, fl, ...c) => bx(t, [0, (fl >> 16) & 255, (fl >> 8) & 255, fl & 255], ...c);
  const MTX = [...u32(0x10000), ...z(12), ...u32(0x10000), ...z(12), ...u32(0x40000000)];

  function trak(id, t, base) {
    const S = t.samples, n = S.length, tsc = t.timescale, v = t.kind === "v";
    const d = S.map((s, i) => (i < n - 1 ? Math.max(1, Math.round((S[i + 1].ts - s.ts) * tsc)) : 0));
    d[n - 1] = n > 1 ? d[n - 2] : Math.round(tsc / 30);
    const dur = d.reduce((a, b) => a + b, 0); t.sec = dur / tsc;
    let entry;
    if (v) entry = bx("avc1", z(6), u16(1), z(16), u16(t.w), u16(t.h), u32(0x480000), u32(0x480000), z(4), u16(1), z(32), u16(24), [255, 255], bx("avcC", t.desc));
    else {
      const L3 = t.desc.length, L2 = 15 + L3, L1 = L2 + 8;
      entry = bx("mp4a", z(6), u16(1), z(8), u16(t.ch), u16(16), z(4), u32(t.rate * 65536),
        fb("esds", 0, [3, L1, 0, 0, 0, 4, L2, 0x40, 0x15, 0, 0, 0], z(8), [5, L3], t.desc, [6, 1, 2]));
    }
    const keys = S.map((s, i) => (s.key ? i + 1 : 0)).filter(Boolean);
    const stbl = bx("stbl", fb("stsd", 0, u32(1), entry), fb("stts", 0, u32(n), u32s(d.flatMap((x) => [1, x]))),
      v ? fb("stss", 0, u32(keys.length), u32s(keys)) : new Uint8Array(0),
      fb("stsc", 0, u32(1), u32(1), u32(1), u32(1)), fb("stsz", 0, u32(0), u32(n), u32s(S.map((s) => s.size))),
      fb("stco", 0, u32(n), u32s(S.map((s) => base + 8 + s.off))));
    const mdia = bx("mdia", fb("mdhd", 0, u32(0), u32(0), u32(tsc), u32(dur), u16(0x55c4), u16(0)),
      fb("hdlr", 0, u32(0), tag(v ? "vide" : "soun"), z(12), tag(v ? "Video" : "Audio"), [0]),
      bx("minf", v ? fb("vmhd", 1, z(8)) : fb("smhd", 0, z(4)), bx("dinf", fb("dref", 0, u32(1), fb("url ", 1))), stbl));
    const tkhd = fb("tkhd", 3, u32(0), u32(0), u32(id), u32(0), u32(Math.round(t.sec * 1000)), z(8), u16(0), u16(0), u16(v ? 0 : 0x100), u16(0), MTX, u32(v ? t.w * 65536 : 0), u32(v ? t.h * 65536 : 0));
    return bx("trak", tkhd, mdia);
  }
  return (tracks, r) => {
    const moov = (base) => {
      const tr = tracks.map((t, i) => trak(i + 1, t, base));
      const dur = Math.round(Math.max(...tracks.map((t) => t.sec)) * 1000);
      return bx("moov", fb("mvhd", 0, u32(0), u32(0), u32(1000), u32(dur), u32(0x10000), u16(0x100), z(10), MTX, z(24), u32(tracks.length + 1)), ...tr);
    };
    const ftyp = bx("ftyp", tag("isom"), u32(512), tag("isom"), tag("iso2"), tag("avc1"), tag("mp41"));
    let m = moov(0); m = moov(ftyp.length + m.length); // el tamaño del moov no depende de los offsets
    return new Blob([ftyp, m, cat(u32(r.mdatLen + 8), tag("mdat")), ...r.parts], { type: "video/mp4" });
  };
})();

async function pickVideoCfg(w, h) {
  if (!w || !h || w % 2 || h % 2) return null; // H.264 pide medidas pares
  for (const codec of ["avc1.640033", "avc1.4d0033", "avc1.420033"]) {
    const cfg = { codec, width: w, height: h, bitrate: recBitrate(), framerate: recFps(), latencyMode: "realtime", avc: { format: "avc" } };
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return cfg; } catch { /* siguiente */ }
  }
  return null;
}
async function checkWc() {
  const no = (reason) => (wcInfo = { ok: false, reason });
  try {
    if (!window.VideoEncoder || !window.VideoFrame) return no("el navegador no incluye WebCodecs");
    if (!(await pickVideoCfg(canvas.width || 1280, canvas.height || 720))) return no("sin codificador H.264 para esta resolución");
    if (state.stream && state.stream.getAudioTracks()[0]) {
      if (!window.AudioEncoder || !window.MediaStreamTrackProcessor) return no("sin captura/codificación de audio");
      const a = await AudioEncoder.isConfigSupported({ codec: "mp4a.40.2", sampleRate: 48000, numberOfChannels: 1, bitrate: 128000 });
      if (!a.supported) return no("sin codificador AAC");
    }
    return (wcInfo = { ok: true });
  } catch (e) { return no("error: " + e.message); }
}
el("btn-open-settings").addEventListener("click", () => checkWc().then(renderCompat));

async function startWebCodecsRecording() {
  if (recBusy) return true;
  recBusy = true;
  try {
    if (!(await checkWc()).ok) return false;
    const w = canvas.width, h = canvas.height;
    const r = { vt: { kind: "v", timescale: 90000, samples: [], w, h, desc: null }, at: null, parts: [], mdatLen: 0, t0: null, last: -1, frames: 0, dropped: 0, stop: false, failed: false };
    const add = (t, d, ts, key) => { t.samples.push({ size: d.length, ts, key, off: r.mdatLen }); r.parts.push(d); r.mdatLen += d.length; };
    r.venc = new VideoEncoder({
      output: (c, m) => {
        const d = new Uint8Array(c.byteLength); c.copyTo(d);
        if (m && m.decoderConfig && m.decoderConfig.description) r.vt.desc = toU8(m.decoderConfig.description);
        add(r.vt, d, c.timestamp / 1e6, c.type === "key");
      },
      error: (e) => { console.warn(e); r.failed = true; },
    });
    r.venc.configure(await pickVideoCfg(w, h));

    const track = state.stream && state.stream.getAudioTracks()[0];
    if (track) {
      r.at = { kind: "a", timescale: 48000, samples: [], ch: 1, rate: 48000, desc: null };
      (async () => {
        r.reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
        let a0 = 0, off = 0;
        while (!r.stop) {
          const { value: data, done } = await r.reader.read();
          if (done || !data) break;
          if (r.frames === 0 || r.stop) { data.close(); continue; } // el audio empieza con el primer frame de video
          if (!r.aenc) {
            Object.assign(r.at, { rate: data.sampleRate, timescale: data.sampleRate, ch: data.numberOfChannels });
            a0 = data.timestamp; off = Math.max(0, r.last);
            r.aenc = new AudioEncoder({
              output: (c, m) => {
                const d = new Uint8Array(c.byteLength); c.copyTo(d);
                if (m && m.decoderConfig && m.decoderConfig.description && !r.at.desc) r.at.desc = toU8(m.decoderConfig.description);
                add(r.at, d, (c.timestamp - a0) / 1e6 + off, true);
              },
              error: (e) => console.warn(e),
            });
            r.aenc.configure({ codec: "mp4a.40.2", sampleRate: data.sampleRate, numberOfChannels: data.numberOfChannels, bitrate: 128000 });
          }
          r.aenc.encode(data); data.close();
        }
      })().catch((e) => console.warn("audio:", e));
    }
    wc = r;
    state.isRecording = true;
    state.recordStart = Date.now();
    el("btn-record").classList.add("is-recording");
    el("rec-timer").classList.remove("hidden");
    timerInterval = setInterval(updateTimerDisplay, 250);
    return true;
  } catch (err) {
    console.warn("WebCodecs falló; uso MediaRecorder:", err);
    wc = null;
    return false;
  } finally { recBusy = false; }
}

// Una vez por frame de cámara ya dibujado en el canvas.
function feedRecorder() {
  const r = wc;
  if (!r || r.failed) return;
  const t = vfMeta ? vfMeta.mediaTime : video.currentTime;
  if (r.t0 === null) r.t0 = t;
  const ts = t - r.t0;
  if (r.frames > 0 && ts - r.last < 0.8 / recFps()) return;
  if (r.venc.encodeQueueSize > 3) { r.dropped++; return; } // sin cola acumulada: se descarta el frame
  const f = new VideoFrame(canvas, { timestamp: Math.round(ts * 1e6) });
  r.venc.encode(f, { keyFrame: r.frames % (recFps() * 2) === 0 });
  f.close(); r.last = ts; r.frames++;
}

async function stopWebCodecs() {
  const r = wc; wc = null; recBusy = true;
  state.isRecording = false;
  el("btn-record").classList.remove("is-recording");
  el("rec-timer").classList.add("hidden");
  clearInterval(timerInterval);
  toast("Guardando video…");
  try {
    r.stop = true;
    if (r.reader) r.reader.cancel().catch(() => {});
    await r.venc.flush(); if (r.aenc) await r.aenc.flush();
    r.venc.close(); if (r.aenc) r.aenc.close();
    if (!r.vt.samples.length || !r.vt.desc) throw new Error("sin video codificado");
    const tracks = [r.vt];
    if (r.at && r.at.samples.length) {
      if (!r.at.desc) { // AudioSpecificConfig AAC-LC por defecto si el codificador no la entregó
        const idx = { 96000: 0, 88200: 1, 64000: 2, 48000: 3, 44100: 4, 32000: 5, 24000: 6, 22050: 7, 16000: 8 }[r.at.rate] ?? 3;
        r.at.desc = new Uint8Array([(2 << 3) | (idx >> 1), ((idx & 1) << 7) | (r.at.ch << 3)]);
      }
      tracks.push(r.at);
    }
    const blob = buildMp4(tracks, r);
    if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
    recordedBlobUrl = URL.createObjectURL(blob);
    el("preview-video").src = recordedBlobUrl;
    el("btn-download").href = recordedBlobUrl;
    el("btn-download").setAttribute("download", "manga-cam.mp4");
    showScreen("screen-preview");
  } catch (err) {
    console.warn(err);
    toast("No se pudo guardar el video. Probá de nuevo.");
  } finally { recBusy = false; }
}

/* ---------------------------------------------------------------------- */
/* 18. Modo fluido: grabar la cámara directa y agregar el personaje después */
/*     La grabación no pasa por canvas ni seguimiento: es la cámara tal     */
/*     cual. El personaje se dibuja luego, frame a frame, sin apuro de      */
/*     tiempo real (también sirve con videos de la cámara nativa).          */
/* ---------------------------------------------------------------------- */

let procBusy = false, procCancel = false, procTs = 0, clockOverride = null, rawRec = null, rawChunks = [];
const clock = () => (clockOverride === null ? performance.now() : clockOverride);

function uiRecStart() {
  state.isRecording = true; state.recordStart = Date.now();
  el("btn-record").classList.add("is-recording"); el("rec-timer").classList.remove("hidden");
  timerInterval = setInterval(updateTimerDisplay, 250);
}
function uiRecStop() {
  state.isRecording = false;
  el("btn-record").classList.remove("is-recording"); el("rec-timer").classList.add("hidden");
  clearInterval(timerInterval);
}
function showResult(blob, name) {
  if (recordedBlobUrl) URL.revokeObjectURL(recordedBlobUrl);
  recordedBlobUrl = URL.createObjectURL(blob);
  el("preview-video").src = recordedBlobUrl;
  el("btn-download").href = recordedBlobUrl;
  el("btn-download").setAttribute("download", name);
  showScreen("screen-preview");
}

function startRawRecording() {
  if (!window.MediaRecorder || !state.stream) return false;
  const mime = ["video/mp4;codecs=avc1,mp4a.40.2", "video/webm;codecs=h264,opus", "video/webm;codecs=vp8,opus", "video/webm;codecs=vp9,opus", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
  try { rawRec = new MediaRecorder(state.stream, { ...(mime ? { mimeType: mime } : {}), videoBitsPerSecond: 16e6, audioBitsPerSecond: 128e3 }); }
  catch { rawRec = null; return false; }
  rawChunks = [];
  rawRec.ondataavailable = (e) => { if (e.data && e.data.size) rawChunks.push(e.data); };
  rawRec.onstop = async () => {
    const type = (rawRec && rawRec.mimeType) || "video/webm";
    rawRec = null;
    const rawBlob = new Blob(rawChunks, { type });
    toast("Midiendo el video grabado…");
    MET.raw = await analyzeFile(rawBlob);
    processVideo(rawBlob, state.facingMode === "user", true);
  };
  rawRec.start(1000);
  uiRecStart();
  return true;
}

function estimateFps(v) {
  return new Promise((res) => {
    const ts = []; let fin = false;
    const done = async () => {
      if (fin) return; fin = true; v.pause();
      const d = []; for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
      d.sort((a, b) => a - b);
      const m = d[d.length >> 1];
      v.currentTime = 0;
      await Promise.race([new Promise((r) => { v.onseeked = r; }), sleep(600)]);
      res(m > 0 ? Math.max(15, Math.min(120, Math.round(1 / m))) : 30);
    };
    const cb = (n, md) => { ts.push(md.mediaTime); ts.length < 10 ? v.requestVideoFrameCallback(cb) : done(); };
    v.requestVideoFrameCallback(cb);
    v.play().catch(() => done());
    setTimeout(done, 2500);
  });
}

async function encodeAudioTrack(blob, r, add) {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ac.decodeAudioData(await blob.arrayBuffer()); ac.close();
    const ch = Math.min(2, buf.numberOfChannels), rate = buf.sampleRate;
    const cfg = { codec: "mp4a.40.2", sampleRate: rate, numberOfChannels: ch, bitrate: 128000 };
    if (!window.AudioEncoder || !(await AudioEncoder.isConfigSupported(cfg)).supported) return;
    r.at = { kind: "a", timescale: rate, samples: [], ch, rate, desc: null };
    const aenc = new AudioEncoder({
      output: (c, m) => {
        const d = new Uint8Array(c.byteLength); c.copyTo(d);
        if (m && m.decoderConfig && m.decoderConfig.description && !r.at.desc) r.at.desc = toU8(m.decoderConfig.description);
        add(r.at, d, c.timestamp / 1e6, true);
      },
      error: (e) => console.warn(e),
    });
    aenc.configure(cfg);
    const N = 4096;
    for (let i = 0; i < buf.length && !procCancel; i += N) {
      const n = Math.min(N, buf.length - i), data = new Float32Array(n * ch);
      for (let c = 0; c < ch; c++) data.set(buf.getChannelData(c).subarray(i, i + n), c * n);
      const ad = new AudioData({ format: "f32-planar", sampleRate: rate, numberOfFrames: n, numberOfChannels: ch, timestamp: Math.round((i / rate) * 1e6), data });
      aenc.encode(ad); ad.close();
      while (aenc.encodeQueueSize > 16) await sleep(5);
    }
    await aenc.flush(); aenc.close();
  } catch (e) { console.warn("audio:", e); r.at = null; } // archivo sin audio o sin AAC: queda solo el video
}

async function processVideo(blob, mirror, keepRaw) {
  if (procBusy) return;
  procBusy = true; procCancel = false;
  const setP = (txt, pct) => { el("proc-text").textContent = txt; el("proc-fill").style.width = pct + "%"; };
  el("proc").classList.remove("hidden");
  const url = URL.createObjectURL(blob), savedFacing = state.facingMode;
  let result = null;
  try {
    if (!window.VideoEncoder || !window.VideoFrame) throw new Error("tu navegador no incluye WebCodecs");
    setP("Preparando…", 0);
    if (!state.faceLandmarker) await initModels();
    if (!state.faceLandmarker) throw new Error("no se pudo cargar el seguimiento facial");
    const v = document.createElement("video");
    v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
    await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("no se pudo abrir el video")); });
    if (!v.requestVideoFrameCallback) throw new Error("el navegador no permite leer el video frame a frame");
    const W = v.videoWidth & ~1, H = v.videoHeight & ~1, fps = await estimateFps(v);
    const bitrate = Math.round(Math.max(6e6, Math.min(24e6, W * H * Math.max(fps, 30) * 0.18)));
    let cfg = null;
    for (const codec of ["avc1.640034", "avc1.640033", "avc1.4d0033", "avc1.420033"]) {
      const c = { codec, width: W, height: H, bitrate, framerate: fps, latencyMode: "realtime", avc: { format: "avc" } };
      try { if ((await VideoEncoder.isConfigSupported(c)).supported) { cfg = c; break; } } catch { /* siguiente */ }
    }
    if (!cfg) throw new Error("tu teléfono no codifica H.264 a " + W + "×" + H);

    const r = { vt: { kind: "v", timescale: 90000, samples: [], w: W, h: H, desc: null }, at: null, parts: [], mdatLen: 0 };
    const add = (t, d, ts, key) => { t.samples.push({ size: d.length, ts, key, off: r.mdatLen }); r.parts.push(d); r.mdatLen += d.length; };
    let vErr = null;
    const venc = new VideoEncoder({
      output: (c, m) => {
        const d = new Uint8Array(c.byteLength); c.copyTo(d);
        if (m && m.decoderConfig && m.decoderConfig.description) r.vt.desc = toU8(m.decoderConfig.description);
        add(r.vt, d, c.timestamp / 1e6, c.type === "key");
      },
      error: (e) => { vErr = e; },
    });
    venc.configure(cfg);

    // Estado de seguimiento limpio y reloj de medios para suavizado/histéresis.
    canvas.width = W; canvas.height = H;
    state.facingMode = mirror ? "user" : "environment";
    state.smooth = null; state.lastFace = null;
    Object.assign(emoState, { current: "neutral", candidate: "neutral", since: 0 });
    Object.assign(angState, { current: "frontal", candidate: "frontal", since: 0 });

    let frames = 0, last = -1, skipped = 0, lastPres = 0, rate = 1, ema = 0;
    await new Promise((resolve, reject) => {
      v.onended = resolve;
      const cb = async (now, md) => {
        if (procCancel) { v.pause(); return resolve(); }
        if (vErr) return reject(vErr);
        const t0 = performance.now();
        if (md.mediaTime > last) {
          ctx.save();
          if (mirror) { ctx.translate(W, 0); ctx.scale(-1, 1); }
          ctx.drawImage(v, 0, 0, W, H);
          ctx.restore();
          clockOverride = md.mediaTime * 1000;
          try { procTs = Math.max(procTs + 1, performance.now()); updateFaceState(state.faceLandmarker.detectForVideo(v, procTs)); } catch { /* sigue con lo último */ }
          clockOverride = null;
          if (state.lastFace) drawSticker(state.lastFace);
          const f = new VideoFrame(canvas, { timestamp: Math.round(md.mediaTime * 1e6) });
          venc.encode(f, { keyFrame: frames % (fps * 2) === 0 }); f.close();
          frames++; last = md.mediaTime;
          if (md.presentedFrames - lastPres > 1) skipped += md.presentedFrames - lastPres - 1;
        }
        lastPres = md.presentedFrames;
        // Velocidad de lectura adaptativa: más lento si el teléfono no llega, para no saltar frames.
        ema = ema ? ema * 0.9 + (performance.now() - t0) * 0.1 : performance.now() - t0;
        const want = Math.max(0.1, Math.min(1, 1000 / fps / (ema * 1.5 + 4)));
        if (Math.abs(want - rate) > 0.05) { rate = want; v.playbackRate = rate; }
        setP(`Agregando personaje… ${Math.round((md.mediaTime / v.duration) * 100)}% · ${frames} frames · ${rate.toFixed(2)}×`, (md.mediaTime / v.duration) * 100);
        if (venc.encodeQueueSize > 6) { v.pause(); while (venc.encodeQueueSize > 2) await sleep(10); v.play().catch(() => {}); }
        if (!v.ended) v.requestVideoFrameCallback(cb);
      };
      v.requestVideoFrameCallback(cb);
      v.play().catch(reject);
    });
    if (procCancel) throw new Error("cancelado");
    setP("Codificando…", 100);
    await venc.flush(); venc.close();
    if (!r.vt.samples.length || !r.vt.desc) throw new Error("no se codificó ningún frame");
    setP("Procesando audio…", 100);
    await encodeAudioTrack(blob, r, add);
    const tracks = [r.vt];
    if (r.at && r.at.samples.length) {
      if (!r.at.desc) {
        const idx = { 96000: 0, 88200: 1, 64000: 2, 48000: 3, 44100: 4, 32000: 5, 24000: 6, 22050: 7, 16000: 8 }[r.at.rate] ?? 3;
        r.at.desc = new Uint8Array([(2 << 3) | (idx >> 1), ((idx & 1) << 7) | (r.at.ch << 3)]);
      }
      tracks.push(r.at);
    }
    result = buildMp4(tracks, r);
    MET.proc = { framesProcessed: frames, framesSkipped: skipped, sourceFps: fps };
    setP("Midiendo el video final…", 100);
    MET.final = await analyzeFile(result);
    if (skipped) toast(`Listo. ${skipped} frames no se pudieron leer.`);
  } catch (err) {
    console.warn(err);
    if (keepRaw) { showResult(blob, "manga-cam-sin-personaje." + (blob.type.includes("mp4") ? "mp4" : "webm")); toast(err.message === "cancelado" ? "Cancelado. Te dejo el video sin personaje." : "No pude agregar el personaje (" + err.message + "). Te dejo el video sin personaje."); }
    else toast(err.message === "cancelado" ? "Cancelado." : "No pude procesar el video: " + err.message);
  } finally {
    clockOverride = null; state.facingMode = savedFacing;
    URL.revokeObjectURL(url); el("proc").classList.add("hidden"); procBusy = false;
  }
  if (result) showResult(result, "manga-cam.mp4");
}

el("cfg-recmode").addEventListener("change", (e) => { camCfg.recMode = e.target.value; saveCfg(); });
el("cfg-mirror-import").addEventListener("change", (e) => { camCfg.mirrorImport = e.target.checked; saveCfg(); });
el("btn-import").addEventListener("click", () => {
  if (!el("screen-camera").classList.contains("active")) { toast("Abrí la cámara primero y volvé a probar."); return; }
  el("file-import").click();
});
el("file-import").addEventListener("change", (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (f) { closeSettings(); processVideo(f, camCfg.mirrorImport, false); }
});
el("btn-proc-cancel").addEventListener("click", () => { procCancel = true; });


/* ---------------------------------------------------------------------- */
/* 19. Vista en vivo a fps de cámara: seguimiento a su propio ritmo       */
/* ---------------------------------------------------------------------- */

const trackCanvas = document.createElement("canvas");
const trackCtx = trackCanvas.getContext("2d");
let lastDetT = 0, detEma = 0;

// Deja libre al menos la mitad del tiempo para dibujar: la detección nunca bloquea cada frame.
function detInterval() {
  const f = rawRec || P().detectEvery > 1 ? 3 : 2;
  return Math.min(150, detEma * f);
}
function easeFace(face, now) {
  const d = state.disp;
  if (!d) return (state.disp = { ...face, t: now });
  const k = 1 - Math.exp(-(now - d.t) / 45);
  for (const key of ["cxNorm", "cyNorm", "faceW", "faceH", "roll", "yaw"]) d[key] += (face[key] - d[key]) * k;
  d.emotion = face.emotion; d.angle = face.angle; d.t = now;
  return d;
}

/* ---------------------------------------------------------------------- */
/* 20. Métricas reales, diagnóstico y captura adaptativa                  */
/*   presented = frames que el navegador mostró (presentedFrames de rVFC; */
/*               no depende de que el hilo principal esté libre)          */
/*   observed  = callbacks de rVFC atendidos por la app                   */
/*   render    = frames que la app dibujó en el canvas                    */
/* ---------------------------------------------------------------------- */

const hasRvfc = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
let newFrame = false, vfMeta = null, lastDetMeta = null, camFrames = 0, renderCount = 0, detCount = 0;
const MET = { detMs: [], frameMs: [], series: [], raw: null, final: null, proc: null, log: [], errors: [], camNow: 0 };
function armRvfc() { video.requestVideoFrameCallback((now, md) => { camFrames++; vfMeta = md; newFrame = true; armRvfc(); }); }
if (hasRvfc) armRvfc();

// <metrics-pure>
const pushSample = (a, v) => { a.push(v); if (a.length > 600) a.shift(); };
const pct = (a, p) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor((s.length * p) / 100))]; };
const summarize = (a) => (a.length ? { mean: +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(1), min: Math.min(...a), p5: pct(a, 5), max: Math.max(...a), n: a.length } : null);
const verdict = (sm) => (!sm ? "n/d" : sm.mean >= 29 && sm.p5 >= 25 ? "✓ alcanza 30" : "✗ por debajo de 30");
// </metrics-pure>

let lastPres = 0, lastObs = 0, lastRender = 0, lastDet = 0, lastDrop = 0;
setInterval(() => {
  const pres = vfMeta ? vfMeta.presentedFrames : 0;
  const q = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality() : null;
  const s = { rec: !!state.isRecording, pres: Math.max(0, pres - lastPres), obs: camFrames - lastObs, render: renderCount - lastRender, det: detCount - lastDet, drop: q ? Math.max(0, q.droppedVideoFrames - lastDrop) : null };
  lastPres = pres; lastObs = camFrames; lastRender = renderCount; lastDet = detCount; if (q) lastDrop = q.droppedVideoFrames;
  if (video.readyState >= 2 && video.videoWidth && !procBusy) {
    MET.series.push(s); if (MET.series.length > 3600) MET.series.shift();
    MET.camNow = s.pres;
  }
  if (!el("diag").classList.contains("hidden")) renderDiag();
}, 1000);

function analyzeFile(blob, maxSec = 6) {
  return new Promise((res) => {
    const v = document.createElement("video"), u = URL.createObjectURL(blob), ts = [];
    v.muted = true; v.playsInline = true; v.src = u;
    let done = false;
    const fin = () => {
      if (done) return; done = true; v.pause(); URL.revokeObjectURL(u);
      const d = []; for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
      const span = ts.length > 1 ? ts[ts.length - 1] - ts[0] : 0;
      res({ durationSec: isFinite(v.duration) ? +v.duration.toFixed(2) : null, size: `${v.videoWidth}×${v.videoHeight}`, framesObserved: ts.length, observedSeconds: +span.toFixed(2),
        effectiveFps: span > 0 ? +((ts.length - 1) / span).toFixed(1) : null, p95IntervalMs: d.length ? Math.round(pct(d, 95) * 1000) : null, maxGapMs: d.length ? Math.round(Math.max(...d) * 1000) : null,
        note: "medido reproduciendo 1× los primeros segundos; un teléfono muy cargado puede subestimar" });
    };
    v.onerror = () => { if (!done) { done = true; res(null); } };
    v.onended = fin;
    v.onloadedmetadata = () => {
      if (!v.requestVideoFrameCallback) return fin();
      const cb = (n, md) => { ts.push(md.mediaTime); if (md.mediaTime - ts[0] >= maxSec || v.ended) fin(); else v.requestVideoFrameCallback(cb); };
      v.requestVideoFrameCallback(cb); v.play().catch(fin);
    };
    setTimeout(fin, (maxSec + 5) * 1000);
  });
}

function buildReport() {
  const t = state.track, st = t && t.getSettings ? t.getSettings() : {}, cp = t && t.getCapabilities ? t.getCapabilities() : {};
  const rec = MET.series.filter((x) => x.rec), w = rec.length >= 3 ? rec : MET.series.slice(-10);
  const S = (k) => summarize(w.map((x) => x[k]));
  const drops = w.some((x) => x.drop === null) ? "no disponible" : w.reduce((a, x) => a + x.drop, 0);
  return {
    userAgent: navigator.userAgent, when: new Date().toISOString(),
    requested: { res: camCfg.res, fps: camCfg.fps, mode: camCfg.recMode, adaptive: camCfg.adapt, profile: perfCfg.profile },
    effective: { width: st.width, height: st.height, frameRate: st.frameRate, maxFrameRate: cp.frameRate && cp.frameRate.max, maxWidth: cp.width && cp.width.max, maxHeight: cp.height && cp.height.max },
    window: { seconds: w.length, duringRecording: rec.length >= 3, presentedFps: S("pres"), observedFps: S("obs"), renderFps: S("render"), detectionsPerSec: S("det"), droppedVideoFrames: drops },
    detectionMs: { p50: pct(MET.detMs, 50), p95: pct(MET.detMs, 95), p99: pct(MET.detMs, 99) }, frameWorkMs: { p50: pct(MET.frameMs, 50), p95: pct(MET.frameMs, 95) },
    rawFile: MET.raw, finalFile: MET.final, postProcess: MET.proc, adaptiveLog: MET.log, errors: MET.errors,
    verdicts: { camera: verdict(S("pres")), appObserved: verdict(S("obs")), render: verdict(S("render")), rawFile: MET.raw ? (MET.raw.effectiveFps >= 29 ? "✓ alcanza 30" : "✗ por debajo de 30") : "n/d", finalFile: MET.final ? (MET.final.effectiveFps >= 29 ? "✓ alcanza 30" : "✗ por debajo de 30") : "n/d" },
  };
}
function renderDiag() {
  const r = buildReport(), f = (x) => (x == null ? "n/d" : x), m = (x) => (x ? `media ${x.mean} · mín ${x.min} · p5 ${x.p5}` : "n/d");
  el("diag-live").textContent = [
    `Pedido: ${r.requested.res}p · ${r.requested.fps} fps · modo ${r.requested.mode}`,
    `Efectivo (getSettings): ${f(r.effective.width)}×${f(r.effective.height)} @ ${f(r.effective.frameRate)} fps`,
    `Máx. anunciado: ${f(r.effective.maxWidth)}×${f(r.effective.maxHeight)}, ${f(r.effective.maxFrameRate)} fps`,
    `Ventana: ${r.window.seconds} s ${r.window.duringRecording ? "(grabando)" : "(sin grabar)"}`,
    `Cámara mostrada: ${m(r.window.presentedFps)} → ${r.verdicts.camera}`,
    `Atendidos por la app: ${m(r.window.observedFps)} → ${r.verdicts.appObserved}`,
    `Dibujados: ${m(r.window.renderFps)} → ${r.verdicts.render}`,
    `Detecciones/s: ${m(r.window.detectionsPerSec)}`,
    `Latencia detección ms: p50 ${f(r.detectionMs.p50 && Math.round(r.detectionMs.p50))} · p95 ${f(r.detectionMs.p95 && Math.round(r.detectionMs.p95))}`,
    `Trabajo por frame ms: p50 ${f(r.frameWorkMs.p50 && Math.round(r.frameWorkMs.p50))} · p95 ${f(r.frameWorkMs.p95 && Math.round(r.frameWorkMs.p95))}`,
    `Frames descartados: ${r.window.droppedVideoFrames}`,
    `Archivo crudo: ${r.rawFile ? `${r.rawFile.effectiveFps} fps · ${r.rawFile.durationSec} s · mayor salto ${r.rawFile.maxGapMs} ms → ${r.verdicts.rawFile}` : "n/d (grabá primero)"}`,
    `Archivo final: ${r.finalFile ? `${r.finalFile.effectiveFps} fps · ${r.finalFile.durationSec} s · mayor salto ${r.finalFile.maxGapMs} ms → ${r.verdicts.finalFile}` : "n/d"}`,
  ].join("\n");
}
el("btn-diag").addEventListener("click", () => { closeSettings(); renderDiag(); el("diag").classList.remove("hidden"); });
el("btn-diag-close").addEventListener("click", () => el("diag").classList.add("hidden"));
el("btn-diag-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(buildReport(), null, 1)); toast("Informe copiado."); } catch { toast("No pude copiar el informe."); }
});

// Captura adaptativa: baja resolución si la cámara muestra menos de 27 fps durante 3 s; sube si hay margen 15 s.
const STEPS = [[1920, 1080], [1280, 720], [960, 540], [640, 480]];
let badSecs = 0, goodSecs = 0;
async function applyStep(i, why) {
  try {
    await state.track.applyConstraints({ width: { ideal: STEPS[i][0] }, height: { ideal: STEPS[i][1] }, frameRate: { ideal: 30 } });
    MET.log.push(`${why}: ${STEPS[i][0]}×${STEPS[i][1]}`); toast(`Cámara ajustada a ${STEPS[i][1]}p (${why}).`);
  } catch (e) { MET.errors.push(`applyConstraints ${STEPS[i][1]}p: ${e.name}`); }
}
setInterval(() => {
  if (!camCfg.adapt || state.isRecording || procBusy || !state.track || !el("screen-camera").classList.contains("active") || !MET.series.length) return;
  const cam = MET.camNow, h = (state.track.getSettings() || {}).height || 720;
  const cur = Math.max(0, STEPS.findIndex(([, sh]) => sh <= h));
  const top = camCfg.res === "1080" ? 0 : camCfg.res === "480" ? 3 : 1;
  if (cam > 0 && cam < 27) { badSecs++; goodSecs = 0; } else if (cam >= 29) { goodSecs++; badSecs = 0; } else badSecs = goodSecs = 0;
  if (badSecs >= 3 && cur < STEPS.length - 1) { badSecs = 0; applyStep(cur + 1, "bajó: <27 fps"); }
  else if (goodSecs >= 15 && cur > top) { goodSecs = 0; applyStep(cur - 1, "subió: hay margen"); }
}, 1000);
el("cfg-adapt").addEventListener("change", (e) => { camCfg.adapt = e.target.checked; saveCfg(); });

initLibrary();
