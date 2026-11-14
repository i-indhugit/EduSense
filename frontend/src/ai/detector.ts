import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
  ObjectDetector,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import { ClassroomMetrics, StudentAnalytics } from '../types/analytics';
import { BlendshapeCategory, extractBehavioralFeatures } from './features';
import { BrowserEngagementFusion } from './fusion';
import { BrowserByteTracker, DetectionCandidate } from './tracker';
import {
  applyNms,
  BBox,
  computeIoU,
  isFaceInPersonHeadRegion,
  isInsideDistractorScreen,
  isPhoneAssociatedWithPerson,
  NormalizedLandmark,
  ScoredBBox,
  suppressDuplicateFaces,
  validateFacialLandmarks,
} from './validation';

export type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface BrowserAiDiagnostics {
  status: ModelStatus;
  delegate: 'GPU' | 'CPU';
  error: string | null;
  loadedModels: string[];
  browserSupported: boolean;
  memoryMb: number | null;
}

const DISTRACTOR_LABELS = new Set(['tv', 'laptop', 'cell phone', 'book', 'monitor']);

export function getBrowserMemoryMb(): number | null {
  const perf = performance as Performance & {
    memory?: { usedJSHeapSize?: number };
  };
  if (perf.memory && typeof perf.memory.usedJSHeapSize === 'number') {
    return Number((perf.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1));
  }
  return null;
}

export class BrowserAiPipeline {
  private objectDetector: ObjectDetector | null = null;
  private faceDetector: FaceDetector | null = null;
  private faceLandmarker: FaceLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;

  private tracker = new BrowserByteTracker(15, 0.25, 2);
  private fusion = new BrowserEngagementFusion();

  private frameCounter = 0;
  private poseInterval = 2;
  private lastPoses: NormalizedLandmark[][] = [];
  private fpsHistory: number[] = [];
  private lastFrameTs = 0;

  public diagnostics: BrowserAiDiagnostics = {
    status: 'idle',
    delegate: 'GPU',
    error: null,
    loadedModels: [],
    browserSupported: typeof WebAssembly === 'object',
    memoryMb: null,
  };

  public resetSession(): void {
    this.tracker.reset();
    this.fusion.reset();
    this.frameCounter = 0;
    this.lastPoses = [];
    this.fpsHistory = [];
    this.lastFrameTs = 0;
  }

  public async initialize(onStatus?: (diag: BrowserAiDiagnostics) => void): Promise<boolean> {
    if (this.diagnostics.status === 'ready') return true;
    if (typeof WebAssembly !== 'object') {
      this.diagnostics = {
        ...this.diagnostics,
        status: 'error',
        browserSupported: false,
        error: 'WebAssembly is not supported in this browser.',
      };
      onStatus?.(this.diagnostics);
      return false;
    }

    this.diagnostics = {
      ...this.diagnostics,
      status: 'loading',
      error: null,
      loadedModels: [],
    };
    onStatus?.(this.diagnostics);

    const tryInitWithDelegate = async (delegate: 'GPU' | 'CPU') => {
      let vision;
      try {
        vision = await FilesetResolver.forVisionTasks('/wasm');
      } catch {
        vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
      }
      const loaded: string[] = [];

      const loadModel = async <T>(
        loader: (path: string) => Promise<T>,
        localPath: string,
        fallbackUrl: string
      ): Promise<T> => {
        try {
          return await loader(localPath);
        } catch {
          return await loader(fallbackUrl);
        }
      };

      const objDet = await loadModel(
        (path) =>
          ObjectDetector.createFromOptions(vision, {
            baseOptions: { modelAssetPath: path, delegate },
            runningMode: 'VIDEO',
            scoreThreshold: 0.32,
            maxResults: 20,
          }),
        '/models/efficientdet_lite0.tflite',
        'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite'
      );
      loaded.push('ObjectDetector (efficientdet_lite0)');

      const faceDet = await loadModel(
        (path) =>
          FaceDetector.createFromOptions(vision, {
            baseOptions: { modelAssetPath: path, delegate },
            runningMode: 'VIDEO',
            minDetectionConfidence: 0.45,
          }),
        '/models/blaze_face_short_range.tflite',
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
      );
      loaded.push('FaceDetector (blaze_face_short_range)');

      const faceLm = await loadModel(
        (path) =>
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: path, delegate },
            runningMode: 'VIDEO',
            numFaces: 10,
            minFaceDetectionConfidence: 0.45,
            minFacePresenceConfidence: 0.45,
            minTrackingConfidence: 0.45,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
          }),
        '/models/face_landmarker.task',
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
      );
      loaded.push('FaceLandmarker (478-pt + blendshapes)');

      const poseLm = await loadModel(
        (path) =>
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: path, delegate },
            runningMode: 'VIDEO',
            numPoses: 10,
            minPoseDetectionConfidence: 0.40,
            minPosePresenceConfidence: 0.40,
            minTrackingConfidence: 0.40,
          }),
        '/models/pose_landmarker_lite.task',
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
      );
      loaded.push('PoseLandmarker (lite)');

      return { objDet, faceDet, faceLm, poseLm, loaded };
    };

    try {
      let delegate: 'GPU' | 'CPU' = 'GPU';
      let res;
      try {
        res = await tryInitWithDelegate('GPU');
      } catch {
        delegate = 'CPU';
        res = await tryInitWithDelegate('CPU');
      }

      this.objectDetector = res.objDet;
      this.faceDetector = res.faceDet;
      this.faceLandmarker = res.faceLm;
      this.poseLandmarker = res.poseLm;

      this.diagnostics = {
        status: 'ready',
        delegate,
        error: null,
        loadedModels: res.loaded,
        browserSupported: true,
        memoryMb: getBrowserMemoryMb(),
      };
      onStatus?.(this.diagnostics);
      return true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load browser MediaPipe models.';
      this.diagnostics = {
        ...this.diagnostics,
        status: 'error',
        error: `${msg} Corrective step: check network, verify public/models assets, or enable hardware acceleration in browser settings.`,
        loadedModels: [],
        browserSupported: true,
        memoryMb: getBrowserMemoryMb(),
      };
      onStatus?.(this.diagnostics);
      return false;
    }
  }

  public dispose(): void {
    try {
      this.objectDetector?.close();
      this.faceDetector?.close();
      this.faceLandmarker?.close();
      this.poseLandmarker?.close();
    } catch {
      // Ignore close errors
    }
    this.objectDetector = null;
    this.faceDetector = null;
    this.faceLandmarker = null;
    this.poseLandmarker = null;
    this.diagnostics.status = 'idle';
  }

  public processVideoFrame(
    source: HTMLVideoElement | HTMLCanvasElement | HTMLImageElement,
    timestampMs: number,
    cameraConnected: boolean
  ): ClassroomMetrics | null {
    if (
      this.diagnostics.status !== 'ready' ||
      !this.objectDetector ||
      !this.faceDetector ||
      !this.faceLandmarker ||
      !this.poseLandmarker
    ) {
      return null;
    }

    const isVideo = source instanceof HTMLVideoElement;
    const isImg = source instanceof HTMLImageElement;
    if (isVideo && (source.readyState < 2 || source.videoWidth === 0 || source.videoHeight === 0)) {
      return null;
    }
    if (isImg && (!source.complete || source.naturalWidth === 0 || source.naturalHeight === 0)) {
      return null;
    }

    const t0 = performance.now();
    const imgW = isVideo ? source.videoWidth : isImg ? source.naturalWidth : (source as HTMLCanvasElement).width;
    const imgH = isVideo ? source.videoHeight : isImg ? source.naturalHeight : (source as HTMLCanvasElement).height;
    if (imgW === 0 || imgH === 0) return null;

    this.frameCounter += 1;

    // 1. Person & Distractor Object Detection
    const objRes = this.objectDetector.detectForVideo(source, timestampMs);
    const rawPersons: ScoredBBox[] = [];
    const distractors: ScoredBBox[] = [];

    for (const det of objRes.detections || []) {
      const cat = det.categories?.[0];
      const bb = det.boundingBox;
      if (!cat || !bb) continue;
      const label = (cat.categoryName || '').toLowerCase();
      const box: ScoredBBox = {
        x1: Math.max(0, bb.originX),
        y1: Math.max(0, bb.originY),
        x2: Math.min(imgW, bb.originX + bb.width),
        y2: Math.min(imgH, bb.originY + bb.height),
        score: cat.score,
        label,
      };
      if (label === 'person' && cat.score >= 0.32) {
        if (box.x2 - box.x1 >= 32 && box.y2 - box.y1 >= 44) {
          rawPersons.push(box);
        }
      } else if (DISTRACTOR_LABELS.has(label) && cat.score >= 0.38) {
        distractors.push(box);
      }
    }

    let persons = applyNms(rawPersons, 0.50);

    // 2. Throttled PoseLandmarker
    if (this.frameCounter % this.poseInterval === 1 || this.lastPoses.length === 0) {
      const poseRes = this.poseLandmarker.detectForVideo(source, timestampMs);
      this.lastPoses = (poseRes.landmarks || []) as NormalizedLandmark[][];
    }

    // Multi-person pose fallback: if ObjectDetector missed or merged adjacent seated students,
    // synthesize a person region for each distinct human pose (nose + both shoulders visible).
    for (const pose of this.lastPoses) {
      const nose = pose[0];
      const ls = pose[11];
      const rs = pose[12];
      if (!nose || !ls || !rs) continue;
      if ((nose.visibility ?? 0) < 0.55 || (ls.visibility ?? 0) < 0.55 || (rs.visibility ?? 0) < 0.55) {
        continue;
      }
      const shoulderW = Math.abs(ls.x - rs.x) * imgW;
      if (shoulderW < 40) continue;

      const synthBox: ScoredBBox = {
        x1: Math.max(0, Math.min(ls.x, rs.x) * imgW - shoulderW * 0.25),
        y1: Math.max(0, nose.y * imgH - shoulderW * 0.55),
        x2: Math.min(imgW, Math.max(ls.x, rs.x) * imgW + shoulderW * 0.25),
        y2: Math.min(imgH, ((ls.y + rs.y) / 2) * imgH + shoulderW * 1.1),
        score: 0.72,
        label: 'person',
      };

      const alreadyCovered = persons.some((p) => computeIoU(synthBox, p) > 0.32);
      if (!alreadyCovered) {
        persons.push(synthBox);
      }
    }
    persons = applyNms(persons, 0.50);

    // Match each confirmed person to closest pose
    const personPoses: Array<NormalizedLandmark[] | null> = persons.map((p) => {
      let bestPose: NormalizedLandmark[] | null = null;
      let bestScore = 0;
      for (const pose of this.lastPoses) {
        const ls = pose[11];
        const rs = pose[12];
        if (!ls || !rs) continue;
        const cx = ((ls.x + rs.x) / 2) * imgW;
        const cy = ((ls.y + rs.y) / 2) * imgH;
        if (cx >= p.x1 && cx <= p.x2 && cy >= p.y1 && cy <= p.y2) {
          const vis = ((ls.visibility ?? 0.5) + (rs.visibility ?? 0.5)) / 2;
          if (vis > bestScore) {
            bestScore = vis;
            bestPose = pose;
          }
        }
      }
      return bestPose;
    });

    // 3. Face Landmarker (primary) + FaceDetector confidence boost, validated against confirmed person head regions
    const faceDetRes = this.faceDetector.detectForVideo(source, timestampMs);
    const faceLmRes = this.faceLandmarker.detectForVideo(source, timestampMs);

    const detFaces: ScoredBBox[] = [];
    for (const det of faceDetRes.detections || []) {
      const bb = det.boundingBox;
      const score = det.categories?.[0]?.score ?? 0.6;
      if (!bb) continue;
      detFaces.push({
        x1: Math.max(0, bb.originX),
        y1: Math.max(0, bb.originY),
        x2: Math.min(imgW, bb.originX + bb.width),
        y2: Math.min(imgH, bb.originY + bb.height),
        score,
      });
    }

    // Build bounding boxes for each FaceLandmarker result (works at both close & classroom distances)
    const lmEntries = (faceLmRes.faceLandmarks || []).map((lms, idx) => {
      let minX = imgW, minY = imgH, maxX = 0, maxY = 0;
      for (const lm of lms) {
        const px = lm.x * imgW;
        const py = lm.y * imgH;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
      }
      const box: BBox = {
        x1: Math.max(0, minX),
        y1: Math.max(0, minY),
        x2: Math.min(imgW, maxX),
        y2: Math.min(imgH, maxY),
      };
      const blendshapes = faceLmRes.faceBlendshapes?.[idx]?.categories as BlendshapeCategory[] | undefined;
      const transformMatrix = faceLmRes.facialTransformationMatrixes?.[idx]?.data as number[] | undefined;
      let detScore = 0.78;
      for (const df of detFaces) {
        if (computeIoU(box, df) > 0.25) {
          detScore = Math.max(detScore, df.score);
        }
      }
      return { lms: lms as NormalizedLandmark[], box, blendshapes, transformMatrix, score: detScore };
    });
    const dedupedLmEntries = suppressDuplicateFaces(lmEntries, 0.50);

    // Pair faces strictly to confirmed person regions (never floating/poster/T-shirt faces)
    const candidates: DetectionCandidate[] = [];
    const personFeaturesMeta: Array<{
      faceLandmarks: NormalizedLandmark[] | null;
      blendshapes: BlendshapeCategory[] | undefined;
      transformMatrix: number[] | undefined;
      poseLandmarks: NormalizedLandmark[] | null;
      faceQuality: number;
      phoneDetected: boolean;
    }> = [];

    const usedLmIndices = new Set<number>();

    persons.forEach((person, pIdx) => {
      const pose = personPoses[pIdx];
      let matchedLmIdx = -1;
      let matchedQuality = 0;

      const phoneDetected = isPhoneAssociatedWithPerson(person, distractors);

      dedupedLmEntries.forEach((entry, eIdx) => {
        if (usedLmIndices.has(eIdx)) return;
        if (isInsideDistractorScreen(entry.box, distractors)) return;
        const headCheck = isFaceInPersonHeadRegion(entry.box, person, pose, imgW, imgH);
        if (!headCheck.valid) return;
        const lmCheck = validateFacialLandmarks(entry.lms, entry.box, imgW, imgH);
        if (!lmCheck.valid) return;
        if (lmCheck.qualityScore > matchedQuality) {
          matchedQuality = lmCheck.qualityScore;
          matchedLmIdx = eIdx;
        }
      });

      let validFaceBox: BBox | null = null;
      let faceConfidence = 0;
      let faceQuality = 0;
      let matchedLms: NormalizedLandmark[] | null = null;
      let matchedBlendshapes: BlendshapeCategory[] | undefined;
      let matchedMatrix: number[] | undefined;

      if (matchedLmIdx !== -1) {
        usedLmIndices.add(matchedLmIdx);
        const entry = dedupedLmEntries[matchedLmIdx];
        validFaceBox = entry.box;
        faceConfidence = entry.score;
        faceQuality = matchedQuality;
        matchedLms = entry.lms;
        matchedBlendshapes = entry.blendshapes;
        matchedMatrix = entry.transformMatrix;
      }

      candidates.push({
        personBox: person,
        faceBox: validFaceBox,
        confidence: person.score,
        faceConfidence,
        qualityScore: faceQuality,
      });

      personFeaturesMeta.push({
        faceLandmarks: matchedLms,
        blendshapes: matchedBlendshapes,
        transformMatrix: matchedMatrix,
        poseLandmarks: pose,
        faceQuality,
        phoneDetected,
      });
    });

    // If ObjectDetector merged side-by-side students into one wide person box,
    // emit additional validated faces inside that wide person's upper head region.
    dedupedLmEntries.forEach((entry, eIdx) => {
      if (usedLmIndices.has(eIdx)) return;
      if (isInsideDistractorScreen(entry.box, distractors)) return;
      const hostPersonIdx = persons.findIndex((p, pIdx) =>
        isFaceInPersonHeadRegion(entry.box, p, personPoses[pIdx], imgW, imgH).valid
      );
      if (hostPersonIdx === -1) return;
      const lmCheck = validateFacialLandmarks(entry.lms, entry.box, imgW, imgH);
      if (!lmCheck.valid) return;

      const host = persons[hostPersonIdx];
      // Narrow primary candidate's personBox around its matched face so split students don't overlap
      const primaryCand = candidates[hostPersonIdx];
      if (primaryCand?.faceBox) {
        const pfw = Math.max(30, primaryCand.faceBox.x2 - primaryCand.faceBox.x1);
        const pfcx = (primaryCand.faceBox.x1 + primaryCand.faceBox.x2) / 2;
        primaryCand.personBox = {
          x1: Math.max(host.x1, pfcx - pfw * 1.4),
          y1: host.y1,
          x2: Math.min(host.x2, pfcx + pfw * 1.4),
          y2: host.y2,
        };
        personFeaturesMeta[hostPersonIdx].phoneDetected = isPhoneAssociatedWithPerson(
          primaryCand.personBox,
          distractors
        );
      }
      const fw = Math.max(30, entry.box.x2 - entry.box.x1);
      const fcx = (entry.box.x1 + entry.box.x2) / 2;
      const subPerson: BBox = {
        x1: Math.max(host.x1, fcx - fw * 1.4),
        y1: host.y1,
        x2: Math.min(host.x2, fcx + fw * 1.4),
        y2: host.y2,
      };
      usedLmIndices.add(eIdx);
      candidates.push({
        personBox: subPerson,
        faceBox: entry.box,
        confidence: host.score,
        faceConfidence: entry.score,
        qualityScore: lmCheck.qualityScore,
      });
      personFeaturesMeta.push({
        faceLandmarks: entry.lms,
        blendshapes: entry.blendshapes,
        transformMatrix: entry.transformMatrix,
        poseLandmarks: personPoses[hostPersonIdx],
        faceQuality: lmCheck.qualityScore,
        phoneDetected: isPhoneAssociatedWithPerson(subPerson, distractors),
      });
    });

    // 4. Multi-person ByteTrack update (only emit confirmed, currently visible tracks)
    const allTracks = this.tracker.update(candidates);
    const activeIds = new Set(allTracks.map((t) => t.trackId));
    this.fusion.prune(activeIds);

    const visibleConfirmedTracks = allTracks.filter(
      (t) => t.visibleThisFrame && t.hits >= this.tracker.minHitsToConfirm
    );

    // 5. Behavioral feature extraction & fusion per active visible track
    const students: StudentAnalytics[] = [];
    for (const track of visibleConfirmedTracks) {
      const meta =
        track.candidateIndex >= 0 && track.candidateIndex < personFeaturesMeta.length
          ? personFeaturesMeta[track.candidateIndex]
          : null;
      const feat = extractBehavioralFeatures(
        track.visibleThisFrame && track.hasValidFace && meta ? meta.faceLandmarks : null,
        meta?.blendshapes,
        meta?.poseLandmarks,
        meta?.faceQuality ?? 0,
        imgW,
        imgH,
        meta?.phoneDetected ?? false,
        meta?.transformMatrix
      );

      students.push(this.fusion.evaluateStudent(track, feat));
    }

    const latencyMs = performance.now() - t0;
    if (this.lastFrameTs > 0) {
      const dt = (timestampMs - this.lastFrameTs) / 1000;
      if (dt > 0 && dt < 2) {
        this.fpsHistory.push(1 / dt);
        if (this.fpsHistory.length > 15) this.fpsHistory.shift();
      }
    }
    this.lastFrameTs = timestampMs;
    const avgFps =
      this.fpsHistory.length > 0
        ? this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length
        : 0;

    this.diagnostics.memoryMb = getBrowserMemoryMb();

    const validFaceCount = students.filter((s) => s.face_detected).length;
    return this.fusion.aggregateClassroom(
      students,
      visibleConfirmedTracks.length,
      validFaceCount,
      avgFps,
      latencyMs,
      cameraConnected
    );
  }
}

const renderBoxCache = new Map<string, [number, number, number, number]>();

export function drawOverlayBoxes(
  canvas: HTMLCanvasElement,
  videoW: number,
  videoH: number,
  students: StudentAnalytics[]
): void {
  if (videoW <= 0 || videoH <= 0) return;
  if (canvas.width !== videoW) canvas.width = videoW;
  if (canvas.height !== videoH) canvas.height = videoH;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, videoW, videoH);

  // Clean stale tracks from render cache
  const activeIds = new Set(students.map((s) => s.student_id));
  for (const id of renderBoxCache.keys()) {
    if (!activeIds.has(id)) renderBoxCache.delete(id);
  }

  for (const s of students) {
    // Draw person box (subtle slate)
    if (s.person_bbox && s.person_bbox.length === 4) {
      const [px1, py1, px2, py2] = s.person_bbox;
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
    }

    // Draw validated face box with anti-shaking deadband filter
    if (s.face_detected && s.bbox && s.bbox.length === 4) {
      const raw = s.bbox;
      let smooth = renderBoxCache.get(s.student_id);
      if (!smooth) {
        smooth = [raw[0], raw[1], raw[2], raw[3]];
        renderBoxCache.set(s.student_id, smooth);
      } else {
        // Deadband filter: if box moved < 4px, freeze it in place (zero tremble/shake!)
        const dx1 = Math.abs(raw[0] - smooth[0]);
        const dy1 = Math.abs(raw[1] - smooth[1]);
        const dx2 = Math.abs(raw[2] - smooth[2]);
        const dy2 = Math.abs(raw[3] - smooth[3]);
        if (dx1 >= 4 || dy1 >= 4 || dx2 >= 4 || dy2 >= 4) {
          const alpha = 0.22;
          smooth[0] = Math.round(smooth[0] + (raw[0] - smooth[0]) * alpha);
          smooth[1] = Math.round(smooth[1] + (raw[1] - smooth[1]) * alpha);
          smooth[2] = Math.round(smooth[2] + (raw[2] - smooth[2]) * alpha);
          smooth[3] = Math.round(smooth[3] + (raw[3] - smooth[3]) * alpha);
        }
      }

      const [fx1, fy1, fx2, fy2] = smooth;
      const color =
        s.final_state === 'DROWSY'
          ? '#f43f5e'
          : s.final_state === 'PHONE DISTRACTED'
          ? '#a855f7'
          : s.final_state === 'ATTENTIVE'
          ? '#10b981'
          : s.final_state === 'MODERATE'
          ? '#f59e0b'
          : '#64748b';

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(fx1, fy1, fx2 - fx1, fy2 - fy1);

      // Primary State Pill
      const phoneIcon = s.final_state === 'PHONE DISTRACTED' ? '📱 ' : '';
      const primaryLabel = `${phoneIcon}${s.student_id} • ${s.final_state} (${Math.round(s.attention_pct)}%)`;
      ctx.font = '600 12px Inter, system-ui, sans-serif';
      const textW = ctx.measureText(primaryLabel).width;
      const pillY = Math.max(0, fy1 - 24);
      ctx.fillStyle = 'rgba(9, 9, 11, 0.88)';
      ctx.fillRect(fx1, pillY, textW + 14, 22);
      ctx.fillStyle = color;
      ctx.fillText(primaryLabel, fx1 + 7, pillY + 15);

      // Secondary Live Behavioral Cue Sub-Pill (e.g., "Eyes: Open/Closed • Gaze • AI Suggestion")
      const eyeText = s.is_drowsy ? 'Eyes: CLOSED' : (s.ear_value != null && s.ear_value < 0.23) ? 'Eyes: Droop' : 'Eyes: Open';
      const actionText = s.ai_suggestion?.action ? ` • AI: ${s.ai_suggestion.action}` : '';
      const cueText = `${eyeText} • Gaze: ${s.gaze_direction}${actionText}`;
      ctx.font = '500 10px Inter, system-ui, sans-serif';
      const subW = ctx.measureText(cueText).width;
      const subPillY = fy2 + 4;
      if (subPillY + 18 <= videoH) {
        ctx.fillStyle = 'rgba(9, 9, 11, 0.85)';
        ctx.fillRect(fx1, subPillY, subW + 10, 18);
        ctx.fillStyle = '#cbd5e1';
        ctx.fillText(cueText, fx1 + 5, subPillY + 13);
      }
    }
  }
}

