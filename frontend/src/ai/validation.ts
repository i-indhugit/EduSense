export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ScoredBBox extends BBox {
  score: number;
  label?: string;
}

export interface NormalizedLandmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export function boxWidth(b: BBox): number {
  return Math.max(0, b.x2 - b.x1);
}

export function boxHeight(b: BBox): number {
  return Math.max(0, b.y2 - b.y1);
}

export function boxArea(b: BBox): number {
  return boxWidth(b) * boxHeight(b);
}

export function computeIoU(a: BBox, b: BBox): number {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = boxArea(a) + boxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

export function applyNms<T extends ScoredBBox>(boxes: T[], iouThreshold = 0.45): T[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: T[] = [];
  for (const cand of sorted) {
    const cArea = Math.max(1, boxArea(cand));
    const duplicates = kept.some((k) => {
      if (computeIoU(k, cand) > iouThreshold) return true;
      // Suppress only true nested duplicate boxes (>= 85% contained inside a higher-scoring box)
      const ix1 = Math.max(k.x1, cand.x1);
      const iy1 = Math.max(k.y1, cand.y1);
      const ix2 = Math.min(k.x2, cand.x2);
      const iy2 = Math.min(k.y2, cand.y2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      return inter / cArea >= 0.85;
    });
    if (!duplicates) kept.push(cand);
  }
  return kept;
}

/**
 * Rejects faces enclosed inside detected screen/book/poster distractor objects
 * (e.g., tv, laptop, cell phone, book).
 */
export function isInsideDistractorScreen(faceBox: BBox, distractors: ScoredBBox[]): boolean {
  const fArea = boxArea(faceBox);
  if (fArea <= 0) return true;
  for (const d of distractors) {
    const ix1 = Math.max(faceBox.x1, d.x1);
    const iy1 = Math.max(faceBox.y1, d.y1);
    const ix2 = Math.min(faceBox.x2, d.x2);
    const iy2 = Math.min(faceBox.y2, d.y2);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    if (inter / fArea >= 0.55) {
      return true;
    }
  }
  return false;
}

/**
 * Associates a detected cell phone with the specific student box holding/using it.
 */
export function isPhoneAssociatedWithPerson(personBox: BBox, distractors: ScoredBBox[]): boolean {
  const pw = boxWidth(personBox);
  for (const d of distractors) {
    if (d.label !== 'cell phone') continue;
    const dcx = (d.x1 + d.x2) / 2;
    const dcy = (d.y1 + d.y2) / 2;
    if (
      computeIoU(personBox, d) > 0 ||
      (dcx >= personBox.x1 - pw * 0.12 &&
        dcx <= personBox.x2 + pw * 0.12 &&
        dcy >= personBox.y1 &&
        dcy <= personBox.y2 + 40)
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Suppresses duplicate overlapping face entries while preserving distinct adjacent faces.
 */
export function suppressDuplicateFaces<T extends { box: BBox; score: number }>(
  faces: T[],
  iouThreshold = 0.50
): T[] {
  const sorted = [...faces].sort((a, b) => b.score - a.score);
  const kept: T[] = [];
  for (const cand of sorted) {
    const cArea = Math.max(1, boxArea(cand.box));
    const isDup = kept.some((k) => {
      if (computeIoU(k.box, cand.box) > iouThreshold) return true;
      const ix1 = Math.max(k.box.x1, cand.box.x1);
      const iy1 = Math.max(k.box.y1, cand.box.y1);
      const ix2 = Math.min(k.box.x2, cand.box.x2);
      const iy2 = Math.min(k.box.y2, cand.box.y2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      return inter / cArea >= 0.80;
    });
    if (!isDup) kept.push(cand);
  }
  return kept;
}

/**
 * Validates whether a face belongs to a confirmed person's upper head region
 * and rejects printed faces on clothing/T-shirts (torso region below shoulder line).
 */
export function isFaceInPersonHeadRegion(
  faceBox: BBox,
  personBox: BBox,
  poseLandmarks?: NormalizedLandmark[] | null,
  imgW = 640,
  imgH = 480
): { valid: boolean; reason: string } {
  const pw = boxWidth(personBox);
  const ph = boxHeight(personBox);
  const fw = boxWidth(faceBox);
  const fh = boxHeight(faceBox);

  if (pw <= 10 || ph <= 10 || fw < 20 || fh < 20) {
    return { valid: false, reason: 'Box too small' };
  }

  const aspect = fw / Math.max(1, fh);
  if (aspect < 0.55 || aspect > 1.75) {
    return { valid: false, reason: 'Implausible face aspect ratio' };
  }

  const fcx = (faceBox.x1 + faceBox.x2) / 2;
  const fcy = (faceBox.y1 + faceBox.y2) / 2;

  // Face center must lie horizontally inside the person box (with small margin)
  if (fcx < personBox.x1 - pw * 0.08 || fcx > personBox.x2 + pw * 0.08) {
    return { valid: false, reason: 'Face outside person horizontal bounds' };
  }

  // Face width shouldn't exceed person width or be tiny (< 10% of person width)
  if (fw > pw * 0.95 || fw < pw * 0.10) {
    return { valid: false, reason: 'Face-to-person scale mismatch' };
  }

  // If pose landmarks are available, check shoulder line (indices 11, 12) directly
  if (poseLandmarks && poseLandmarks.length >= 13) {
    const ls = poseLandmarks[11];
    const rs = poseLandmarks[12];
    const lsVis = ls?.visibility ?? 0;
    const rsVis = rs?.visibility ?? 0;
    if (lsVis > 0.4 && rsVis > 0.4) {
      const shoulderY = ((ls.y + rs.y) / 2) * imgH;
      const lsX = ls.x * imgW;
      const rsX = rs.x * imgW;
      const minSX = Math.min(lsX, rsX) - pw * 0.15;
      const maxSX = Math.max(lsX, rsX) + pw * 0.15;
      if (fcx >= minSX && fcx <= maxSX && fcy > shoulderY + ph * 0.04) {
        return { valid: false, reason: 'Face below pose shoulder line (T-shirt/torso rejection)' };
      }
      return { valid: true, reason: 'Validated above pose shoulder line' };
    }
  }

  // Fallback when pose shoulders aren't visible: face must be in upper 54% of person box
  if (fcy < personBox.y1 - ph * 0.05 || fcy > personBox.y1 + ph * 0.54) {
    return { valid: false, reason: 'Face in torso/lower region (possible clothing print)' };
  }

  return { valid: true, reason: 'Validated in person head region' };
}

/**
 * Validates MediaPipe 468/478-point facial landmarks for anatomical plausibility
 * and computes a normalized landmark quality score [0..1].
 */
export function validateFacialLandmarks(
  landmarks: NormalizedLandmark[] | null | undefined,
  faceBox: BBox,
  imgW: number,
  imgH: number
): { valid: boolean; qualityScore: number; reason: string } {
  if (!landmarks || landmarks.length < 468) {
    return { valid: false, qualityScore: 0, reason: 'Missing facial landmarks' };
  }

  const nose = landmarks[1];
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  const lx = leftEye.x * imgW;
  const ly = leftEye.y * imgH;
  const rx = rightEye.x * imgW;
  const ry = rightEye.y * imgH;
  const nx = nose.x * imgW;
  const ny = nose.y * imgH;
  const fy = forehead.y * imgH;
  const cy = chin.y * imgH;

  const eyeDist = Math.hypot(rx - lx, ry - ly);
  const faceHeightLm = Math.abs(cy - fy);
  const fw = boxWidth(faceBox);
  const fh = boxHeight(faceBox);

  if (eyeDist < Math.max(8, fw * 0.18) || faceHeightLm < Math.max(12, fh * 0.35)) {
    return { valid: false, qualityScore: 0.15, reason: 'Degenerate landmark geometry' };
  }

  // Nose must sit roughly between/below eyes and above chin
  const eyeMidY = (ly + ry) / 2;
  if (ny < eyeMidY - fh * 0.1 || ny > cy + fh * 0.05) {
    return { valid: false, qualityScore: 0.2, reason: 'Anatomically inconsistent nose/eye/chin order' };
  }

  // Ensure nose is reasonably within face box
  if (nx < faceBox.x1 - fw * 0.15 || nx > faceBox.x2 + fw * 0.15) {
    return { valid: false, qualityScore: 0.2, reason: 'Landmarks misaligned with face box' };
  }

  const sizeFactor = Math.min(1, Math.min(fw, fh) / 110);
  const geoFactor = Math.min(1, eyeDist / Math.max(1, fw * 0.38));
  const qualityScore = Math.max(0.35, Math.min(1, 0.55 + 0.25 * sizeFactor + 0.2 * geoFactor));

  return { valid: true, qualityScore, reason: 'Valid facial landmarks' };
}

