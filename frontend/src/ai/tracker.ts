import { BBox, boxHeight, boxWidth, computeIoU } from './validation';

export interface DetectionCandidate {
  personBox: BBox;
  faceBox: BBox | null;
  confidence: number;
  faceConfidence: number;
  qualityScore: number;
}

export interface TrackedPerson {
  trackId: number;
  studentId: string;
  personBox: BBox;
  faceBox: BBox | null;
  confidence: number;
  faceConfidence: number;
  qualityScore: number;
  trackConfidence: number;
  lifecycleState: 'tentative' | 'confirmed' | 'occluded';
  visibleThisFrame: boolean;
  hasValidFace: boolean;
  missingFrames: number;
  ageFrames: number;
  hits: number;
  candidateIndex: number;
}

/**
 * Lightweight browser-side ByteTrack-style multi-object tracker.
 * - Two-stage matching: high-confidence (>= 0.55) then low-confidence (>= 0.35)
 * - Monotonic student track IDs (never random per frame)
 * - Zero stale face boxes when occluded or missing
 * - Automatic expiration after maxMissingFrames
 */
export class BrowserByteTracker {
  private nextTrackId = 1;
  private tracks: Map<number, TrackedPerson> = new Map();
  private readonly maxMissingFrames: number;
  private readonly iouThreshold: number;
  public readonly minHitsToConfirm: number;

  constructor(maxMissingFrames = 12, iouThreshold = 0.28, minHitsToConfirm = 1) {
    this.maxMissingFrames = maxMissingFrames;
    this.iouThreshold = iouThreshold;
    this.minHitsToConfirm = minHitsToConfirm;
  }

  public reset(): void {
    this.tracks.clear();
    this.nextTrackId = 1;
  }

  private getLowestAvailableId(): number {
    const existingIds = new Set(this.tracks.keys());
    let candidate = 1;
    while (existingIds.has(candidate)) {
      candidate++;
    }
    return candidate;
  }

  private smoothBox(prev: BBox, next: BBox, alpha = 0.22): BBox {
    // Check if movement is micro-jitter vs intentional head movement
    const dx = Math.abs(next.x1 - prev.x1);
    const dy = Math.abs(next.y1 - prev.y1);
    const effectiveAlpha = dx < 4 && dy < 4 ? 0.08 : dx < 10 && dy < 10 ? 0.18 : alpha;
    return {
      x1: effectiveAlpha * next.x1 + (1 - effectiveAlpha) * prev.x1,
      y1: effectiveAlpha * next.y1 + (1 - effectiveAlpha) * prev.y1,
      x2: effectiveAlpha * next.x2 + (1 - effectiveAlpha) * prev.x2,
      y2: effectiveAlpha * next.y2 + (1 - effectiveAlpha) * prev.y2,
    };
  }

  private matchCost(track: TrackedPerson, cand: DetectionCandidate): number {
    const iou = computeIoU(track.personBox, cand.personBox);
    if (iou >= this.iouThreshold) return iou;

    // Fallback normalized center distance for fast movement
    const tcx = (track.personBox.x1 + track.personBox.x2) / 2;
    const tcy = (track.personBox.y1 + track.personBox.y2) / 2;
    const ccx = (cand.personBox.x1 + cand.personBox.x2) / 2;
    const ccy = (cand.personBox.y1 + cand.personBox.y2) / 2;
    const diag = Math.hypot(boxWidth(track.personBox), boxHeight(track.personBox)) || 1;
    const distNorm = Math.hypot(tcx - ccx, tcy - ccy) / diag;
    return distNorm < 0.55 ? 0.25 * (1 - distNorm) : 0;
  }

  public update(candidates: DetectionCandidate[]): TrackedPerson[] {
    const highConf: Array<{ cand: DetectionCandidate; origIdx: number }> = [];
    const lowConf: Array<{ cand: DetectionCandidate; origIdx: number }> = [];
    candidates.forEach((c, origIdx) => {
      if (c.confidence >= 0.50) highConf.push({ cand: c, origIdx });
      else if (c.confidence >= 0.30) lowConf.push({ cand: c, origIdx });
    });

    const unmatchedTracks = new Set<number>(this.tracks.keys());

    const assignBatch = (batch: Array<{ cand: DetectionCandidate; origIdx: number }>, allowCreate: boolean) => {
      const pairs: Array<{ trackId: number; batchIdx: number; score: number }> = [];
      for (const trackId of unmatchedTracks) {
        const track = this.tracks.get(trackId)!;
        batch.forEach((item, idx) => {
          const score = this.matchCost(track, item.cand);
          if (score >= 0.12) {
            pairs.push({ trackId, batchIdx: idx, score });
          }
        });
      }
      pairs.sort((a, b) => b.score - a.score);

      const usedCands = new Set<number>();
      for (const p of pairs) {
        if (!unmatchedTracks.has(p.trackId) || usedCands.has(p.batchIdx)) continue;
        unmatchedTracks.delete(p.trackId);
        usedCands.add(p.batchIdx);

        const track = this.tracks.get(p.trackId)!;
        const { cand, origIdx } = batch[p.batchIdx];
        track.personBox = this.smoothBox(track.personBox, cand.personBox);
        track.faceBox =
          track.faceBox && cand.faceBox
            ? this.smoothBox(track.faceBox, cand.faceBox)
            : cand.faceBox; // Never reuse stale faceBox if null
        track.hasValidFace = cand.faceBox !== null;
        track.confidence = cand.confidence;
        track.faceConfidence = cand.faceConfidence;
        track.qualityScore = cand.qualityScore;
        track.visibleThisFrame = true;
        track.missingFrames = 0;
        track.ageFrames += 1;
        track.hits += 1;
        track.trackConfidence = Number(
          Math.min(1, 0.5 * cand.confidence + 0.3 * Math.min(1, track.hits / 5) + 0.2 * cand.qualityScore).toFixed(2)
        );
        track.lifecycleState = track.hits >= this.minHitsToConfirm ? 'confirmed' : 'tentative';
        track.candidateIndex = origIdx;
      }

      if (allowCreate) {
        batch.forEach((item, idx) => {
          if (usedCands.has(idx)) return;
          const { cand, origIdx } = item;
          const id = this.getLowestAvailableId();
          const hits = 1;
          this.tracks.set(id, {
            trackId: id,
            studentId: `Student ${String(id).padStart(2, '0')}`,
            personBox: cand.personBox,
            faceBox: cand.faceBox,
            confidence: cand.confidence,
            faceConfidence: cand.faceConfidence,
            qualityScore: cand.qualityScore,
            trackConfidence: Number((0.65 * cand.confidence).toFixed(2)),
            lifecycleState: hits >= this.minHitsToConfirm ? 'confirmed' : 'tentative',
            visibleThisFrame: true,
            hasValidFace: cand.faceBox !== null,
            missingFrames: 0,
            ageFrames: 1,
            hits,
            candidateIndex: origIdx,
          });
        });
      }
    };

    // Stage 1: High-confidence detections (can create new tracks)
    assignBatch(highConf, true);
    // Stage 2: Low-confidence detections (can only recover existing tracks)
    assignBatch(lowConf, false);

    // Handle unmatched / occluded tracks: clear faceBox immediately (no stale boxes!)
    for (const trackId of unmatchedTracks) {
      const track = this.tracks.get(trackId)!;
      track.visibleThisFrame = false;
      track.hasValidFace = false;
      track.faceBox = null;
      track.candidateIndex = -1;
      track.missingFrames += 1;
      track.lifecycleState = 'occluded';
      track.trackConfidence = Number(Math.max(0, track.trackConfidence * 0.85).toFixed(2));
      const maxAllowedMisses =
        track.hits < this.minHitsToConfirm ? Math.min(2, this.maxMissingFrames) : this.maxMissingFrames;
      if (track.missingFrames > maxAllowedMisses) {
        this.tracks.delete(trackId);
      }
    }

    return Array.from(this.tracks.values()).sort((a, b) => a.trackId - b.trackId);
  }
}

