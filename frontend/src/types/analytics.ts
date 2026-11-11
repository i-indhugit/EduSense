export interface StudentComponents {
  attention_contribution: number;
  emotion_contribution: number;
  participation_contribution: number;
  posture_contribution: number;
  drowsiness_penalty: number;
}

export interface StudentDebugInfo {
  student_id: string;
  raw_model_output?: number;
  raw_feature_values?: {
    gaze?: string;
    head_pose?: string;
    posture?: string;
    ear?: number;
    face_quality?: number;
    emotion?: string;
    drowsy_evidence_frame?: boolean;
  };
  temporal_values?: {
    raw_attention_score?: number;
    smoothed_attention_score?: number;
    drowsy_window_count?: number;
    drowsy_window_total?: number;
    drowsy_window_ratio_pct?: number;
    is_drowsy_consistent?: boolean;
  };
  reliability_values?: {
    face_quality?: number;
    confidence_pct?: number;
    is_valid_evidence?: boolean;
  };
  final_attention_score: number;
  final_state: string;
  reason_for_state: string;
}

export interface StudentAnalytics {
  student_id: string;
  confidence_pct?: number;
  face_quality?: number;
  attention_pct: number;
  attention_status?: string;
  final_state?: string;
  reason_for_state?: string;
  gaze_direction: string;
  head_pose?: string;
  emotion: string;
  is_drowsy: boolean;
  is_yawning: boolean;
  hand_raised: boolean;
  posture: string;
  participation_pct: number;
  engagement_pct: number;
  confusion_index: number;
  confusion_level: string;
  esi_score: number;
  status: string;
  ear_value?: number;
  components?: StudentComponents;
  debug_info?: StudentDebugInfo;
  xai?: {
    attention?: { reasons: string[]; confidence_pct?: number };
    engagement?: { reasons: string[]; summary_rationale?: string; attributions?: string[] };
    confusion?: { reasons: string[] };
    drowsiness?: { reasons: string[] };
    counterfactual?: { explanation?: string; [key: string]: any };
  };
}

export interface TeacherRecommendation {
  id: string;
  timestamp: string;
  type: string;
  severity: 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  title: string;
  message: string;
  action: string;
}

export interface ClassroomMetrics {
  eci_score: number;
  eci_status: string;
  avg_attention: number;
  avg_engagement: number;
  avg_participation: number;
  avg_confusion: number;
  drowsy_count: number;
  hand_raise_count: number;
  live_detected_persons?: number;
  active_tracks?: number;
  students?: StudentAnalytics[];
  recommendations?: TeacherRecommendation[];
}

export interface DashboardPayload {
  timestamp?: string;
  frame?: string;
  analytics?: ClassroomMetrics;
  classroom?: ClassroomMetrics;
  live_detected_persons?: number;
  active_tracks?: number;
  students?: StudentAnalytics[];
  recommendations?: TeacherRecommendation[];
}
