# MS-COCO 2017 Phone Distraction Dataset Benchmark
**Evaluation Date**: 2026-11-15  
**License**: Creative Commons Attribution 4.0 International (CC-BY 4.0) — 100% Free & Open Access  
**Dataset**: MS-COCO Validation Set (val2017)  

## Key Benchmark Statistics
| Metric | Value |
| :--- | :--- |
| **Total Images in Benchmark Split** | 5,000 |
| **Total Ground-Truth Cell Phone Annotations** | 262 |
| **Unique Images with Ground-Truth Phones** | 214 |
| **Images with Co-occurring Person + Phone** | 170 |
| **Person-Associated Phone Usage Rate** | 65.6% (172/262) |
| **MediaPipe Object Detector Category** | Category ID `77`: `'cell phone'` |
| **Spatial Fusion Validator** | `isPhoneAssociatedWithPerson()` |
| **Primary Classroom Trigger State** | `PHONE DISTRACTED` (Vivid Purple `#a855f7`) |

## Evaluation Samples Integrated into EduSense Dashboard
1. **Sample 1 (`000000332570.jpg`)**: Single student actively holding and using mobile phone.
2. **Sample 2 (`000000459272.jpg`)**: Multi-student classroom cohort spatial tracking.
3. **Sample 3 (`000000537991.jpg`)**: Study desk scenario isolating cell phone from laptop distractor screens.
4. **Sample 4 (`000000212166.jpg`)**: Upright attentive baseline student with board-directed focus.
