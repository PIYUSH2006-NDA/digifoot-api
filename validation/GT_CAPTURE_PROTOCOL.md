# GT Capture Protocol — 10-pair validation set

Goal: 10 paired scans (clinic GT + iPhone app) for accuracy validation.

## Recruit
- 10 volunteers
- mix: 5 male / 5 female
- mix: 2 flat / 6 normal / 2 high arch
- mix shoe size: EU 36–46
- adults only, healthy feet

## Per session (~15 min/person)

```
1. consent form (template below)
2. clean foot (wet wipe, dry)
3. mark 4 anatomical points with washable marker
   - heel point (most posterior)
   - 1st metatarsal head (big-toe ball)
   - 5th metatarsal head (little-toe ball)
   - hallux tip (big toe end)
4. CLINIC SCAN (left foot first):
   - subject stands on glass plate / on scanner platform
   - capture full foot weight-bearing
   - export .ply in mm units
   - save: clinic_<id>_L.ply
5. iPHONE APP SCAN immediately after:
   - same pose, same load
   - run full pipeline
   - save raw zip + final .obj
6. repeat for right foot
7. measure foot manually with calipers as sanity:
   - length (heel to longest toe)
   - ball width
   - record in csv
```

## Consent form template (short)

```
I, ___________, agree to:
- have my feet 3D-scanned with a clinical scanner and an iPhone app
- have these scans used to validate the accuracy of [Solely] software
- have my anonymized scan data stored for up to 5 years for ML evaluation
I understand:
- no identifying info beyond shoe size, gender, age range is kept
- I can withdraw consent and request deletion at any time
Signed: ______ Date: ______
```

## Output structure

```
validation_set/
  pair_001/
    clinic_L.ply     clinic_R.ply
    iphone_L.obj     iphone_R.obj
    iphone_L_raw.zip iphone_R_raw.zip
    manual_measurements.csv
    meta.json   (gender, age_range, arch_type, shoe_size_eu)
  pair_002/ ... pair_010/
```

## Run validation

```bash
for i in 001 002 ... 010; do
  python validate_accuracy.py \
    --pred validation_set/pair_$i/iphone_L.obj \
    --gt   validation_set/pair_$i/clinic_L.ply \
    --report validation_set/pair_$i/report.json \
    --heatmap validation_set/pair_$i/heatmap.ply
done
python summarize_validation.py validation_set/
```

## Targets

- median <= 1.0 mm
- p95 <= 2.0 mm
- F@1mm >= 0.85
- F@2mm >= 0.97

If failing: tune TSDF voxel, frame count, ICP iterations.
