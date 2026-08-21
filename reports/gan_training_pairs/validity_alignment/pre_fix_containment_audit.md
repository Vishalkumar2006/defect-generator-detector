# G1.3a discriminator-view validity alignment audit

- Status: **FAIL_CONTAINMENT**
- Requested pairs: 128
- Runtime seconds: 121.65419129999646
- Sampling rate: 1.0521626803991893
- Corrective alignment executed: False
- Samples below full containment: 20
- Support inside real_valid: minimum 0.9595225739491438, mean 0.9979195047754319, full-sample rate 0.84375
- Support inside fake_valid: minimum 1.0, mean 1.0, full-sample rate 1.0
- Support inside joint_valid: minimum 0.9595225739491438, mean 0.9979195047754319, full-sample rate 0.84375
- Original validity asymmetry: {'minimum': 0.0, 'mean': 0.17515826225280762, 'maximum': 0.6317520141601562}
- Aligned validity asymmetry: None
- Maximum difference outside joint validity: None
- Maximum native-valid mutation: None
- Padding-only equality rate: None
- Generator gradient coverage: None
- Validation rows loaded: 0
- Official-test rows loaded: 0
- Training steps: 0
- Materialized generated training images: 0

## Target-contact strata

`{"bottom": 13, "bottom+left": 13, "bottom+right": 12, "left": 13, "left+right": 12, "none": 13, "right": 13, "top": 13, "top+left": 13, "top+right": 13}`

## Containment failures

- `train-10352:0:0` on `train-12022` (none): containment `{"fake_valid": 1.0, "joint_valid": 0.9603612644254892, "real_valid": 0.9603612644254892}`; support outside real validity 79; canonical pixels outside 0; outside bbox `{"bottom": 287, "left": 204, "right": 206, "top": 256}`
- `train-10352:0:0` on `train-10984` (none): containment `{"fake_valid": 1.0, "joint_valid": 0.9595225739491438, "real_valid": 0.9595225739491438}`; support outside real validity 78; canonical pixels outside 0; outside bbox `{"bottom": 104, "left": 143, "right": 145, "top": 73}`
- `train-10382:0:0` on `train-12022` (left): containment `{"fake_valid": 1.0, "joint_valid": 0.9941240478781284, "real_valid": 0.9941240478781284}`; support outside real validity 81; canonical pixels outside 68; outside bbox `{"bottom": 406, "left": 12, "right": 12, "top": 326}`
- `train-11775:0:0` on `train-11400` (right): containment `{"fake_valid": 1.0, "joint_valid": 0.9935356942102305, "real_valid": 0.9935356942102305}`; support outside real validity 23; canonical pixels outside 8; outside bbox `{"bottom": 79, "left": 242, "right": 242, "top": 57}`
- `train-11612:0:0` on `train-10684` (right): containment `{"fake_valid": 1.0, "joint_valid": 0.9980095541401274, "real_valid": 0.9980095541401274}`; support outside real validity 30; canonical pixels outside 16; outside bbox `{"bottom": 39, "left": 240, "right": 240, "top": 10}`
- `train-10585:0:0` on `train-10773` (right): containment `{"fake_valid": 1.0, "joint_valid": 0.9963290008929457, "real_valid": 0.9963290008929457}`; support outside real validity 37; canonical pixels outside 26; outside bbox `{"bottom": 194, "left": 240, "right": 240, "top": 158}`
- `train-12314:0:0` on `train-11781` (top+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9873862696443342, "real_valid": 0.9873862696443342}`; support outside real validity 61; canonical pixels outside 54; outside bbox `{"bottom": 60, "left": 240, "right": 240, "top": 0}`
- `train-11957:0:0` on `train-10424` (top+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9622291021671827, "real_valid": 0.9622291021671827}`; support outside real validity 61; canonical pixels outside 56; outside bbox `{"bottom": 60, "left": 239, "right": 239, "top": 0}`
- `train-11195:0:0` on `train-11630` (bottom+left): containment `{"fake_valid": 1.0, "joint_valid": 0.9863674147963425, "real_valid": 0.9863674147963425}`; support outside real validity 82; canonical pixels outside 71; outside bbox `{"bottom": 511, "left": 13, "right": 13, "top": 430}`
- `train-11506:0:0` on `train-10662` (bottom+left): containment `{"fake_valid": 1.0, "joint_valid": 0.9827586206896551, "real_valid": 0.9827586206896551}`; support outside real validity 23; canonical pixels outside 18; outside bbox `{"bottom": 511, "left": 10, "right": 10, "top": 489}`
- `train-11506:0:0` on `train-10114` (bottom+left): containment `{"fake_valid": 1.0, "joint_valid": 0.9822150363783346, "real_valid": 0.9822150363783346}`; support outside real validity 22; canonical pixels outside 17; outside bbox `{"bottom": 511, "left": 14, "right": 14, "top": 490}`
- `train-11195:0:0` on `train-10932` (bottom+left): containment `{"fake_valid": 1.0, "joint_valid": 0.9847789824854045, "real_valid": 0.9847789824854045}`; support outside real validity 73; canonical pixels outside 63; outside bbox `{"bottom": 511, "left": 14, "right": 14, "top": 439}`
- `train-12314:0:0` on `train-11141` (bottom+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9886662286465178, "real_valid": 0.9886662286465178}`; support outside real validity 69; canonical pixels outside 62; outside bbox `{"bottom": 511, "left": 238, "right": 238, "top": 443}`
- `train-12301:0:0` on `train-11498` (bottom+right): containment `{"fake_valid": 1.0, "joint_valid": 0.990943669625068, "real_valid": 0.990943669625068}`; support outside real validity 50; canonical pixels outside 38; outside bbox `{"bottom": 490, "left": 242, "right": 242, "top": 441}`
- `train-11214:0:0` on `train-11389` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9875019859132553, "real_valid": 0.9875019859132553}`; support outside real validity 236; canonical pixels outside 226; outside bbox `{"bottom": 261, "left": 235, "right": 235, "top": 26}`
- `train-10421:0:0` on `train-11464` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9948682860075265, "real_valid": 0.9948682860075265}`; support outside real validity 30; canonical pixels outside 20; outside bbox `{"bottom": 332, "left": 238, "right": 238, "top": 303}`
- `train-10907:0:0` on `train-10111` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9970637251011825, "real_valid": 0.9970637251011825}`; support outside real validity 74; canonical pixels outside 53; outside bbox `{"bottom": 167, "left": 234, "right": 234, "top": 94}`
- `train-10421:0:0` on `train-10942` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9948746110195863, "real_valid": 0.9948746110195863}`; support outside real validity 28; canonical pixels outside 18; outside bbox `{"bottom": 367, "left": 21, "right": 21, "top": 340}`
- `train-10421:0:0` on `train-11287` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9949596774193549, "real_valid": 0.9949596774193549}`; support outside real validity 30; canonical pixels outside 20; outside bbox `{"bottom": 426, "left": 15, "right": 15, "top": 397}`
- `train-10907:0:0` on `train-10786` (left+right): containment `{"fake_valid": 1.0, "joint_valid": 0.9972008658654923, "real_valid": 0.9972008658654923}`; support outside real validity 75; canonical pixels outside 54; outside bbox `{"bottom": 379, "left": 17, "right": 17, "top": 305}`
