# Ethical considerations

Draft for the ethics section. Positions that are the team's to take rather than
something the evidence settles were marked **[team decision]** while open; all
four have now been decided and are recorded below as **Decision**, with the
evidence that supports each one kept alongside it.

The aim is to ground each concern in something measured in this project. Generic
ethics text — "AI systems may be biased", "human oversight is important" — is
true and worth nothing to a marker. Every claim below points at a number.

---

## 1. What this system would actually do

A classifier that labels an olive leaf photograph as healthy, *Aculus olearius*
infestation, or peacock spot. In deployment it would advise a grower on whether
and how to treat. That makes it decision support in a domain where the actions it
influences — spraying, pruning, removing trees — cost money, affect food
production, and put chemicals into the environment.

Two properties of the system shape everything else. It is **three-class and
closed**: a leaf with a fourth condition, or two conditions at once, is forced
into one of three labels, confidently. And it is **trained on curated images** —
detached leaves on plain backgrounds, every image resized to 800×600 — while
deployment would involve leaves photographed in a grove.

## 2. Asymmetric errors

The two error directions are not equally harmful, and the project's own numbers
let this be stated precisely rather than in the abstract.

**False negatives — a diseased leaf called healthy — are the dangerous error.**
An untreated infection spreads. The current model's per-class recall on the
official test split is 0.950 / 0.885 / 0.950 for MobileNetV2 and 0.895 / 0.970 /
0.954 for DenseNet121, so roughly one diseased leaf in ten is missed at the class
level. Compression makes this slightly worse in a specific place: full integer
quantisation drops `aculus_olearius` recall from 0.885 to 0.875 on MobileNetV2,
and from 0.970 to 0.855 on DenseNet121.

That last figure deserves emphasis. Choosing the most aggressive compression on
the wrong architecture costs **11.5 points of recall on a disease class** while
overall accuracy falls by less than half that. **Optimising a deployment for size
or speed using aggregate accuracy alone can silently degrade exactly the
capability that matters.** This is an ethical argument for reporting per-class
recall, not merely a methodological preference.

**False positives — a healthy leaf called diseased — cause unnecessary
treatment.** The cost is wasted pesticide, expense, and avoidable environmental
load, including effects on non-target organisms and resistance pressure. It is
the less severe error, but it is not free, and a system tuned to minimise misses
will produce more of them.

**The observed errors are between the two diseases.** In the annotated
explanation panels, both high-confidence mistakes are `aculus_olearius` ↔
`olive_peacock_spot` at 0.979 and 0.991 confidence; none is a diseased leaf
called healthy. This is the benign failure direction, and it is worth reporting
because it is not visible in macro-F1. It also has a practical consequence: the
likely harm from this model is **the wrong treatment**, not **no treatment** —
a mite infestation and a fungal infection require different interventions.

## 3. Overconfidence

Confidence is not calibrated, and the failures show it. `a245.jpg` is
misclassified at **0.979 confidence**; a second error sits at 0.991. A user
interface that displays a confidence score would present these as near-certain.

This is a design obligation rather than an observation.

**Decision.** No raw softmax output is presented to a grower as a probability,
and the system abstains below a confidence threshold, returning "unclear —
consult an expert" rather than a class. Any numeric confidence that is displayed
must be calibrated first.

That last clause is a commitment, not a description: no calibration was performed
in this project, and the two errors above at 0.979 and 0.991 are the evidence
that the uncalibrated scores cannot be shown as they stand. Until a calibration
step exists — temperature scaling on a held-out split is the obvious candidate,
and it is future work — the defensible interface shows a coarse band rather than
a number, plus the abstention path. Stating it this way keeps the decision
honest: the team has chosen what to display, and has not yet built the thing that
would make a displayed probability meaningful.

## 4. Bias and the limits of generalisation

The bias here is documented, quantified, and unusually concrete.

**Capture source predicts the label.** In the training split, all 690
`aculus_olearius` images have bare numeric filenames and no other class does;
`B-*` images are 99.8% `Healthy`. A classifier given nothing but the filename
prefix reaches **91.8% accuracy on training data** against a 38.2% baseline. The
same camera identity shows up in the pixels: within the `Healthy` class, the
brown fraction is 0.255 for `B-*` images and 0.004 for phone images — a factor of
64 driven by one photographer's wooden table.

**The trained model does not exploit it on average.** Grad-CAM attention outside
a leaf mask is 10.9% against a 15.5% background area, a ratio of 0.70; errors
show no greater background reliance than correct predictions.

**But it fails on individuals whose capture context is unusual.** `a245.jpg` is
a true `aculus_olearius` photographed against a grey-green surface with an `a*`
prefix — a source that is 88.8% `olive_peacock_spot` in training — and is
confidently misclassified.

The ethical content is in the third point. A system that is unbiased *on average*
can still fail systematically for a particular user: whoever owns the camera and
the background that the training set under-represents. If deployment reaches a
region, a cultivar, or a phone model absent from the training data, the people
harmed are a specific identifiable group, not a random sample. Aggregate fairness
metrics would not detect this.

**Undocumented population.** The dataset records no cultivar, no geography, and
covers one partial season (February–August 2019). Claims about performance for
olive growers in general are unsupported by construction, not merely unproven.

## 5. Human oversight

The failure analysis points to a specific control rather than a general appeal to
human-in-the-loop design.

Because the model's errors are between diseases rather than between diseased and
healthy, the most valuable human check is at the point of **choosing a
treatment**, not at the point of detection. A workflow where the model flags
candidate leaves and an agronomist confirms the specific condition before
treatment captures most of the value while bounding the realistic harm.

**Decision.** The system is framed as a **triage and decision-support tool** that
prioritises which trees an expert should inspect. It does not claim to diagnose,
and no output is presented as a diagnosis.

The alternative framing — a diagnostic aid proposing a condition for expert
confirmation — claims more than this project can support. Two measurements rule
it out. The label set is closed at three classes, so any condition outside it is
silently mapped onto one of the three. And the errors that do occur are confident
ones, between diseases rather than between diseased and healthy. A triage claim
survives both facts; a diagnostic claim does not. The rest of the ethics argument
below is built on this framing.

## 6. Environmental and resource considerations

The compression work has a direct environmental reading that is worth one
paragraph, since it is the project's own contribution rather than borrowed
argument.

Inference cost is not negligible at scale. Converting to TFLite and quantising
takes MobileNetV2 from 53.7 ms to 9.4 ms per image on CPU — a **5.7× reduction in
compute per inference** for 0.5 points of macro-F1. Running on-device rather than
in the cloud also removes per-image network transfer and avoids sending imagery
off the farm.

Set against this, training has its own footprint: the two baselines plus the
compression experiments amount to roughly six hours of CPU-saturated compute.
That is small, but the honest framing is that efficiency gains at inference are
what justify the training cost, and only if the model is actually used at volume.

## 7. Data provenance, licensing and privacy

**Privacy risk is low.** Leaf photographs contain no personal data. The residual
risks are indirect: EXIF metadata could carry GPS coordinates identifying a
grower's location, and aggregated disease reports could reveal commercially
sensitive information about a specific farm's health.

**Decision.** EXIF metadata is stripped by default from any image shared outside
the project. Where provenance is needed for research — capture device and
timestamp both matter to the bias analysis in section 4 — it is retained
separately from the image rather than embedded in it.

Note the current state accurately: the preparation pipeline already drops EXIF,
but only as a side effect of re-encoding through PIL, not as a stated guarantee.
The decision above turns an incidental behaviour into an intended one, which is
the difference between a pipeline that happens to be safe today and one that can
be relied on to stay safe.

**Licensing is clean but attribution is owed.** The dataset is published on Kaggle
under CC0 / public domain (Olive Leaf Image Dataset,
`habibulbasher01644/olive-leaf-image-dataset`). CC0 imposes no legal obligation to
attribute, but the source should be cited as an academic norm regardless.

**Provenance is incomplete, and that is a limitation worth stating.** The dataset
does not document who photographed the leaves, where, with what consent, or how
labels were assigned. The 119 duplicate pairs and 264 burst-linked images found
during preparation indicate the published train/test split was not constructed
with leakage in mind, which is a reason for caution about the curation generally.

## 8. Transparency about this work

**Reproducibility.** The compression and analysis pipelines run on the unmodified
course image with no packages installed, and every result regenerates from
`pipeline.sh`. Model artefacts and the prepared dataset are excluded from version
control but are reproducible from source.

**Corrections are disclosed, not silently applied.** The de-duplicated training
set produces a lower headline figure than the original split (0.9387 against
0.946). Reporting the higher number without the correction would have been the
easier choice and would have been wrong.

**AI use declaration.**

**Decision.** The declaration states the scope of AI assistance truthfully. Under
the TU Dublin Level 2 policy the submitted text is the team's own: AI-drafted
material is rewritten, its claims verified against the artefacts, and
responsibility for the result rests with the team. No AI draft is submitted as
final content.

The verifiable facts the declaration has to be consistent with: all seven commits
on `develop/model-compression` carry `Co-Authored-By` trailers recording AI
assistance, covering the compression pipeline, the corrections and bias
investigation, the dataset audit, the results tables, the report drafts, the
scratch and common-protocol experiments, and the integration notebook. That
record is permanent and public within the repository, so a declaration that
understated the scope would be contradicted by the repository itself.

Two things this does not settle, both of which the team must supply: the
per-person account of who used what, and the rewriting itself. Every claim in
these drafts points at a number in `RESULTS.md` or a file under `*/results/`, so
verification is a matter of checking each against its artefact rather than taking
it on trust — which is the form the Level 2 obligation takes here.

---

## 9. Summary of positions the team must settle

| # | Question | Why it cannot be answered from the evidence |
|---|---|---|
| 1 | Triage tool or diagnostic aid? | Determines what the system claims and therefore what standard of evidence applies |
| 2 | How is confidence presented, given the 0.979 error? | A UI design decision with an ethical dimension |
| 3 | Is metadata stripping stated as a policy? | Currently incidental; making it a commitment is a choice |
| 4 | What does the AI declaration say? | Institutional policy plus the team's own account of its process |
