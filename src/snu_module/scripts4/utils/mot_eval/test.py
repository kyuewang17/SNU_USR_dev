import motmetrics as mm
import numpy as np

# Create an accumulator that will be updated during each frame
acc = mm.MOTAccumulator(auto_id=True)

# Call update once for per frame. For now, assume distances between
# frame objects / hypotheses are given.
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)

print(acc.mot_events)
print("\n")

frameid = acc.update(
    [1, 2],
    [1],
    [
        [0.2],
        [0.4]
    ]
)

print(acc.mot_events.loc[frameid])
print("\n")

frameid = acc.update(
    [1, 2],
    [1, 3],
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
)

frameid2 = acc.update(
    [1, 2],
    [1, 3],
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
)

print(acc.mot_events.loc[frameid])
print("\n")

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=["num_frames", "mota", "motp"], name="acc")
print(summary)
print("\n")

summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=["num_frames", "mota", "motp", "idf1", "recall", "precision"],
    names=['full', 'part'],
    generate_overall=True
)

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
print("\n")


# Object related points
o = np.array([
    [1., 2],
    [2., 2],
    [3., 2],
])

# Hypothesis related points
h = np.array([
    [0., 0],
    [1., 1],
])

C = mm.distances.norm2squared_matrix(o, h, max_d2=5.)


a = np.array([
    [0, 0, 1, 2],    # Format X, Y, Width, Height
    [0, 0, 0.8, 1.5],
])

b = np.array([
    [0, 0, 1, 2],
    [0, 0, 1, 1],
    [0.1, 0.2, 2, 2],
])
mm.distances.iou_matrix(a, b, max_iou=0.5)


if __name__ == "__main__":
    pass
