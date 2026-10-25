"""Pixel-free probes (Table 2): can the category be predicted from acquisition metadata alone?"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder


def run_probes(df, num_classes):
    chance = 1.0 / num_classes
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    y = df.label.values

    def probe(X):
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1)
        return round(float(cross_val_score(rf, np.asarray(X, float), y, cv=cv).mean()), 3)

    dev = OrdinalEncoder().fit_transform(pd.DataFrame({"a": df.Make.fillna("?").astype(str),
                                                       "b": df.Model.fillna("?").astype(str)}))
    feats = [("Capture timestamp", np.c_[df.ts.astype("int64") // 10 ** 9, df.ts.dt.hour, df.ts.dt.dayofweek]),
             ("Time of day (date removed)", np.c_[df.ts.dt.hour + df.ts.dt.minute / 60]),
             ("Capture device", dev),
             ("Image dimensions", df[["w", "h"]].values)]
    t2 = pd.DataFrame([{"feature_set": n, "accuracy": probe(X)} for n, X in feats])
    t2["multiple_of_chance"] = (t2.accuracy / chance).round(1)
    return t2, chance
