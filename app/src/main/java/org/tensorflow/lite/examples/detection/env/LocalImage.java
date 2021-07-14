package org.tensorflow.lite.examples.detection.env;

import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;

public class LocalImage {
    public LocalImage(String name, SimilarityClassifier.Recognition rec) {
        this.name = name;
        this.rec = rec;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public SimilarityClassifier.Recognition getRec() {
        return rec;
    }

    public void setRec(SimilarityClassifier.Recognition rec) {
        this.rec = rec;
    }

    String name;
    SimilarityClassifier.Recognition rec;
}
